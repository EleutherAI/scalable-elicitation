from __future__ import annotations

import random
from abc import ABC
from typing import Literal, Optional

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft.tuners.lora.layer import LoraLayer
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from w2s.model import Predictor, TransformerPredictor
from w2s.sft import is_sft_cached, lm_sft, prepare_for_trainer
from w2s.sft_utils import get_gpu_mem_used
from w2s.utils import (
    assert_type,
    ds_with_labels,
    make_few_shot_prefix,
    uncertainty_sample,
)


class Oracle:
    _df: pd.DataFrame
    ids_labeled: list

    def __init__(self, gt_dataset: Dataset, input_col: str = "txt") -> None:
        assert (
            "id" in gt_dataset.column_names and "soft_label" in gt_dataset.column_names
        )
        assert "gt_soft_label" not in gt_dataset.column_names

        self._df = assert_type(pd.DataFrame, gt_dataset.to_pandas())
        self._df.set_index("id", inplace=True, drop=False)
        if "labels" in self._df.columns:
            self._df.drop(columns=["labels"], inplace=True)
        self.input_col = input_col

        self.ids_labeled = list()

    def query_id(self, id: str) -> float:
        """
        Get an oracle label for a single id, and note that it has been queried.
        """
        self.ids_labeled.append(id)
        return assert_type(float, self._df.loc[id]["soft_label"])

    def query_ids(self, ids: list) -> pd.DataFrame:
        """
        Get oracle labels for a list of ids, and note that they have been queried.
        """
        self.ids_labeled.extend(ids)
        return self._df.loc[ids]

    def get_inputs(self) -> pd.DataFrame:
        # remove soft_label from inputs
        return self._df.drop(columns=["soft_label", "hard_label"], inplace=False)

    def reset(self):
        self.ids_labeled = list()


class Reporter(ABC):
    """
    This is a reporter in the terminology of ELK
    https://www.lesswrong.com/posts/qHCDysDnvhteW7kRd/arc-s-first-technical-report-eliciting-latent-knowledge

    It is a method of eliciting latent knowledge from a strong model. E.g. finetuning.
    """

    weak_ds: Dataset
    oracle: Oracle
    test_ds: Dataset
    strong_model: Predictor

    def __init__(
        self,
        weak_ds: Dataset,
        oracle: Oracle,
        test_ds: Dataset,
        strong_model: Predictor,
        input_col: str = "txt",
        save_dir: str = "./results",
    ):
        """
        weak_ds: a dataset with columns ["id", input_col, "soft_pred"]
        oracle: an Oracle object
        strong_model: the model to elicit latent knowledge from
        input_col: the column in weak_ds that contains the input

        """
        assert input_col in weak_ds.column_names
        assert "soft_pred" in weak_ds.column_names
        assert "id" in weak_ds.column_names
        self.weak_ds = weak_ds

        self.oracle = oracle
        self.test_ds = test_ds
        self.strong_model = strong_model
        self.input_col = input_col
        self.save_dir = save_dir

        assert (
            set(weak_ds["id"]) & set(oracle.get_inputs()["id"]) == set()
        ), "Weak and oracle datasets must be disjoint"

    def fit(self) -> "Reporter":
        """
        max_queries: the maximum number of queries to the oracle
        """
        ...

    def __call__(self, inputs) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions
        """
        ...

    def to_dict(self) -> dict[str, str | int | float]:
        """A summary of the reporter that approximately uniquely identifies it.
        It should include a name and all the important hyperparameters."""
        ...


class SftStage:
    """
    The val set is included and the EarlyStoppingCallback is added if
    args["load_best_model_at_end"] is True or early_stopping_multiplier is not None.
    """

    modules_with_grad: Literal["all", "head", "body"]
    reinit_head: bool
    train_args: dict
    type: Literal["weak", "oracle"]
    size: int
    n_val: int
    n_test: int
    loss: Literal["xent", "logconf"]
    sampling: Literal["random", "most_confident_label", "least_confident_pred"]
    sample_temp: float
    reuse_optimizer_checkpoint: bool
    early_stopping_multiplier: float | None
    weak_ids_used: list
    oracle_ids_used: list

    def __init__(
        self,
        type: Literal["weak", "oracle"] = "weak",
        size: int = 1000,  # number of train + val examples
        sampling: Literal[
            "random", "most_confident_label", "least_confident_pred"
        ] = "random",
        n_val: int = 0,
        n_test: int = 0,
        modules_with_grad: Literal["all", "head", "body"] = "all",
        reinit_head: bool = False,
        sample_temp: float = 0.25,
        reuse_optimizer_checkpoint: bool = False,
        early_stopping_multiplier: float | None = None,
        loss: Literal["xent", "logconf"] = "xent",
        **kwargs,
    ):
        self.type = type
        self.size = round(size)
        self.sampling = sampling
        self.n_val = round(n_val)
        self.n_test = round(n_test)
        self.modules_with_grad = modules_with_grad
        self.reinit_head = reinit_head
        self.sample_temp = sample_temp
        self.reuse_optimizer_checkpoint = bool(reuse_optimizer_checkpoint)
        self.early_stopping_multiplier = early_stopping_multiplier
        self.loss = loss
        self.train_args = kwargs
        self.weak_ids_used = []
        self.oracle_ids_used = []

    def get_dataset(
        self,
        oracle: Oracle,
        weak_ds: Dataset,
        test_ds: Dataset,
        reporter: ModularSftReporter,
    ) -> DatasetDict:
        inputs = oracle.get_inputs() if self.type == "oracle" else weak_ds
        label_col = "soft_pred" if self.type == "weak" else "soft_label"

        if self.sampling == "random":
            idxs = random.sample(range(len(inputs)), k=self.size)  # without replacement
        elif self.sampling == "least_confident_pred":
            print("Selecting examples with highest reporter entropy for training.")
            pred_logodds = reporter(inputs["txt"])  # type: ignore
            probs = torch.nn.functional.sigmoid(pred_logodds)
            probs = torch.stack([1 - probs, probs], dim=-1)

            idxs = uncertainty_sample(
                probs, self.size, self.sample_temp, most_confident=False
            )
        elif self.sampling == "most_confident_label":
            print("Selecting examples with lowest label entropy for training.")
            probs = torch.softmax(
                torch.tensor(inputs[label_col], dtype=torch.float32), dim=-1
            )
            idxs = uncertainty_sample(
                probs, self.size, self.sample_temp, most_confident=True
            )
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

        if self.type == "oracle":
            ids = (
                [inputs["id"].iloc[int(idx)] for idx in idxs]  # type: ignore
                if len(inputs) > 0
                else []
            )
            train_ds = Dataset.from_pandas(oracle.query_ids(ids), preserve_index=False)
            self.oracle_ids_used.extend(train_ds["id"])
        else:
            train_ds = weak_ds.select(idxs)
            self.weak_ids_used.extend(train_ds["id"])

        ids_used = self.weak_ids_used if self.type == "weak" else self.oracle_ids_used
        if len(set(ids_used)) != self.size:
            print(
                f"WARNING: {self.type} stage requested {self.size} ids, "
                f"but {len(ids_used)} unique ids were used"
            )

        ds_dict = {"train": ds_with_labels(train_ds, labels_column=label_col)}
        if self.n_test > 0:
            ds_dict["test"] = ds_with_labels(
                test_ds.shuffle().select(range(self.n_test)), labels_column="soft_label"
            )
        if (
            self.train_args["load_best_model_at_end"]
            or self.early_stopping_multiplier is not None
        ):
            metric = self.train_args["metric_for_best_model"]
            assert metric.startswith("val_") or metric.startswith("eval_val_")
            assert self.n_val > 0
            # TODO: allow for more configurations for early stopping dataset, such at using oracle
            # for now we just split off a chunk of the train dataset
            ds_dict["val"] = ds_dict["train"].select(range(self.n_val))
            ds_dict["train"] = ds_dict["train"].select(
                range(self.n_val, len(ds_dict["train"]))
            )
        # print balance
        train_labs = (torch.tensor(ds_dict["train"]["labels"]) > 0.5).float()
        print(
            f"Train labels balance: {train_labs.mean()} (n={len(train_labs)} {self.type})"
        )
        return DatasetDict(**ds_dict)

    def run(
        self,
        reporter: ModularSftReporter,
        optimizer_checkpoint: Optional[str] = None,
    ) -> str:
        assert isinstance(reporter.strong_model, TransformerPredictor)
        if reporter.strong_model.cfg.enable_lora:
            # TODO: support models without `score` attribute
            lora_params = [
                (*m.lora_A.parameters(), *m.lora_B.parameters())
                for m in reporter.strong_model.transformer.modules()
                if isinstance(m, LoraLayer)
            ]
            lora_params = [p for params in lora_params for p in params]
            if self.modules_with_grad == "all":
                for p in lora_params:
                    p.requires_grad_()
                reporter.strong_model.transformer.score.requires_grad_()
            elif self.modules_with_grad == "head":
                reporter.strong_model.transformer.requires_grad_(False)
                reporter.strong_model.transformer.score.requires_grad_()
            elif self.modules_with_grad == "body":
                for p in lora_params:
                    if not p.is_floating_point():
                        print(f"Skipping parameter {p} with dtype {p.dtype}")
                    p.requires_grad_()
                reporter.strong_model.transformer.score.requires_grad_(False)
            else:
                raise ValueError(f"Unknown modules_with_grad: {self.modules_with_grad}")
        else:
            raise ValueError("Only Lora models are supported")

        if self.reinit_head:
            score_data = reporter.strong_model.transformer.score.weight.data
            score_data.normal_(0, 0.01 / score_data.shape[-1] ** 0.5)

        # we temporarily change the sampling method to avoid doing
        # inference for cached training run data selection
        # since the trained model is cached we don't actually use the sampled data
        actual_sampling = self.sampling

        if is_sft_cached(self.train_args["output_dir"]):
            self.sampling = "random"
        ds_dict = self.get_dataset(
            reporter.oracle, reporter.weak_ds, reporter.test_ds, reporter
        )
        if is_sft_cached(self.train_args["output_dir"]):
            self.sampling = actual_sampling
        train_args = self.train_args.copy()

        print(f"{get_gpu_mem_used() * 100}% of all GPU mem in use before training")

        lm_sft(
            ds_dict=ds_dict,
            model=reporter.strong_model.transformer,
            tokenizer=reporter.strong_model.tokenizer,
            train_args=TrainingArguments(**train_args),
            loss=self.loss,
            store_pre_hiddens=False,
            store_post_hiddens=False,
            cfg=reporter.to_dict(),
            predict_dict=None,
            resume_from_checkpoint=optimizer_checkpoint
            if self.reuse_optimizer_checkpoint
            else None,
            early_stopping_multiplier=self.early_stopping_multiplier,
        )

        return f"{train_args['output_dir']}/best-ckpt/optimizer.pt"

    def to_dict(self) -> dict:
        d = vars(self).copy()
        del d["weak_ids_used"]
        del d["oracle_ids_used"]
        d["num_weak"] = len(set(self.weak_ids_used))
        d["num_oracle"] = len(set(self.oracle_ids_used))
        d["num_weak_nonunique"] = len(self.weak_ids_used)
        d["num_oracle_nonunique"] = len(self.oracle_ids_used)
        return d


class ModularSftReporter(Reporter):
    strong_model: TransformerPredictor  # override type

    def __init__(
        self,
        weak_ds: Dataset,
        oracle: Oracle,
        test_ds: Dataset,  # for logging
        stages: list[SftStage],
        strong_model: TransformerPredictor,
        input_col: str = "txt",
        few_shot_ds: Dataset | None = None,
        few_shot_type: Literal["oracle", "weak"] | None = None,
        targets: tuple[str, str] = ("0", "1"),
    ):
        super().__init__(weak_ds, oracle, test_ds, strong_model, input_col)
        self.stages = stages
        self.test_ds = ds_with_labels(test_ds)
        self.few_shot_ds = few_shot_ds
        self.few_shot_type = few_shot_type
        assert (few_shot_ds is None) == (
            few_shot_type is None
        ), "incompatible few-shot args"
        assert input_col == "txt", "Only LM SFT is supported"
        self.targets = targets
        if few_shot_ds is not None:
            self.prefix_train_datasets()

    def fit(self) -> ModularSftReporter:
        optimizer_checkpoint = None
        for i, stage_config in enumerate(self.stages):
            print(f"\n\033[32m [Stage {i}] \033[0m")  # green text
            optimizer_checkpoint = stage_config.run(self, optimizer_checkpoint)

        return self

    def prefix_train_datasets(self):
        def fn(txt):
            assert self.few_shot_ds is not None
            return (
                make_few_shot_prefix(self.few_shot_ds.shuffle(), self.targets)
                + txt
                + "\n"
            )

        self.weak_ds = self.weak_ds.map(lambda x: {"txt": fn(x["txt"])})
        self.oracle._df["txt"] = self.oracle._df["txt"].apply(fn)

    def __call__(self, inputs: list) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions
        """
        if self.few_shot_ds is not None:
            inputs = [
                make_few_shot_prefix(self.few_shot_ds.shuffle(), self.targets)
                + x
                + "\n"
                for x in inputs
            ]
        predict_ds = prepare_for_trainer(
            Dataset.from_dict({self.input_col: inputs}), self.strong_model.tokenizer
        )
        # turn off wandb logging in trainer
        targs = self.stages[0].train_args.copy()
        targs["report_to"] = "none"
        targs["output_dir"] = "tmp"
        targs["run_name"] = "tmp"
        trainer = Trainer(
            args=TrainingArguments(**targs),
            data_collator=DataCollatorWithPadding(
                self.strong_model.tokenizer,
                max_length=self.strong_model.cfg.max_ctx,
                padding="longest",
            ),  # NOTE: this could silently truncate some datasets
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
        )
        pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)  # type: ignore # noqa
        return pred_logits.diff(dim=-1).squeeze()

    def to_dict(self) -> dict:
        if self.few_shot_ds is not None:
            fs_weak_ids = (
                set(self.few_shot_ds["id"]) if self.few_shot_type == "weak" else set()
            )
            fs_oracle_ids = (
                set(self.few_shot_ds["id"]) if self.few_shot_type == "oracle" else set()
            )
        else:
            fs_weak_ids, fs_oracle_ids = set(), set()
        return {
            "method": self.__class__.__name__,
            "stages": [s.to_dict() for s in self.stages],
            "model": self.strong_model.to_dict(),
            "num_weak": len(
                set.union(*(set(s.weak_ids_used) for s in self.stages)).union(
                    fs_weak_ids
                )
            ),
            "num_weak_nonunique": sum(
                len(s.weak_ids_used) for s in self.stages
            ),  # counted once per train stage
            "num_oracle": len(set(self.oracle.ids_labeled).union(fs_oracle_ids)),
            "num_oracle_nonunique": len(self.oracle.ids_labeled),
            "few_shot_type": self.few_shot_type,
            "n_few_shot": len(self.few_shot_ds) if self.few_shot_ds is not None else 0,
        }


REPORTER_REGISTRY: dict[str, type[Reporter]] = {
    c.__name__: c
    for c in locals().values()
    if isinstance(c, type) and issubclass(c, Reporter) and c is not Reporter
}
