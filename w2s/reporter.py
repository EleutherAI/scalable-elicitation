from __future__ import annotations

import random
from abc import ABC
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from peft.tuners.lora.layer import LoraLayer
from torch import nn, optim
from tqdm import tqdm
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from w2s.ds_registry import encode_choice
from w2s.metrics import roc_auc
from w2s.model import LMPredictor, Predictor, TransformerPredictor
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

        # ensure that weak_ds and oracle are disjoint
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


class FewShotReporter(Reporter):
    strong_model: LMPredictor

    def __init__(
        self,
        num_weak: int,
        num_oracle: int,
        targets: tuple[str, str] = ("0", "1"),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_weak = num_weak
        self.num_oracle = num_oracle
        self.weak_ids_used = []
        self.oracle_ids_used = []
        self.targets = targets
        assert self.input_col == "txt", "FewShotReporter only supports input_col='txt'"
        assert isinstance(self.strong_model, LMPredictor)

    def get_dataset(self):
        weak_ds = self.weak_ds.shuffle().select(range(self.num_weak))
        self.weak_ids_used.extend(weak_ds["id"])
        all_oracle_ids = (
            self.oracle.get_inputs()["id"].values.tolist()
            if len(self.oracle.get_inputs()) > 0
            else []
        )
        ids = random.sample(all_oracle_ids, self.num_oracle)
        oracle_ds = Dataset.from_pandas(
            self.oracle.query_ids(ids), preserve_index=False
        )
        self.oracle_ids_used.extend(oracle_ds["id"])

        return concatenate_datasets(
            [
                ds_with_labels(weak_ds, "soft_pred"),
                ds_with_labels(oracle_ds, "soft_label"),
            ]
        ).shuffle()

    def fit(self):
        self.few_shot_ds = self.get_dataset()
        print(
            make_few_shot_prefix(self.few_shot_ds.shuffle(), self.targets)
        )  # TODO: remove
        return self

    def __call__(self, inputs: list[str]) -> torch.Tensor:
        fs_prefix = make_few_shot_prefix(self.few_shot_ds.shuffle(), self.targets)
        if isinstance(inputs, str):
            inputs = [inputs]

        logodds = []
        with torch.no_grad():
            for txt in tqdm(inputs, desc="Few-shot inference", total=len(inputs)):
                prompt = fs_prefix + txt + "\n"
                encodings = self.strong_model.tokenizer(prompt, return_tensors="pt").to(
                    self.strong_model.transformer.device
                )
                logits = self.strong_model.transformer(**encodings).logits.squeeze(0)

                # Note that the LM's performance might be poorer if "\n" and "0" (target)
                # are merged into a single "\n0" token in the few-shot prompt. For Llama's,
                # GPT2's, and pythia's tokenizers, they are not merged, as desired.
                target_toks = [
                    encode_choice(self.targets[0], self.strong_model.tokenizer),
                    encode_choice(self.targets[1], self.strong_model.tokenizer),
                ]
                target_logits = logits[encodings.input_ids.shape[1] - 1, target_toks]
                logodds.append(target_logits.diff(dim=-1).item())
        return torch.tensor(logodds)

    def to_dict(self):
        return {
            "method": self.__class__.__name__,
            "num_weak": len(set(self.weak_ids_used)),
            "num_oracle": len(set(self.oracle_ids_used)),
            "num_weak_nonunique": len(self.weak_ids_used),
            "num_oracle_nonunique": len(self.oracle_ids_used),
            "targets": self.targets,
        }


class FewShotPromptedSFTReporter(Reporter):
    """
    This reporter uses oracle labels in a few-shot prompt
    and weak labels to train the classification model when given this
    few-shot prompt
    """

    strong_model: TransformerPredictor

    def __init__(
        self,
        num_oracle: int,
        train_cfg: SftStage,
        targets: tuple[str, str] = ("0", "1"),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_oracle = num_oracle
        self.weak_ids_used = []
        self.oracle_ids_used = []
        self.targets = targets
        self.train_cfg = train_cfg
        assert (
            self.input_col == "txt"
        ), "FewShotPromptedSFTReporter only supports input_col='txt'"
        assert isinstance(self.strong_model, TransformerPredictor)
        assert self.train_cfg.type == "weak"
        self.few_shot_ds = self.get_few_shot_dataset()

    def get_few_shot_dataset(self):
        all_oracle_ids = (
            self.oracle.get_inputs()["id"].values.tolist()
            if len(self.oracle.get_inputs()) > 0
            else []
        )
        ids = random.sample(all_oracle_ids, self.num_oracle)
        oracle_ds = Dataset.from_pandas(
            self.oracle.query_ids(ids), preserve_index=False
        )
        self.oracle_ids_used.extend(oracle_ds["id"])

        return ds_with_labels(oracle_ds, "soft_label")

    def prefix_weak_ds(self):
        self.weak_ds = self.weak_ds.map(
            lambda x: {
                "txt": make_few_shot_prefix(self.few_shot_ds.shuffle(), self.targets)
                + x["txt"]
                + "\n"
            }
        )

    def fit(self):
        self.prefix_weak_ds()

        print("\n\033[32m [Training on weak labels] \033[0m")  # green text
        self.train_cfg.run(self, None)

        return self

    def __call__(self, inputs: list[str]) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions
        """
        inputs = [
            make_few_shot_prefix(self.few_shot_ds.shuffle(), self.targets) + x + "\n"
            for x in inputs
        ]
        predict_ds = prepare_for_trainer(
            Dataset.from_dict({self.input_col: inputs}), self.strong_model.tokenizer
        )
        # turn off wandb logging in trainer
        targs = self.train_cfg.train_args.copy()
        targs["report_to"] = "none"
        targs["output_dir"] = "tmp"
        targs["run_name"] = "tmp"
        trainer = Trainer(
            args=TrainingArguments(**targs),
            data_collator=DataCollatorWithPadding(
                self.strong_model.tokenizer,
                max_length=self.strong_model.cfg.max_ctx,
                padding="max_length",
            ),  # NOTE: this could mess up some datasets
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
        )
        pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)  # type: ignore # noqa
        return pred_logits.diff(dim=-1).squeeze()

    def to_dict(self):
        return {
            "method": self.__class__.__name__,
            "num_weak": len(set(self.train_cfg.weak_ids_used)),
            "num_oracle": len(set(self.oracle_ids_used)),
            "num_weak_nonunique": len(self.train_cfg.weak_ids_used),
            "num_oracle_nonunique": len(self.oracle_ids_used),
            "train_cfg": self.train_cfg.to_dict(),
            "targets": self.targets,
        }


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
        reporter: ModularSftReporter | FewShotPromptedSFTReporter,
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
        reporter: ModularSftReporter | FewShotPromptedSFTReporter,
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
    ):
        super().__init__(weak_ds, oracle, test_ds, strong_model, input_col)
        self.stages = stages
        self.test_ds = ds_with_labels(test_ds)

        assert input_col == "txt", "Only LM SFT is supported"

    def fit(self) -> ModularSftReporter:
        optimizer_checkpoint = None
        for i, stage_config in enumerate(self.stages):
            print(f"\n\033[32m [Stage {i}] \033[0m")  # green text
            optimizer_checkpoint = stage_config.run(self, optimizer_checkpoint)

        return self

    def to_dict(self) -> dict:
        return {
            "method": self.__class__.__name__,
            "stages": [s.to_dict() for s in self.stages],
            "model": self.strong_model.to_dict(),
            "num_weak": len(set.union(*(set(s.weak_ids_used) for s in self.stages))),
            "num_weak_nonunique": sum(len(s.weak_ids_used) for s in self.stages),
            "num_oracle": len(set(self.oracle.ids_labeled)),
            "num_oracle_nonunique": len(self.oracle.ids_labeled),
        }

    def __call__(self, inputs: list) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions
        """
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
                padding="max_length",
            ),  # NOTE: this could mess up some datasets
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
        )
        pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)  # type: ignore # noqa
        return pred_logits.diff(dim=-1).squeeze()


class DivDisSftReporter(Reporter):
    """
    Diversify and Disambiguate finetuning: https://arxiv.org/abs/2202.03418

    Diversification:
    - Train multiple heads with different random initializations
    - Use trusted (weak) data with xent loss and diversify on unlabeled target (oracle) data
    Disambiguation:
    - Use oracle examples to select a head
    - Calibrate that head with Platt scaling
    """

    strong_model: TransformerPredictor
    best_head: int
    bias = torch.nn.Parameter(torch.tensor(0.0))
    scale = torch.nn.Parameter(torch.tensor(1.0))

    def __init__(
        self,
        weak_ds: Dataset,
        oracle: Oracle,
        test_ds: Dataset,
        strong_model: TransformerPredictor,
        input_col: str = "txt",
        save_dir: str = "./results",
        **kwargs,
    ):
        super().__init__(weak_ds, oracle, test_ds, strong_model, input_col)
        self.test_ds = ds_with_labels(test_ds)
        self.weak_train_args = kwargs
        self.weak_train_args[
            "run_name"
        ] = f"div_{self.weak_train_args.get('run_name', 'default')}"
        self.weak_train_args["output_dir"] = str(Path(save_dir) / "div")

        assert input_col == "txt", "Only LM SFT is supported"

    def fit(self, max_queries: int) -> "DivDisSftReporter":
        # ### Diversification ###
        # we use label -1 for target data, and pass a custom loss function that deals
        # with -1 examples separately
        weak_ds = ds_with_labels(self.weak_ds, labels_column="soft_pred")
        train_target_ds = (
            Dataset.from_pandas(self.oracle.get_inputs(), preserve_index=False)
            .shuffle()
            .select(range(len(weak_ds)))  # NOTE: this is a hyperparameter
        )
        train_target_ds = train_target_ds.add_column(  # type: ignore
            "labels", [-1.0] * len(train_target_ds)
        ).cast(weak_ds.features)
        weak_ds = concatenate_datasets([weak_ds, train_target_ds])
        weak_ds_dict = DatasetDict(train=weak_ds, test=self.test_ds)

        self.div_trainer = lm_sft(
            ds_dict=weak_ds_dict,
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
            train_args=TrainingArguments(**self.weak_train_args),
            loss="divdis",
            store_pre_hiddens=False,
            store_post_hiddens=False,
            cfg=self.to_dict(),
            predict_dict=None,
        )

        # then disambiguate
        if max_queries > 0:
            oracle_ds = ds_with_labels(self.get_oracle_ds(max_queries))
            self._disambiguate(oracle_ds)
            self._platt_scale(oracle_ds)
        else:
            self.best_head = 0

        return self

    def get_oracle_ds(self, max_queries: int) -> Dataset:
        # Select examples according to the amount of disagreement between heads
        # Lee et al. use total distance between head predictions (hardened, I believe)
        # but we would prefer to also care about the confidence of disagreements
        # so we use the total cross entropy between every pair of heads

        all_oracle_inputs = Dataset.from_pandas(
            self.oracle.get_inputs(), preserve_index=False
        )

        print(
            "Selecting examples with average cross entropy between pairs of heads for training."
        )

        pred_logodds = self._call_all_heads(all_oracle_inputs["txt"])
        logprobs = torch.nn.functional.logsigmoid(pred_logodds)  # [b, h]
        log1mprobs = torch.nn.functional.logsigmoid(-pred_logodds)
        probs = logprobs.exp()
        # xent = -p * log(q) - (1-p) * log(1-q) for each pair p, q
        xents = -torch.einsum("bh,bg->bhg", probs, logprobs) - torch.einsum(
            "bh,bg->bhg", 1 - probs, log1mprobs
        )  # [b, h, h]
        avg_xents = xents.mean(dim=-1).mean(dim=-1)  # [b]

        uncertain_idxs = torch.multinomial(avg_xents, max_queries, replacement=False)

        oracle_ids = (
            [all_oracle_inputs["id"][idx] for idx in uncertain_idxs]
            if len(all_oracle_inputs) > 0
            else []
        )
        return Dataset.from_pandas(
            self.oracle.query_ids(oracle_ids), preserve_index=False
        ).shuffle()

    def _disambiguate(self, oracle_ds: Dataset) -> int:
        # get predictions from all heads
        pred_logits = self._call_all_heads(oracle_ds[self.input_col])

        # pick the head with the highest auroc on the oracle data
        labels = (torch.as_tensor(oracle_ds["labels"]) > 0.5).long()
        labels = labels.unsqueeze(-1).expand(-1, pred_logits.shape[-1])
        aurocs = roc_auc(labels, pred_logits)
        self.best_head = int(aurocs.argmax())
        return self.best_head

    def _platt_scale(self, oracle_ds: Dataset, max_iter: int = 100) -> None:
        """Fit the scale and bias terms to data with LBFGS.

        Args:
            oracle_ds: Dataset with columns ["txt", "labels"]
            max_iter: Maximum number of iterations for LBFGS.
        """
        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(torch.bfloat16).eps,
            tolerance_grad=torch.finfo(torch.bfloat16).eps,
        )

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy(
                torch.sigmoid(self(oracle_ds[self.input_col])),
                torch.as_tensor(oracle_ds["labels"]).float(),
            )

            loss.backward()
            return float(loss)

        opt.step(closure)

    def _call_all_heads(self, inputs: list) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions for all heads
        """
        predict_ds = prepare_for_trainer(
            Dataset.from_dict({self.input_col: inputs}), self.strong_model.tokenizer
        )
        # turn off wandb logging in trainer
        targs = self.weak_train_args.copy()
        targs["report_to"] = "none"
        targs["output_dir"] = "tmp"
        targs["run_name"] = "tmp"
        trainer = Trainer(
            args=TrainingArguments(**targs),
            data_collator=DataCollatorWithPadding(
                self.strong_model.tokenizer,
                max_length=self.strong_model.cfg.max_ctx,
                padding="max_length",
            ),  # NOTE: this could mess up some datasets
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
        )
        pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)  # type: ignore # noqa
        return pred_logits.diff(dim=-1).squeeze()

    def __call__(self, inputs: list) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions from its best head
        inputs: a list of strings
        """
        assert hasattr(self, "best_head"), "Must fit before calling"
        lo = self._call_all_heads(inputs)[..., self.best_head]
        return self.scale.to(lo.dtype).to(lo.device) * lo + self.bias.to(lo.dtype).to(
            lo.device
        )

    def to_dict(self) -> dict:
        return {
            "method": self.__class__.__name__,
            "weak_train_args": self.weak_train_args,
            "model": self.strong_model.to_dict(),
        }


class DivDisProbingReporter(Reporter):
    # optionally finetunes on trusted examples first
    ...


REPORTER_REGISTRY: dict[str, type[Reporter]] = {
    c.__name__: c
    for c in locals().values()
    if isinstance(c, type) and issubclass(c, Reporter) and c is not Reporter
}
