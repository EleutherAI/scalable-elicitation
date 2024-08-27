import random
from typing import Literal

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from w2s.ds_registry import concatenate_datasets, encode_choice
from w2s.model import LMPredictor, TransformerPredictor
from w2s.reporter import Reporter, SftStage
from w2s.sft import prepare_for_trainer
from w2s.utils import ds_with_labels, make_few_shot_prefix


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
        num_few_shot: int,
        few_shot_type: Literal["oracle", "weak"],
        train_stage: SftStage,
        targets: tuple[str, str] = ("0", "1"),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_few_shot = num_few_shot
        self.few_shot_type = few_shot_type
        self.weak_ids_used = []
        self.oracle_ids_used = []
        self.targets = targets
        self.train_stage = train_stage
        assert (
            self.input_col == "txt"
        ), "FewShotPromptedSFTReporter only supports input_col='txt'"
        assert isinstance(self.strong_model, TransformerPredictor)
        self.few_shot_ds = self.get_few_shot_dataset()

    def get_few_shot_dataset(self):
        if self.few_shot_type == "oracle":
            all_oracle_ids = (
                self.oracle.get_inputs()["id"].values.tolist()
                if len(self.oracle.get_inputs()) > 0
                else []
            )
            ids = random.sample(all_oracle_ids, self.num_few_shot)
            oracle_ds = Dataset.from_pandas(
                self.oracle.query_ids(ids), preserve_index=False
            )
            self.oracle_ids_used.extend(oracle_ds["id"])

            return ds_with_labels(oracle_ds, "soft_label")
        else:
            weak_ds = self.weak_ds.shuffle().select(range(self.num_few_shot))
            self.weak_ids_used.extend(weak_ds["id"])
            return ds_with_labels(weak_ds, "soft_pred")

    def prefix_train_ds(self):
        def fn(txt):
            return (
                make_few_shot_prefix(self.few_shot_ds.shuffle(), self.targets)
                + txt
                + "\n"
            )

        if self.train_stage.type == "weak":
            self.weak_ds = self.weak_ds.map(lambda x: {"txt": fn(x["txt"])})
        elif self.train_stage.type == "oracle":
            self.oracle._df["txt"] = self.oracle._df["txt"].apply(fn)

    def fit(self):
        self.prefix_train_ds()

        print("\n\033[32m [Training on weak labels] \033[0m")  # green text
        self.train_stage.run(self, None)  # type: ignore

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
        targs = self.train_stage.train_args.copy()
        targs["report_to"] = "none"
        targs["output_dir"] = "tmp"
        targs["run_name"] = "tmp"
        trainer = Trainer(
            args=TrainingArguments(**targs),
            data_collator=DataCollatorWithPadding(
                self.strong_model.tokenizer,
                max_length=self.strong_model.cfg.max_ctx,
                padding="longest",
            ),
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
        )
        pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)  # type: ignore # noqa
        return pred_logits.diff(dim=-1).squeeze()

    def to_dict(self):
        return {
            "method": self.__class__.__name__,
            "num_weak": len(
                set(self.train_stage.weak_ids_used) | set(self.weak_ids_used)
            ),
            "num_oracle": len(
                set(self.train_stage.oracle_ids_used) | set(self.oracle_ids_used)
            ),
            "num_weak_nonunique": len(self.train_stage.weak_ids_used)
            + len(self.weak_ids_used),
            "num_oracle_nonunique": len(self.train_stage.oracle_ids_used)
            + len(self.oracle_ids_used),
            "train_stage": self.train_stage.to_dict(),
            "targets": self.targets,
        }
