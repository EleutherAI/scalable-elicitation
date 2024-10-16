import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb
from w2s.loss import CustomLossTrainer
from w2s.sft_utils import (
    AccuracyStoppingCallback,
    EarlyStoppingCallback,
    assert_type,
    clear_mem,
    compute_acc_and_auroc,
    delete_all_ckpts,
    gather_hiddens,
    get_gpu_mem_used,
    move_best_ckpt,
)


def is_sft_cached(output_dir: Union[str, Path]) -> bool:
    save_path = Path(output_dir)
    results_path = save_path / "config.json"
    return results_path.exists() and (save_path / "best-ckpt").exists()


def prepare_for_trainer(
    ds: Union[DatasetDict, Dataset], tokenizer, discard_other_cols=True
):
    keep_cols = {"labels", "input_ids", "attention_mask"}

    def preprocess(exs):
        out = tokenizer(
            exs["txt"], truncation=True, max_length=tokenizer.model_max_length
        )
        if any(len(ids) == tokenizer.model_max_length for ids in out["input_ids"]):
            raise ValueError(
                "Text was truncated during tokenization."
                " Increase max_length or use a different model."
            )
        if discard_other_cols:
            return {k: v for k, v in out.items() if k in keep_cols}
        return out

    ds.reset_format()
    columns_names = set(
        ds.column_names
        if isinstance(ds, Dataset)
        else next(iter(ds.values())).column_names
    )
    ds = ds.map(
        preprocess,
        batched=True,
        remove_columns=list(columns_names - keep_cols) if discard_other_cols else None,
    )
    return ds


def lm_sft(
    ds_dict: DatasetDict,
    model,
    tokenizer,
    train_args: TrainingArguments,
    loss: str,
    cfg: dict,
    store_post_hiddens: bool = False,
    store_pre_hiddens: bool = False,
    predict_dict: Union[None, Dict, DatasetDict] = None,
    resume_from_checkpoint: Optional[str] = None,
    save: bool = True,
    target_accuracy: float | None = None,
    early_stopping_multiplier: float | None = None,
) -> Trainer:
    """
    ds_dict: DatasetDict with splits for train, val, test,
        with columns "txt" and "labels"
    model: model for Sequence Classification
    train_args: TrainingArguments with the training hyperparameters
    loss: a string indicating the loss function to use
    store_pre_hiddens: whether to store the hiddens (all layers,
        final token position, on train set) before training
    store_post_hiddens: whether to store the hiddens after training
    cfg: a dictionary containing all the relevant details for reproducibility.
        This will be updated with your train_args and model_cfg before saving.

    This function trains a model on ds_dict["train"], uses ds_dict["val"] for early stopping,
        and evaluates on ds_dict["test"].
    It also optionally predicts on ds_dict["predict"] and saves the predictions.
    """
    save_dir = Path(train_args.output_dir)
    train_args.run_name = (
        train_args.run_name.replace("=", "_") if train_args.run_name else "default"
    )

    clear_mem()
    print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before training")

    ds_dict = assert_type(DatasetDict, prepare_for_trainer(ds_dict, tokenizer))
    if train_args.load_best_model_at_end or early_stopping_multiplier is not None:
        callbacks: list = [
            EarlyStoppingCallback(
                multiplier=early_stopping_multiplier
                if early_stopping_multiplier is not None
                else 1.0
            )
        ]
    else:
        train_args.metric_for_best_model = None
        callbacks = []

    if target_accuracy is not None:
        callbacks.append(AccuracyStoppingCallback(target_accuracy))

    cls = CustomLossTrainer
    trainer = cls(
        loss_name=loss,
        resume_from_optimizer_checkpoint=resume_from_checkpoint,
        args=train_args,
        compute_metrics=compute_acc_and_auroc,
        data_collator=DataCollatorWithPadding(
            tokenizer,
            padding="longest",
        ),  # NOTE: this could mess up some datasets
        eval_dataset={
            k: ds_dict[k] for k in {"val", "test"}.intersection(ds_dict.keys())
        },
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_dict["train"],
        callbacks=callbacks,
    )

    results_path = save_dir / "config.json"
    if is_sft_cached(train_args.output_dir):
        print(
            f"Results for sft run already exist at {results_path}. "
            "Skipping training and evaluation. Loading the saved model."
        )
        trainer.state.best_model_checkpoint = str(save_dir / "best-ckpt")
        trainer._load_best_model()

    else:
        # store pre hiddens
        if store_pre_hiddens:
            print("Gathering hiddens")
            hiddens = gather_hiddens(model, ds_dict["train"])
            torch.save(hiddens, save_dir / "pre_hiddens.pt")

        # train
        trainer.train()

        # evaluate on test dataset
        if "test" in ds_dict:
            eval_results = trainer.evaluate(ds_dict["test"])  # type: ignore

            # save results
            with open(results_path, "w") as f:
                json.dump(eval_results, f, indent=2)
        move_best_ckpt(trainer)

        # save config
        with open(save_dir / "config.json", "w") as f:
            cfg["train_args"] = train_args.to_dict()
            json.dump(cfg, f, indent=2)
        wandb.config.update(cfg)

    # save predictions
    if predict_dict is not None and not (save_dir / "predictions").exists():
        for name, predict_ds in predict_dict.items():
            if (save_dir / "predictions" / name).exists():
                print(f"Predictions for {name} already exist. Skipping.")
                continue
            ready_ds = prepare_for_trainer(
                predict_ds, tokenizer, discard_other_cols=True
            )
            print("Gathering predictions for", name)
            pred_logits = torch.from_numpy(trainer.predict(ready_ds).predictions)  # type: ignore
            preds = pred_logits.softmax(-1).tolist()
            pred_ds = predict_ds.add_column("soft_pred", preds)  # type: ignore
            pred_ds.save_to_disk(str(save_dir / "predictions" / name))

    # save hiddens
    if store_post_hiddens and not is_sft_cached(train_args.output_dir):
        print("Gathering hiddens")
        hiddens = gather_hiddens(model, ds_dict["train"])
        torch.save(hiddens, save_dir / "post_hiddens.pt")

    wandb.finish()
    if not save:
        delete_all_ckpts(trainer)

    return trainer
