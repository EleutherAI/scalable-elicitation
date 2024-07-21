from pathlib import Path
from typing import Union

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from fire import Fire
from transformers import TrainingArguments

from w2s.ds_registry import load_and_process_dataset
from w2s.model import MODEL_REGISTRY, ModelConfig, TransformerPredictor
from w2s.sft import lm_sft
from w2s.sft_config import set_default_args
from w2s.utils import ds_with_labels


def add_idx_col(ds: Union[DatasetDict, Dataset]) -> Union[DatasetDict, Dataset]:
    if isinstance(ds, DatasetDict):
        for split in ds:
            ds[split] = add_idx_col(ds[split])
        return ds
    else:
        ds = ds.add_column("idx", range(len(ds)))  # type: ignore
        return ds


def main(
    ds_name: str,
    model_name: str = "Qwen/Qwen1.5-0.5B",
    n_train: int = 8_000,
    n_val: int = 500,
    n_test: int = 5_000,
    n_predict: int = 50_000,
    results_folder=None,
    run_name: str | None = None,
    also_save_shuffled_error_labels: bool = False,
    disable_lora: bool = False,
    target_accuracy: float | None = None,
    **train_args,
):
    train_args["num_train_epochs"] = train_args.get("num_train_epochs", 3)
    train_args = set_default_args(train_args, model_name=model_name, run_name=run_name)

    model_last = model_name.split("/")[-1]
    if run_name is None:
        run_name = f"{ds_name}_{model_last}"
    if results_folder is None:
        results_folder = Path(__file__).parent / "results"
    output_dir = Path(results_folder) / run_name  # type: ignore

    if (output_dir / "weak_train").exists() and (output_dir / "weak_test").exists():
        print(f"\033[33m===== Weak labels already exist for {output_dir} =====\033[0m")
        return

    # load dataset
    source_ds = load_and_process_dataset(ds_name, n_train, n_val, n_test, n_predict)

    # train weak floor, save predictions on train and test
    print(f"\n\033[32m===== Training {model_name} =====\033[0m")  # green text
    mc = ModelConfig(model_name, not disable_lora, TransformerPredictor)
    model = mc.initialize_model()
    train_args["output_dir"] = output_dir
    train_args["learning_rate"] = MODEL_REGISTRY[model_name]["lr"]
    ds_dict = DatasetDict(
        train=ds_with_labels(source_ds["train"]),
        val=ds_with_labels(source_ds["val"]),
    )
    predict_ds_dict = DatasetDict(
        train=concatenate_datasets(
            [
                ds_dict["train"],
                ds_dict["val"],
                ds_with_labels(source_ds["predict"]),
            ]
        ),
        test=ds_with_labels(source_ds["test"]),
    )
    if target_accuracy is not None:
        # we want to estimate the accuracy of the labels we will actually
        # end up using so we take a random sample of predict["train"]
        # as our val set
        ds_dict["val"] = predict_ds_dict["train"].select(
            torch.randperm(len(predict_ds_dict["train"]))[:n_val]
        )
    exp_cfg = mc.to_dict()
    exp_cfg.update(
        {
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_predict": n_predict,
        }
    )
    lm_sft(
        ds_dict=ds_dict,
        model=model.transformer,
        tokenizer=model.tokenizer,
        train_args=TrainingArguments(**train_args),
        loss="xent",
        store_pre_hiddens=False,
        store_post_hiddens=False,
        cfg=exp_cfg,
        predict_dict=predict_ds_dict,
        target_accuracy=target_accuracy,
    )

    # read the predictions
    predict_dir = output_dir / "predictions"
    train_ds = load_from_disk(str(predict_dir / "train"))
    test_ds = load_from_disk(str(predict_dir / "test"))

    # save to disk
    train_ds.save_to_disk(str(output_dir / "weak_train"))
    test_ds.save_to_disk(str(output_dir / "weak_test"))

    # Also to decrease salience of the errors I can "shuffle them around"
    # ie. take err = abs(weak - gold),
    # then permute err, make a new weak label set with
    # weak_nonsalient = gold + np.where(gold, -shuffled_err, shuffled_err)
    if also_save_shuffled_error_labels:
        for name, ds in [("train", train_ds), ("test", test_ds)]:
            ds = ds.with_format("torch")
            err = (ds["soft_pred"] - ds["soft_label"]).abs()
            shuffled_err = err.clone()
            shuffled_err[torch.randperm(len(shuffled_err)), :] = err
            ds = ds.remove_columns(["soft_pred"])
            ds = ds.add_column(
                "soft_pred",
                (
                    ds["soft_label"]
                    + torch.where(ds["soft_label"] > 0.5, -shuffled_err, shuffled_err)
                ).tolist(),
            )
            ds = ds.with_format("python")
            save_f = str(output_dir) + f"_shuffled_err/weak_{name}"
            ds.save_to_disk(save_f)
            print(f"Saving shuffled error labels to {save_f}")


if __name__ == "__main__":
    Fire(main)
