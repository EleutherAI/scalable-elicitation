from typing import Optional

import fire
from datasets import DatasetDict
from transformers import TrainingArguments

from w2s.ds_registry import load_and_process_dataset
from w2s.model import ModelConfig, TransformerPredictor
from w2s.sft import lm_sft
from w2s.sft_config import set_default_args
from w2s.utils import ds_with_labels


def main(
    ds_name,
    model_name,
    n_train,
    n_val,
    n_test,
    save_predictions: bool = False,
    results_folder: Optional[str] = None,
    disable_lora: bool = False,
    quantize: bool = False,
    run_name: Optional[str] = None,
    **train_args,
):
    run_name = run_name or str(hash(train_args))[-8:]  # note this is not deterministic
    train_args = set_default_args(train_args, model_name=model_name, run_name=run_name)
    results_folder = results_folder or f"results/{run_name}"

    # load dataset
    source_ds = load_and_process_dataset(ds_name, n_train, n_val, n_test, 0)

    # train weak floor, save predictions on train and test
    print("\n\033[32m===== Training {model_name} =====\033[0m")  # green text
    mc = ModelConfig(
        model_name, not disable_lora, TransformerPredictor, quantize=quantize
    )
    model = mc.initialize_model()
    train_args["output_dir"] = results_folder
    ds_dict = DatasetDict(
        **{k: ds_with_labels(ds) for k, ds in source_ds.items() if len(ds) > 0}
    )

    lm_sft(
        ds_dict=ds_dict,
        model=model.transformer,
        tokenizer=model.tokenizer,
        train_args=TrainingArguments(**train_args),
        loss="xent",
        store_pre_hiddens=False,
        store_post_hiddens=False,
        cfg=mc.to_dict(),
        predict_dict=ds_dict if save_predictions else None,
    )


if __name__ == "__main__":
    fire.Fire(main)
