import random
from pathlib import Path
from typing import Literal, Optional

import fire
import numpy as np
import torch
from datasets import Dataset, load_from_disk

from w2s.few_shot_reporter import FewShotPromptedSFTReporter
from w2s.model import ModelConfig, TransformerPredictor
from w2s.reporter import Oracle, SftStage
from w2s.reporter_experiment import ExperimentConfig, train_and_eval_reporter
from w2s.sft_config import set_default_args
from w2s.sft_utils import clear_mem, get_gpu_mem_used
from w2s.utils import assert_type


def few_shot_prompted_sft_reporter(
    weak_ds_path: str,
    oracle_ds_path: str,
    test_ds_path: str,
    weak_pool_size: int,
    oracle_pool_size: int,
    n_test: int,
    num_few_shot: int,
    few_shot_type: Literal["oracle", "weak"],
    # model config
    strong_model_name: str = "meta-llama/Meta-Llama-3-8B",
    disable_lora: bool = False,
    # ExperimentConfig
    results_folder: Optional[str] = None,
    run_name: str = "default",
    input_col: str = "txt",
    seed: int = 42,
    **reporter_args,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    reporter_args = set_default_args(
        reporter_args, model_name=strong_model_name, run_name=run_name, seed=seed
    )

    # default to parent / "results"
    if results_folder is None:
        results_folder = str(Path(__file__).parent / "results")
    reporter_args["output_dir"] = str(Path(results_folder) / run_name)

    train_stage = SftStage(**reporter_args)

    # load datasets
    weak_ds = assert_type(Dataset, load_from_disk(weak_ds_path))
    weak_ds = weak_ds.remove_columns(["soft_label", "hard_label"])
    oracle_ds = assert_type(Dataset, load_from_disk(oracle_ds_path)).shuffle()
    oracle_ds = oracle_ds.select(range(min(oracle_pool_size, len(oracle_ds))))
    test_ds = assert_type(Dataset, load_from_disk(test_ds_path))
    test_ds = test_ds.select(range(min(n_test, len(test_ds))))

    dataset_cfg_dict = {
        "weak_ds_path": str(weak_ds_path),
        "oracle_ds_path": str(oracle_ds_path),
        "test_ds_path": str(test_ds_path),
        "n_test": n_test,
        "weak_pool_size": len(weak_ds),
        "oracle_pool_size": len(oracle_ds),
    }
    weak_ds = weak_ds.shuffle().select(range(min(weak_pool_size, len(weak_ds))))

    mcfg = ModelConfig(
        strong_model_name,
        not disable_lora,
        TransformerPredictor,
        num_heads=1,
    )
    exp_cfg = ExperimentConfig(
        results_folder=results_folder,
        run_name=run_name,
        input_col=input_col,
    )
    clear_mem()
    get_gpu_mem_used()
    strong_model = mcfg.initialize_model()
    reporter = FewShotPromptedSFTReporter(
        num_few_shot=num_few_shot,
        few_shot_type=few_shot_type,
        weak_ds=weak_ds,
        oracle=Oracle(oracle_ds),
        test_ds=test_ds,
        strong_model=strong_model,
        train_stage=train_stage,
        input_col=input_col,
    )

    train_and_eval_reporter(
        reporter,
        weak_ds,
        oracle_ds,
        test_ds,
        mcfg,
        exp_cfg,
        dataset_cfg_dict=dataset_cfg_dict,
    )


if __name__ == "__main__":
    fire.Fire(few_shot_prompted_sft_reporter)
