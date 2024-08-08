import random
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
from datasets import Dataset, load_from_disk

from w2s.few_shot_reporter import FewShotReporter
from w2s.model import LMPredictor, ModelConfig
from w2s.reporter import Oracle
from w2s.reporter_experiment import ExperimentConfig, train_and_eval_reporter
from w2s.sft_utils import clear_mem, get_gpu_mem_used
from w2s.utils import assert_type


def few_shot_reporter(
    weak_ds_path: str,
    oracle_ds_path: str,
    test_ds_path: str,
    num_weak: int,
    num_oracle: int,
    n_test: int,
    # model config
    strong_model_name: str = "meta-llama/Meta-Llama-3-8B",
    # ExperimentConfig
    results_folder: Optional[str] = None,
    run_name: str = "default",
    input_col: str = "txt",
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # default to parent / "results"
    if results_folder is None:
        results_folder = str(Path(__file__).parent / "results")

    # load datasets
    weak_ds = assert_type(Dataset, load_from_disk(weak_ds_path))
    weak_ds = (
        weak_ds.remove_columns(["soft_label", "hard_label"])
        .shuffle()
        .select(range(min(num_weak, len(weak_ds))))
    )
    oracle_ds = assert_type(Dataset, load_from_disk(oracle_ds_path)).shuffle()
    oracle_ds = oracle_ds.select(range(min(num_oracle, len(oracle_ds))))
    test_ds = assert_type(Dataset, load_from_disk(test_ds_path))
    test_ds = test_ds.select(range(min(n_test, len(test_ds))))

    dataset_cfg_dict = {
        "weak_ds_path": str(weak_ds_path),
        "oracle_ds_path": str(oracle_ds_path),
        "test_ds_path": str(test_ds_path),
        "n_test": n_test,
        "weak_pool_size": num_weak,
        "oracle_pool_size": num_oracle,
    }

    mcfg = ModelConfig(
        strong_model_name,
        False,
        LMPredictor,
    )
    exp_cfg = ExperimentConfig(
        results_folder=results_folder,
        run_name=run_name,
        input_col=input_col,
    )
    clear_mem()
    get_gpu_mem_used()
    strong_model = mcfg.initialize_model()
    reporter = FewShotReporter(
        num_weak,
        num_oracle,
        weak_ds=weak_ds,
        oracle=Oracle(oracle_ds),
        test_ds=test_ds,
        strong_model=strong_model,
        input_col="txt",
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
    fire.Fire(few_shot_reporter)
