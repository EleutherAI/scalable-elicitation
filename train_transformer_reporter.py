import random
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
from datasets import Dataset, load_from_disk

from w2s.model import ModelConfig, TransformerPredictor
from w2s.reporter import ModularSftReporter, Oracle, SftStage
from w2s.reporter_experiment import ExperimentConfig, train_and_eval_reporter
from w2s.sft_config import set_default_args
from w2s.sft_utils import clear_mem, get_gpu_mem_used
from w2s.utils import assert_type, split_args_by_prefix


def train_reporter_on_transformer(
    weak_ds_path: str,
    oracle_ds_path: str,
    test_ds_path: str,
    weak_pool_size: int,
    oracle_pool_size: int,
    n_test: int,
    # model config
    strong_model_name: str = "meta-llama/Meta-Llama-3-8B",
    disable_lora: bool = False,
    max_ctx: int = 8192,
    # ExperimentConfig
    reporter_stages=1,
    results_folder: Optional[str] = None,
    run_name: str = "default",
    input_col: str = "txt",
    seed: int = 42,
    num_heads: int = 1,
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

    stage_args = split_args_by_prefix(
        reporter_args, [f"stage{i}_" for i in range(reporter_stages)]  # type: ignore
    )
    for stage in stage_args:
        stage_args[stage]["output_dir"] = str(
            Path(reporter_args["output_dir"]) / stage[:-1]
        )
        stage_args[stage]["run_name"] = f"{run_name}-{stage[:-1]}"
    stages = [SftStage(**stage_args[f"stage{i}_"]) for i in range(reporter_stages)]

    def load_from_disk_and_dedup(path: str) -> Dataset:
        ds = assert_type(Dataset, load_from_disk(path))
        df = ds.to_pandas()
        df.drop_duplicates(subset=["txt"], inplace=True)
        ds = Dataset.from_pandas(df)
        return ds

    # load datasets
    weak_ds = load_from_disk_and_dedup(weak_ds_path)
    weak_ds = weak_ds.remove_columns(["soft_label", "hard_label"])
    oracle_ds = load_from_disk_and_dedup(oracle_ds_path)
    test_ds = load_from_disk_and_dedup(test_ds_path)
    test_ds = test_ds.select(range(min(n_test, len(test_ds))))

    if weak_ds_path == oracle_ds_path:
        # apportion the weak and oracle pool by how many of each are requested
        total_num_weak = sum(stage.size for stage in stages if stage.type == "weak")
        total_num_oracle = sum(stage.size for stage in stages if stage.type == "oracle")
        num_to_weak = int(
            len(weak_ds) * total_num_weak / (total_num_weak + total_num_oracle)
        )
        # get random partition
        idxs = list(range(len(weak_ds)))
        random.shuffle(idxs)
        weak_ds = weak_ds.select(idxs[:num_to_weak])
        oracle_ds = oracle_ds.select(idxs[num_to_weak:])

    oracle_ds = oracle_ds.shuffle().select(range(min(oracle_pool_size, len(oracle_ds))))
    weak_ds = weak_ds.shuffle().select(range(min(weak_pool_size, len(weak_ds))))

    dataset_cfg_dict = {
        "weak_ds_path": str(weak_ds_path),
        "oracle_ds_path": str(oracle_ds_path),
        "test_ds_path": str(test_ds_path),
        "n_test": n_test,
        "weak_pool_size": weak_pool_size,
        "oracle_pool_size": len(oracle_ds),
    }

    assert num_heads == 1
    mcfg = ModelConfig(
        strong_model_name,
        not disable_lora,
        TransformerPredictor,
        num_heads=num_heads,
        max_ctx=max_ctx,
    )
    exp_cfg = ExperimentConfig(
        results_folder=results_folder,
        run_name=run_name,
        input_col=input_col,
    )
    clear_mem()
    get_gpu_mem_used()
    strong_model = mcfg.initialize_model()
    reporter = ModularSftReporter(
        weak_ds=weak_ds,
        oracle=Oracle(oracle_ds),
        test_ds=test_ds,
        strong_model=strong_model,
        stages=stages,
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
    fire.Fire(train_reporter_on_transformer)
