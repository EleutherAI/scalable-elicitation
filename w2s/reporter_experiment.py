import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

from w2s.metrics import acc_ci, roc_auc_ci
from w2s.model import ModelConfig
from w2s.reporter import Reporter


@dataclass
class ExperimentConfig:
    results_folder: str = "./results"
    run_name: str = "default"
    input_col: str = "txt"


def train_and_eval_reporter(
    reporter: Reporter,
    # this dataset is cheap to query but may not have perfect labels
    weak_ds: Dataset,
    # this dataset is expensive to query but has perfect labels
    oracle_ds: Dataset,
    # this dataset is untrusted
    test_ds: Dataset,
    predictor_config: ModelConfig,
    cfg: ExperimentConfig,
    dataset_cfg_dict: dict,
):
    save_path = Path(cfg.results_folder) / cfg.run_name
    if (save_path / "results.json").exists():
        print(f"Results for queries already exist at {save_path}.")
        return
    os.makedirs(save_path, exist_ok=True)

    print(
        "\n\033[32m===== Training reporter with oracle queries =====\033[0m"
    )  # green text

    reporter.fit()
    with torch.no_grad():
        cal_logodds = reporter(test_ds[reporter.input_col])  # type: ignore

    cal_logodds = cal_logodds.cpu().float().numpy()
    gt_labels = np.array(test_ds["soft_label"])[:, 1]
    if not ((gt_labels == 0) | (gt_labels == 1)).all():
        warnings.warn("Ground truth labels are not binary, so we're thresholding them.")
    auc_result = roc_auc_ci(gt_labels > 0.5, cal_logodds)
    acc_result = acc_ci((cal_logodds > 0), (gt_labels > 0.5))

    if "soft_pred" in test_ds.column_names:
        weak_test_labels = np.array(test_ds["soft_pred"])[:, 1]
        weak_auc_result = roc_auc_ci(weak_test_labels > 0.5, cal_logodds)
        weak_acc_result = acc_ci((cal_logodds > 0), (weak_test_labels > 0.5))

        weak_results = {
            "auroc_against_weak": float(weak_auc_result.estimate),
            "auroc_against_weak_lo": float(weak_auc_result.lower),
            "auroc_against_weak_hi": float(weak_auc_result.upper),
            "acc_against_weak": float(weak_acc_result.estimate),
            "acc_against_weak_lo": float(weak_acc_result.lower),
            "acc_against_weak_hi": float(weak_acc_result.upper),
            "weak_soft_labels": weak_test_labels.tolist(),
        }
    else:
        weak_results = {}

    result = {
        "auroc": float(auc_result.estimate),
        "auroc_lo": float(auc_result.lower),
        "auroc_hi": float(auc_result.upper),
        "acc": float(acc_result.estimate),
        "acc_lo": float(acc_result.lower),
        "acc_hi": float(acc_result.upper),
        **weak_results,
        "oracle_ids": list(reporter.oracle.ids_labeled),
        "ids": test_ds["id"],
        "calibrated_logodds": cal_logodds.tolist(),
        "gt_soft_labels": gt_labels.tolist(),
        "reporter": reporter.to_dict(),
    }
    with open(save_path / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(
        {
            k: v
            for k, v in result.items()
            if k
            not in [
                "calibrated_logodds",
                "gt_soft_labels",
                "weak_soft_labels",
                "ids",
                "oracle_ids",
                "stages",
            ]
        }
    )

    # save configuration
    config: dict = {
        "dataset": dataset_cfg_dict,
        "model": predictor_config.to_dict(),
        "reporter": reporter.to_dict(),
    }
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
