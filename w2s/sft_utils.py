import gc
import shutil
from pathlib import Path
from typing import Dict

import pynvml
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    PretrainedConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import EvaluationStrategy

from w2s.metrics import roc_auc
from w2s.utils import assert_type


def compute_acc_and_auroc(eval_pred):
    predictions, labels = map(torch.from_numpy, eval_pred)
    if predictions.dim() == 3:
        predictions = predictions.mean(dim=1)

    hard_labels = (labels > 0.5).long()
    return dict(
        accuracy=predictions.argmax(dim=-1).eq(hard_labels).float().mean(),
        auroc=roc_auc(hard_labels, predictions[..., 1]).mean(),
    )


@torch.no_grad()
def gather_hiddens(model: torch.nn.Module, dataset: Dataset):
    dataset = dataset.with_format("torch", device="cuda")

    cfg = assert_type(PretrainedConfig, model.config)
    D = assert_type(int, cfg.hidden_size)
    L = assert_type(int, cfg.num_hidden_layers)

    buffer = torch.empty(L, len(dataset), D, device=model.device, dtype=model.dtype)
    for i, ex in enumerate(tqdm(dataset)):
        ex = assert_type(dict, ex)

        out = model(ex["input_ids"][None], output_hidden_states=True)
        buffer[i] = torch.stack(out.hidden_states)[:, 0, -1]  # Final token

    return buffer


def move_best_ckpt(trainer: Trainer):
    checkpoints = list(Path(trainer.args.output_dir).glob("checkpoint-*"))
    path = trainer.state.best_model_checkpoint
    if not checkpoints or path is None:
        print("No checkpoints found, saving final model")
        trainer.save_model(f"{trainer.args.output_dir}/best-ckpt")
        trainer._save_optimizer_and_scheduler(f"{trainer.args.output_dir}/best-ckpt")
        return

    perf = trainer.state.best_metric
    if perf is not None:
        print(f"Best model (auroc {perf:.3f}) saved at: {path}")

    src = Path(path)
    dest = src.parent / "best-ckpt"
    src.rename(dest)


def delete_all_ckpts(trainer: Trainer):
    for ckpt in Path(trainer.args.output_dir).glob("checkpoint-*"):
        shutil.rmtree(ckpt)
    ckpt = Path(trainer.args.output_dir) / "best-ckpt"
    if ckpt.exists():
        shutil.rmtree(ckpt)


def clear_mem(verbose: bool = False):
    """
    This function is used to clear the memory allocated by PyTorch.
    It does so by calling the garbage collector to release unused GPU memory.
    After clearing the memory, it prints the current amount of memory still
    allocated by PyTorch (post-clean).

    Parameters:
    verbose (bool): Whether to print additional information.
    """

    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
    )

    if verbose:

        def try_attr(x, a):
            return getattr(x, a, None)

        for obj in gc.get_objects():
            if torch.is_tensor(obj) or torch.is_tensor(try_attr(obj, "data")):
                print(type(obj), obj.size(), obj.dtype)


def get_gpu_mem_used() -> float:
    """returns proportion of used GPU memory averaged across all GPUs"""
    prop_sum = 0
    pynvml.nvmlInit()
    try:
        num_devices = pynvml.nvmlDeviceGetCount()
        for i in range(num_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            prop_sum += int(meminfo.used) / int(meminfo.total)
    finally:
        pynvml.nvmlShutdown()
    return prop_sum / num_devices


class EarlyStoppingCallback(TrainerCallback):
    """
    This callback stops training upon the `early_stopping_patience`th consecutive
    evaluation that fails to improve upon the best-yet metric value by at least
    `early_stopping_threshold`.

    If `multiplier` is passed, it delays early stopping by that factor, so
    training halts on step `multiplier * n` instead of step `n`.
    """

    def __init__(
        self,
        early_stopping_patience: int = 4,
        early_stopping_threshold: float = 0.01,
        multiplier: float = 1.0,
    ):
        assert multiplier >= 1
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.multiplier = multiplier
        self.early_stopping_patience_counter = 0
        self.best_score = None
        self.stop_at = None

    def check_metric_value(self, args, state, control, metric_value):
        if self.best_score is None:
            self.best_score = metric_value
        elif (
            state.global_step > 0 and args.evaluation_strategy != EvaluationStrategy.NO
        ):
            if (
                metric_value != metric_value
                or metric_value < self.best_score + self.early_stopping_threshold
            ):
                self.early_stopping_patience_counter += 1
                if self.early_stopping_patience_counter >= self.early_stopping_patience:
                    self.stop_at = int(self.multiplier * state.global_step)
                    if metric_value != metric_value:
                        control.best_model_checkpoint = str(
                            Path(args.output_dir) / f"checkpoint-{state.global_step}"
                        )
                        control.best_metric = metric_value
            else:
                self.best_score = metric_value
                self.early_stopping_patience_counter = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        name = args.metric_for_best_model
        if name is None:
            print('No metric for best model found, defaulting to "eval_val_auroc"')
            name = "eval_val_auroc"
            args.metric_for_best_model = name
            args.greater_is_better = True
        name = name if name.startswith("eval_") else f"eval_{name}"
        metric_value = metrics.get(name)  # type: ignore

        if metric_value and self.stop_at is None:
            metric_value = metric_value if args.greater_is_better else -metric_value
            self.check_metric_value(args, state, control, metric_value)

        if state.global_step == self.stop_at:
            control.should_training_stop = True


class AccuracyStoppingCallback(TrainerCallback):
    """
    This callback stops training as soon as val accuracy exceeds a threshold
    """

    def __init__(self, target_accuracy: float = 0.0):
        self.target_accuracy = target_accuracy

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        if metrics["eval_val_accuracy"] >= self.target_accuracy:
            control.should_training_stop = True
