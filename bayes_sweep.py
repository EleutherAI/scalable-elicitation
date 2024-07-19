import argparse
import copy
import json
import random
import time
import warnings
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def load_result(path):
    path = Path(path)
    try:
        with open(path / "results.json") as f:
            data = json.load(f)
        with open(path / "config.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        return

    stages_cfg = config["reporter"]["stages"]
    weak_compute = sum(
        stage["num_weak_nonunique"] * stage["train_args"]["num_train_epochs"]
        for stage in stages_cfg
    )
    oracle_compute = sum(
        stage["num_oracle_nonunique"] * stage["train_args"]["num_train_epochs"]
        for stage in stages_cfg
    )
    total_compute = weak_compute + oracle_compute

    # oracle_cost = int(path.name.split("_")[0].split("=")[1]
    seed = int(path.name.split("_")[-1].split("s")[-1])
    if "m=" in path.name:
        sweep_name = "_".join(path.name.split("_")[3:-1])
    else:
        sweep_name = "_".join(path.name.split("_")[2:-1])
    return {
        "auroc": data["auroc"],
        "model_name": config["model"]["name"],
        "num_oracle": data["num_oracle"],
        "num_weak": data["num_weak"],
        "num_oracle_nonunique": data["num_oracle_nonunique"],
        "num_weak_nonunique": data["num_weak_nonunique"],
        "weak_compute": weak_compute,
        "oracle_compute": oracle_compute,
        "total_compute": total_compute,
        "seed": seed,
        "ds_name": path.parent.name,
        "sweep_name": sweep_name,
    }


def get_results_df(ds_names=None, patterns=None):
    patterns = patterns or [
        "nw=*_seq_sft_both_estop_s*",
    ]
    results = []
    if ds_names is None:
        ds_names = [d.name for d in Path("results").iterdir() if d.is_dir()]
    for ds_name in ds_names:
        for subdir in chain(
            *[Path(f"results/{ds_name}").glob(pattern) for pattern in patterns]
        ):
            if result := load_result(subdir):
                results.append(result)
    results_df = pd.DataFrame(results)
    results_df.set_index(["ds_name", "model_name"], inplace=True, drop=False)
    return results_df


reparam = "log({x} + 1)"


def reparametrize(x):
    if reparam == "log({x} + 1)":
        return np.log10(x + 1)
    elif reparam == "{x}":
        return x
    else:
        raise ValueError(f"Unknown reparameterization: {reparam}")


def is_too_close(pair, prev_pairs, radius=0.5):
    p0, p1 = reparametrize(pair[0]), reparametrize(pair[1])
    return any(
        (p0 - reparametrize(pp[0])) ** 2 + (p1 - reparametrize(pp[1])) ** 2
        <= radius**2
        for pp in prev_pairs
    )


stages = [
    {
        "modules_with_grad": "all",
        "type": "weak",
        "sampling": "random",
        "warmup_steps": 40,
        "val_frac": 0.2,
        "load_best_model_at_end": True,
    },
    {
        "modules_with_grad": "all",
        "type": "oracle",
        "sampling": "random",
        "warmup_steps": 0,
        "val_frac": 0.2,
        "load_best_model_at_end": True,
        "reuse_optimizer_checkpoint": True,
    },
]

parser = argparse.ArgumentParser(description="Bayesian sweep configuration")
parser.add_argument(
    "--strong_model_name",
    type=str,
    default="meta-llama/Meta-Llama-3-8B",
    help="Name of the strong model",
)
parser.add_argument(
    "--sweep_name", type=str, default="seq_sft_both_estop", help="Name of the sweep"
)
parser.add_argument(
    "--weak_ds", type=str, default="boolq_Qwen1.5-0.5B", help="Name of the weak dataset"
)
parser.add_argument(
    "--cmds_file",
    type=str,
    default="cmd_lists/bayes_sweep.sh",
    help="File to store commands",
)

args = parser.parse_args()

strong_model_name = args.strong_model_name
sweep_name = args.sweep_name
sweep_pattern = f"nw=*m={strong_model_name.split('/')[-1]}_{sweep_name}_s*"
weak_ds = args.weak_ds
cmds_file = args.cmds_file
root = "/mnt/ssd-1/alexm/w2s/results"
base_command = (
    "python train_transformer_reporter.py "
    "{weak_ds_path} "
    "{oracle_ds_path} "
    "{test_ds_path} "
    "10_000 10_000 1000 "
    "--seed {seed} "
    "--strong_model_name {model_name} "
    "--reporter_stages {reporter_stages} "
    "--num_train_epochs 1 "
    "--eval_steps 50 "
    "--save_steps 50 "
    "--save_total_limit 1 "
    "--per_device_train_batch_size 1 "
    "--per_device_eval_batch_size 3 "
    "--gradient_accumulation_steps 32 "
    f"--results_folder {root}/{weak_ds} "
    '--run_name "{run_name}" '
)
weak_ds_path = f"{root}/{weak_ds}/weak_train"
oracle_ds_path = f"{root}/{weak_ds}/weak_train"
test_ds_path = f"{root}/{weak_ds}/weak_test"


def get_command(stages, num_weak, num_oracle):
    stages = copy.deepcopy(stages)
    # keep only the stages where there is any data to run them with
    stages = [
        stage
        for stage in stages
        if (stage["type"] == "weak" and num_weak > 0)
        or (stage["type"] == "oracle" and num_oracle > 0)
    ]
    # make sure the first stage uses warmup
    if stages[0].get("warmup_steps") == 0:
        stages[0]["warmup_steps"] = 40
    total_points = 20_000
    for stage in stages:
        num = num_weak if stage["type"] == "weak" else num_oracle
        num_points = round(total_points * num / (num_weak + num_oracle))
        num_epochs = max(num_points / num, 1)
        stage["size"] = num
        if stage.get("load_best_model_at_end"):
            stage["n_val"] = max(int(num * stage["val_frac"]), 2)
            del stage["val_frac"]
        stage["num_train_epochs"] = num_epochs

    seed = 5
    model_last = strong_model_name.split("/")[-1]
    run_name = f"nw={num_weak}_no={num_oracle}_m={model_last}_{sweep_name}_s{seed}"
    command = base_command.format(
        weak_ds_path=weak_ds_path,
        oracle_ds_path=oracle_ds_path,
        test_ds_path=test_ds_path,
        seed=seed,
        reporter_stages=len(stages),
        run_name=run_name,
        model_name=strong_model_name,
    )

    for j, stage in enumerate(stages):
        prefix = f"stage{j}_"
        for key, value in stage.items():
            if isinstance(value, bool):
                if value:
                    command += f"--{prefix}{key} "
            else:
                command += f"--{prefix}{key} {value} "

    return command


pairs_added = []
results_df = pd.DataFrame()
while True:
    new_results_df = get_results_df(ds_names=[weak_ds], patterns=[sweep_pattern])
    num_new_rows = min(new_results_df.shape[0] - results_df.shape[0], 8)
    if num_new_rows == 0:
        continue
    results_df = new_results_df

    X_raw = np.stack(
        [results_df["num_weak"].values, results_df["num_oracle"].values],  # type: ignore
        axis=1,
    )
    # we didn't actually run this run, but we know that if you train
    #  on 0 data you'd get 0.5 AUROC on average and we want our GP to reflect that
    X_raw = np.vstack([X_raw, np.zeros((1, 2))])
    X = reparametrize(X_raw)
    y = results_df["auroc"].values
    y = np.hstack([y, np.ones(1) / 2])  # type: ignore

    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds="fixed") + WhiteKernel(
        noise_level=0.1
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=2, normalize_y=True
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            gp.fit(X, y)
            use_random = False
    except ConvergenceWarning:
        print("ConvergenceWarning")
        use_random = True

    xs = np.geomspace(1, 10**4, 100) - 1
    xx1, xx2 = np.meshgrid(xs, xs)
    X_new = np.stack([xx1.ravel(), xx2.ravel()], axis=1)
    auroc, auroc_std = gp.predict(reparametrize(X_new), return_std=True)  # type: ignore
    flat_stds = auroc_std.flatten()
    best_idxs = np.unravel_index(np.argsort(-flat_stds), xx1.shape)
    sorted_pairs = [(xs[i], xs[j]) for i, j in zip(*best_idxs)]

    # get most uncertain or random points
    for i in range(num_new_rows):
        if use_random:
            best_pair = random.choice(sorted_pairs)
        else:
            j = i
            while j < len(sorted_pairs) and is_too_close(
                sorted_pairs[j], pairs_added[-8:]
            ):
                j += 1
            best_pair = sorted_pairs[j]

        pairs_added.append(best_pair)
        cmd = get_command(stages, best_pair[1], best_pair[0])
        with open(cmds_file, "a") as f:
            f.write(cmd + "\n")
        print(cmd)
    time.sleep(10)
