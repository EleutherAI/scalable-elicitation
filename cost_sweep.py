import copy
from functools import partial
from pathlib import Path

import numpy as np


def openai_batch_size_epochs(n):
    if n < 1024:
        batch_size = 1
    elif n < 4096:
        batch_size = 2
    else:
        batch_size = 8

    if n < 30:
        n_epochs = np.ceil(100 / (n // batch_size))
    elif n < 4096:
        n_epochs = 3
    elif n < 16_384:
        n_epochs = 2
    else:
        n_epochs = 1
    return batch_size, n_epochs


def estop_modify_stages_by_n(stages, num_weak, num_oracle, mbs=1, bs=32):
    stages = [
        stage
        for stage in stages
        if (stage["type"] == "weak" and num_weak > 0)
        or (stage["type"] == "oracle" and num_oracle > 0)
    ]
    # make sure the first stage uses warmup
    if stages[0].get("warmup_steps") == 0:
        stages[0]["warmup_steps"] = 40
    for stage in stages:
        is_weak = stage["type"] == "weak"
        # NOTE: total number of datapoints, including repetions over epochs
        total_points = 30_000
        num = num_weak if is_weak else num_oracle
        num_epochs = max(total_points / num, 1)
        stage["size"] = num
        steps_per_epoch = int(np.ceil(stage["size"] / bs))
        eval_every = min(
            default_eval_every, steps_per_epoch
        )  # eval at least every epoch
        stage["eval_steps"], stage["save_steps"] = (
            eval_every,
            eval_every,
        )
        # set num warmup steps to no more than the number of steps per epoch
        if "warmup_steps" in stage:
            stage["warmup_steps"] = max(min(stage["warmup_steps"], steps_per_epoch), 2)
        if stage.get("load_best_model_at_end"):
            assert "val_frac" in stage
        if "val_frac" in stage:
            stage["n_val"] = max(int(num * stage["val_frac"]), 2)
            del stage["val_frac"]
        stage["num_train_epochs"] = num_epochs
        stage["per_device_train_batch_size"] = mbs
        stage["gradient_accumulation_steps"] = bs // mbs
        stage["per_device_eval_batch_size"] = mbs
    return stages


def openai_modify_stages_by_n(stages, num_weak, num_oracle):
    stages = [
        stage
        for stage in stages
        if (stage["type"] == "weak" and num_weak > 0)
        or (stage["type"] == "oracle" and num_oracle > 0)
    ]
    # make sure the first stage uses warmup
    if stages[0].get("warmup_steps") == 0:
        stages[0]["warmup_steps"] = 40
    for stage in stages:
        is_weak = stage["type"] == "weak"
        num = num_weak if is_weak else num_oracle
        bs, num_epochs = openai_batch_size_epochs(num)
        mbs = 1
        stage["size"] = num
        steps_per_epoch = int(np.ceil(stage["size"] / bs))
        # don't do intermediate evals
        stage["eval_steps"] = stage["save_steps"] = 1_000_000
        # set num warmup steps to no more than the number of steps per epoch
        if "warmup_steps" in stage:
            stage["warmup_steps"] = max(min(stage["warmup_steps"], steps_per_epoch), 2)
        assert "val_frac" not in stage and not stage.get("load_best_model_at_end")
        stage["num_train_epochs"] = num_epochs
        stage["per_device_train_batch_size"] = mbs
        stage["gradient_accumulation_steps"] = bs // mbs
        stage["per_device_eval_batch_size"] = mbs
    return stages


# CFG 1: LP(weak), FT(GT), FT(weak) with new head, FT(GT)
cfgs = {
    "seq_sft_openai_settings": {
        "stages": [
            {
                "modules_with_grad": "all",
                "type": "weak",
                "sampling": "random",
                "warmup_steps": 40,
                "load_best_model_at_end": False,
            },
            {
                "modules_with_grad": "all",
                "type": "oracle",
                "sampling": "random",
                "warmup_steps": 40,
                "reuse_optimizer_checkpoint": False,
            },
        ],
        "modify_stages_by_n": openai_modify_stages_by_n,
    },
    "seq_sft_both_estop_clean_disjoint_2shot": {
        "stages": [
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
                "warmup_steps": 40,
                "val_frac": 0.2,
                "load_best_model_at_end": True,
                "reuse_optimizer_checkpoint": False,
            },
        ],
        "modify_stages_by_n": estop_modify_stages_by_n,
        "extra_args": ["--n_few_shot 2", "--few_shot_type weak"],
    },
    "seq_sft_both_estop_clean_disjoint_32shot_weak": {
        "stages": [
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
                "warmup_steps": 40,
                "val_frac": 0.2,
                "load_best_model_at_end": True,
                "reuse_optimizer_checkpoint": False,
            },
        ],
        "modify_stages_by_n": estop_modify_stages_by_n,
        "extra_args": ["--n_few_shot 32", "--few_shot_type weak"],
    },
    "seq_sft_both_estop_clean_disjoint": {
        "stages": [
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
                "warmup_steps": 40,
                "val_frac": 0.2,
                "load_best_model_at_end": True,
                "reuse_optimizer_checkpoint": False,
            },
        ],
        "modify_stages_by_n": estop_modify_stages_by_n,
    },
    "seq_sft_both_estop_disjoint_logconf": {
        "stages": [
            {
                "modules_with_grad": "all",
                "type": "weak",
                "sampling": "random",
                "warmup_steps": 40,
                "val_frac": 0.2,
                "load_best_model_at_end": True,
                "loss": "logconf",
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "per_device_eval_batch_size": 8,
            },
            {
                "modules_with_grad": "all",
                "type": "oracle",
                "sampling": "random",
                "warmup_steps": 40,
                "val_frac": 0.2,
                "load_best_model_at_end": True,
                "reuse_optimizer_checkpoint": False,
            },
        ],
        "modify_stages_by_n": partial(estop_modify_stages_by_n, mbs=8, bs=32),
        "extra_args": ["--max_ctx 403"],
    },
    "seq_sft_both_estop_active_oracle_disjoint": {
        "stages": [
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
                "sampling": "least_confident_pred",
                "sample_temp": 0.0,
                "warmup_steps": 40,
                "load_best_model_at_end": True,
                "val_frac": 0.2,
                "reuse_optimizer_checkpoint": False,
            },
        ],
        "modify_stages_by_n": estop_modify_stages_by_n,
    },
}


root = str(Path(__file__).parent)

weak_models = [
    "Qwen/Qwen1.5-0.5B",
]
ds_names = [
    "boolq",
    "hellaswag",
    "paws",
    "sciq",
    "cola",
    "cosmos_qa",
    "quail",
    "social_i_qa",
]
weak_ds_list = [
    f"{ds_name}_{model_name.split('/')[-1]}"
    for ds_name in ds_names
    for model_name in weak_models
]
strong_model_names = [
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
]
default_eval_every = 50

for seed in [0,]:
    for i, strong_model_name in list(enumerate(strong_model_names)):
        quantize = "70B" in strong_model_name
        for weak_ds in weak_ds_list:
            for sweep_name, cfg in cfgs.items():
                base_command = (
                    "python train_transformer_reporter.py "
                    "{weak_ds_path} "
                    "{oracle_ds_path} "
                    "{test_ds_path} "
                    "100_000 100_000 1000 "
                    "--seed {seed} "
                    "--strong_model_name {model_name} "
                    "--reporter_stages {reporter_stages} "
                    f"--eval_steps {default_eval_every} "
                    f"--save_steps {default_eval_every} "
                    "--save_total_limit 1 "
                    f"--results_folder {root}/{weak_ds} "
                    '--run_name "{run_name}" '
                )
                for extra_arg in cfg.get("extra_args", []):
                    base_command += f"{extra_arg} "
                if quantize:
                    base_command += "--quantize "

                weak_ds_path = f"{root}/{weak_ds}/weak_train"
                oracle_ds_path = f"{root}/{weak_ds}/weak_train"
                test_ds_path = f"{root}/{weak_ds}/weak_test"

                def get_command(cfg, num_weak, num_oracle):
                    stages = copy.deepcopy(cfg["stages"])
                    stages = cfg["modify_stages_by_n"](stages, num_weak, num_oracle)

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

                weak_marginal_costs = [1 / 10]
                oracle_affordables = [16, 64, 256, 1024, 4096]
                oracle_spending_fracs = [0.8, 0.6, 0.4, 0.2, 0.05]

                pairs = []
                for weak_marginal_cost in weak_marginal_costs:
                    for oracle_affordable in oracle_affordables:
                        accs = []
                        actual_osfs = []
                        for osf in oracle_spending_fracs:
                            n_oracle = int(osf * oracle_affordable)
                            n_weak = int(
                                (oracle_affordable - n_oracle) / weak_marginal_cost
                            )
                            n_oracle = min(n_oracle, 23_000)
                            pairs.append((n_weak, n_oracle))
                
                pairs.append((0, 8192))
                

                for num_weak, num_oracle in pairs:
                    cmd = get_command(cfg, num_weak, num_oracle)
                    if cmd:
                        print(cmd)
