import copy
import os

import pandas as pd

# CFG 1: LP(weak), FT(GT), FT(weak) with new head, FT(GT)
cfgs = {
    "salience_cfg0": [
        [
            {
                "modules_with_grad": "head",
                "type": "weak",
                "sampling": "most_confident_label",
                "sample_temp": 0.25,
            },
            {
                "modules_with_grad": "all",
                "type": "oracle",
                "sampling": "least_confident_pred",
                "sample_temp": 0.25,
            },
            {
                "modules_with_grad": "all",
                "reinit_head": True,
                "type": "weak",
                "sampling": "most_confident_label",
                "sample_temp": 0.25,
            },
            {
                "modules_with_grad": "all",
                "type": "oracle",
                "sampling": "least_confident_pred",
                "sample_temp": 0.25,
            },
        ],
    ],
    "seq_sft": [
        [
            {
                "modules_with_grad": "all",
                "type": "weak",
                "sampling": "random",
            },
            {
                "modules_with_grad": "all",
                "type": "oracle",
                "sampling": "random",
                "num_warmup_steps": 0,
                "reuse_optimizer_checkpoint": True,
            },
        ],
    ],
}

root = "/mnt/ssd-1/alexm/w2s/results"
salience_df = pd.read_json("results/salience_results_all_models.json", lines=True)
salience_df = salience_df[salience_df["against"] == "oracle"]

weak_ds_list = ["boolq_Qwen1.5-0.5B"]
weak_ds_list += [
    f"{ds_name}_{prompt}"
    for ds_name in [
        "ethics_deontology_excuse_only",
        "amazon_polarity_title_only",
        "sciq_support_contains",
        "paws_consistency",
    ]
    for prompt in [
        "weak_amplified",
        "both_amplified",
        "neither_amplified",
        "gt_amplified",
    ]
]
model_names = [
    # "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
]

for model_name in model_names:
    for sweep_name, stages_list in cfgs.items():
        for weak_ds in weak_ds_list:
            curr_df = salience_df[
                (salience_df["ds_name"] == weak_ds)
                & (salience_df["model_name"] == model_name)
            ]
            smallest_good_num_points = round(curr_df["smallest_good_num_points"].mean())

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
                total_points = smallest_good_num_points * 1
                for stage in stages:
                    num = num_weak if stage["type"] == "weak" else num_oracle
                    num_points = round(total_points * num / (num_weak + num_oracle))
                    num_epochs = max(num_points / num, 1)
                    stage["size"] = num
                    stage["num_train_epochs"] = num_epochs

                seed = 5
                model_last = model_name.split("/")[-1]
                run_name = (
                    f"nw={num_weak}_no={num_oracle}_m={model_last}_{sweep_name}_s{seed}"
                )
                command = base_command.format(
                    weak_ds_path=weak_ds_path,
                    oracle_ds_path=oracle_ds_path,
                    test_ds_path=test_ds_path,
                    seed=seed,
                    reporter_stages=len(stages),
                    run_name=run_name,
                    model_name=model_name,
                )

                if os.path.exists(f"{root}/{weak_ds}/{run_name}/results.json"):
                    raise ValueError(f"Results already exist for {run_name}")

                for j, stage in enumerate(stages):
                    prefix = f"stage{j}_"
                    for key, value in stage.items():
                        if isinstance(value, bool):
                            if value:
                                command += f"--{prefix}{key} "
                        else:
                            command += f"--{prefix}{key} {value} "

                return command

            # pairs = [
            #     (8000, 400),
            #     (6000, 220),
            #     (8000, 3000),
            #     (50, 10),
            #     (2, 70),
            #     (2, 400),
            #     (800, 8),
            #     (450, 50),
            #     (800, 20),
            #     (3000, 300),
            #     (2000, 130),
            #     (1000, 25),
            #     (800, 800),
            #     (2500, 120),
            #     (100, 500),
            #     (400, 400),
            #     (4000, 700),
            #     (6000, 1000),
            #     (100, 100),
            #     (10, 100),
            #     (800, 200),
            #     (1000, 10),
            #     (1000, 100),
            #     (500, 100),
            #     (6500, 2),
            #     (7000, 10),
            #     (6400, 20),
            #     (7000, 100),
            #     (6800, 300),
            #     (6600, 2000),
            #     (6800, 5000),
            #     (1000, 5500),
            #     (200, 700),
            #     (1000, 4),
            #     (5000, 20),
            # ]
            pairs = [
                # (2, 100),
                # (2, 120),
                # (2, 200),
                (50, 10),
                # (2, 70),
                # (2, 400),
                (800, 8),
                (450, 50),
                (800, 20),
                # (3000, 300),
                # (2000, 130),
                (1000, 25),
                # (100, 800),
                (2500, 120),
                (100, 500),
                # (400, 400),
                (4000, 700),
                (6000, 1000),
                (100, 100),
                # (10, 100),
                # (100, 1000),
                (1000, 10),
                (1000, 100),
                (500, 100),
                # (6500, 2),
                (7000, 10),
                # (6400, 20),
                (7000, 100),
                (6800, 300),
                # (6600, 2000),
                (6800, 5000),
                (1000, 5500),
                # (7000, 100),
                # (2, 7000),
                # (2, 5000),
                # (20, 7000),
                (1000, 4),
                (5000, 20),
            ]
            pairs += [
                (0, num_oracle) for num_oracle in [10, 100, 300, 1000, 3000, 10_000]
            ]
            pairs += [(num_weak, 0) for num_weak in [100, 600, 3000, 10_000]]
            for stages in stages_list:
                for num_weak, num_oracle in pairs:
                    cmd = get_command(stages, num_weak, num_oracle)
                    if cmd:
                        print(cmd)
