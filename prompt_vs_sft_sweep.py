import copy
from collections import defaultdict

import numpy as np

root = "/mnt/ssd-1/alexm/w2s/results"

weak_models = [
    "Qwen/Qwen1.5-0.5B",
]
cfg = {
    "modules_with_grad": "all",
    "sampling": "random",
    "warmup_steps": 40,
    "val_frac": 0.2,
    "load_best_model_at_end": True,
    "reuse_optimizer_checkpoint": False,
}
seed = 0

ds_names = [
    "hellaswag",
    "cosmos_qa",
    "social_i_qa",
]
weak_ds_list = [
    f"{ds_name}_{model_name.split('/')[-1]}"
    for ds_name in ds_names
    for model_name in weak_models
]

strong_model_names = [
    "meta-llama/Meta-Llama-3-8B",
]

default_eval_every = 50
bs, mbs = 32, 1
for i, strong_model_name in list(enumerate(strong_model_names)):  # NOTE
    for weak_ds in weak_ds_list:
        base_command = (
            "python few_shot_prompted_sft.py "
            "{weak_ds_path} "
            "{oracle_ds_path} "
            "{test_ds_path} "
            "10000 10000 1000 "
            "--seed {seed} "
            "--strong_model_name {model_name} "
            f"--eval_steps {default_eval_every} "
            f"--save_steps {default_eval_every} "
            "--save_total_limit 1 "
            f"--per_device_train_batch_size {mbs} "
            f"--per_device_eval_batch_size {mbs} "
            f"--gradient_accumulation_steps {bs // mbs} "
            f"--results_folder {root}/{weak_ds} "
            '--run_name "{run_name}" '
            "--few_shot_type {few_shot_type} "
            "--num_few_shot {num_few_shot} "
        )

        weak_ds_path = f"{root}/{weak_ds}/weak_train"
        oracle_ds_path = f"{root}/{weak_ds}/weak_train"
        test_ds_path = f"{root}/{weak_ds}/weak_test"

        def get_command(stage_cfg, num_few_shot, num_sft, few_shot_type):
            model_last = strong_model_name.split("/")[-1]
            num = defaultdict(int)
            num[few_shot_type] += num_few_shot
            num[stage_cfg["type"]] += num_sft
            sweep_name = (
                f"{num_few_shot}{few_shot_type}_prompt_{stage_cfg['type']}_sft_estop"
            )
            run_name = f"nw={num['weak']}_no={num['oracle']}_m={model_last}_{sweep_name}_s{seed}"
            command = base_command.format(
                weak_ds_path=weak_ds_path,
                oracle_ds_path=oracle_ds_path,
                test_ds_path=test_ds_path,
                seed=seed,
                run_name=run_name,
                model_name=strong_model_name,
                num_few_shot=num_few_shot,
                few_shot_type=few_shot_type,
            )
            # total number of datapoints, including repetions over epochs
            total_points = 20_000  # NOTE

            num_epochs = max(total_points / num_sft, 1)
            stage_cfg["size"] = num_sft
            steps_per_epoch = int(np.ceil(stage_cfg["size"] / bs))
            eval_every = min(
                default_eval_every, steps_per_epoch
            )  # eval at least every epoch
            stage_cfg["eval_steps"] = stage_cfg["save_steps"] = eval_every
            # set num warmup steps to no more than the number of steps per epoch
            if "warmup_steps" in stage_cfg:
                stage_cfg["warmup_steps"] = max(
                    min(stage_cfg["warmup_steps"], steps_per_epoch), 2
                )
            if stage_cfg.get("load_best_model_at_end"):
                assert "val_frac" in stage_cfg
            if "val_frac" in stage_cfg:
                stage_cfg["n_val"] = max(int(num_sft * stage_cfg["val_frac"]), 2)
                del stage_cfg["val_frac"]
            stage_cfg["num_train_epochs"] = num_epochs

            for k, v in stage_cfg.items():
                if isinstance(v, bool):
                    if v:
                        command += f"--{k} "
                else:
                    command += f"--{k} {v} "

            return command

        for num_few_shot in [2, 8, 32]:
            for few_shot_type in ["weak", "oracle"]:
                for num_sft in [16, 64, 256, 1024, 4096]:
                    for sft_type in ["weak", "oracle"]:
                        current_cfg = copy.deepcopy(cfg)
                        current_cfg["type"] = sft_type
                        cmd = get_command(
                            current_cfg, num_few_shot, num_sft, few_shot_type
                        )
                        if cmd:
                            print(cmd)

                # also do a regular few shot run here
                num_weak = num_few_shot if few_shot_type == "weak" else 0
                num_oracle = num_few_shot if few_shot_type == "oracle" else 0
                model_last = strong_model_name.split("/")[-1]
                run_name = (
                    f"nw={num_weak}_no={num_oracle}_m={model_last}_few_shot_s{seed}"
                )
                cmd = (
                    "python few_shot_prompt.py "
                    f"{weak_ds_path} "
                    f"{oracle_ds_path} "
                    f"{test_ds_path} "
                    f"{num_weak} {num_oracle} 1000 "
                    f"--seed {seed} "
                    f"--strong_model_name {strong_model_name} "
                    f"--results_folder {root}/{weak_ds} "
                    f'--run_name "{run_name}" '
                )
                print(cmd)
