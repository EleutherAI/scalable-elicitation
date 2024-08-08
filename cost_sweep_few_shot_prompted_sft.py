import copy
import random

import numpy as np

root = "/mnt/ssd-1/alexm/w2s/results"

weak_models = [
    "Qwen/Qwen1.5-0.5B",
    # "Qwen/Qwen1.5-4B",
    # "Qwen/Qwen1.5-7B",
]
cfgs = {
    "weak_prompted_oracle_sft_estop": {
        "modules_with_grad": "all",
        "type": "oracle",
        "sampling": "random",
        "warmup_steps": 40,
        "val_frac": 0.2,
        "load_best_model_at_end": True,
        "reuse_optimizer_checkpoint": False,
    }
}
few_shot_type = "weak"

ds_names = [
    # "boolq",
    # "anli-r2",
    # "ethics-virtue",
    # "ethics-utilitarianism",
    # "ethics-justice",
    # "ethics-deontology",
    "hellaswag",
    # "amazon_polarity",
    # "paws",
    # "sciq_with_support",
    # "sciq",
    # "cola",
    "cosmos_qa",
    # "quail",
    "social_i_qa",
    # "dream",
]
weak_ds_list = [
    [
        # f"{ds_name}_{'Meta-Llama-3-8B'}_stopped_at_{model_name.split('/')[-1]}",
        f"{ds_name}_{model_name.split('/')[-1]}",
        # f"{ds_name}_{model_name.split('/')[-1]}_shuffled_err",
    ]
    for ds_name in ds_names
    for model_name in weak_models
]
weak_ds_list = [item for sublist in weak_ds_list for item in sublist]
# weak_ds_list += [f"{weak_ds}_shuffled_err" for weak_ds in weak_ds_list]
# weak_ds_list += [
#     f"{ds_name}_{prompt}"
#     for ds_name in [
#         "ethics_deontology_excuse_only",
#         "amazon_polarity_title_only",
#         "sciq_support_contains",
#         "paws_consistency",
#     ]
#     for prompt in [
#         "weak_amplified",
#         "both_amplified",
#         "neither_amplified",
#         "gt_amplified",
#     ]
# ]
strong_model_names = [
    # "Qwen/Qwen1.5-0.5B",
    # "Qwen/Qwen1.5-4B",
    # "Qwen/Qwen1.5-7B",
    "meta-llama/Meta-Llama-3-8B",
]

default_eval_every = 50
bs, mbs = 32, 2
for i, strong_model_name in list(enumerate(strong_model_names)):  # NOTE
    for weak_ds in weak_ds_list:
        for sweep_name, stage_cfg in cfgs.items():
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
                "--per_device_eval_batch_size 3 "
                f"--gradient_accumulation_steps {bs // mbs} "
                f"--results_folder {root}/{weak_ds} "
                '--run_name "{run_name}" '
                f"--few_shot_type {few_shot_type} "
                "--num_few_shot {num_few_shot} "
            )

            weak_ds_path = f"{root}/{weak_ds}/weak_train"
            oracle_ds_path = f"{root}/{weak_ds}/weak_train"
            test_ds_path = f"{root}/{weak_ds}/weak_test"

            def get_command(stage_cfg, num_weak, num_oracle):
                seed = random.randint(0, 100)
                model_last = strong_model_name.split("/")[-1]
                run_name = (
                    f"nw={num_weak}_no={num_oracle}_m={model_last}_{sweep_name}_s{seed}"
                )
                num_few_shot = num_weak if few_shot_type == "weak" else num_oracle
                if num_few_shot > 256:
                    return
                command = base_command.format(
                    weak_ds_path=weak_ds_path,
                    oracle_ds_path=oracle_ds_path,
                    test_ds_path=test_ds_path,
                    seed=seed,
                    num_weak=num_weak,
                    num_oracle=num_oracle,
                    run_name=run_name,
                    model_name=strong_model_name,
                    num_few_shot=num_few_shot,
                )

                # if os.path.exists(f"{root}/{weak_ds}/{run_name}/results.json"):
                #     raise ValueError(f"Results already exist for {run_name}")

                is_weak = stage_cfg["type"] == "weak"
                total_points = (
                    20_000  # NOTE: total number of datapoints, including repetions
                )
                # over epochs
                num = num_weak if is_weak else num_oracle
                if num <= 1:
                    return
                num_epochs = max(total_points / num, 1)
                stage_cfg["size"] = num
                steps_per_epoch = int(np.ceil(stage_cfg["size"] / bs))
                eval_every = min(
                    default_eval_every, steps_per_epoch
                )  # eval at least every epoch
                stage_cfg["eval_steps"], stage_cfg["save_steps"] = (
                    eval_every,
                    eval_every,
                )
                # set num warmup steps to no more than the number of steps per epoch
                if "warmup_steps" in stage_cfg:
                    stage_cfg["warmup_steps"] = max(
                        min(stage_cfg["warmup_steps"], steps_per_epoch), 2
                    )
                if stage_cfg.get("load_best_model_at_end"):
                    assert "val_frac" in stage_cfg
                if "val_frac" in stage_cfg:
                    stage_cfg["n_val"] = max(int(num * stage_cfg["val_frac"]), 2)
                    del stage_cfg["val_frac"]
                stage_cfg["num_train_epochs"] = num_epochs

                for k, v in stage_cfg.items():
                    if isinstance(v, bool):
                        if v:
                            command += f"--{k} "
                    else:
                        command += f"--{k} {v} "

                return command

            weak_marginal_costs = [1 / 10]
            oracle_spending_fracs = [0.0, 0.05, 0.5, 0.95, 1.0]
            oracle_affordables = [8, 16, 64, 256]

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
                cmd = get_command(copy.deepcopy(stage_cfg), num_weak, num_oracle)
                if cmd:
                    print(cmd)
