import copy

import numpy as np

# CFG 1: LP(weak), FT(GT), FT(weak) with new head, FT(GT)
cfgs = {
    # "w2s_then_active_lp_gt": [
    #     {
    #         "modules_with_grad": "all",
    #         "type": "weak",
    #         "sampling": "random",
    #         "warmup_steps": 40,
    #         "val_frac": 0.2,
    #         "load_best_model_at_end": True,
    #     },
    #     {
    #         "modules_with_grad": "head",
    #         "type": "oracle",
    #         "sampling": "least_confident_pred",
    #         "sample_temp": 0.25,
    #         "warmup_steps": 40,
    #         "per_device_train_batch_size": 8,
    #         "gradient_accumulation_steps": 4,
    #         "per_device_eval_batch_size": 8,
    #         "load_best_model_at_end": False,
    #         "reuse_optimizer_checkpoint": False,
    #     },
    # ],
    # "w2s_then_lp_gt": [
    #     {
    #         "modules_with_grad": "all",
    #         "type": "weak",
    #         "sampling": "random",
    #         "warmup_steps": 40,
    #         "val_frac": 0.2,
    #         "load_best_model_at_end": True,
    #     },
    #     {
    #         "modules_with_grad": "head",
    #         "type": "oracle",
    #         "sampling": "random",
    #         "warmup_steps": 40,
    #         "per_device_train_batch_size": 8,
    #         "gradient_accumulation_steps": 4,
    #         "per_device_eval_batch_size": 8,
    #         "load_best_model_at_end": False,
    #         "reuse_optimizer_checkpoint": False,
    #     },
    # ],
    "seq_sft_both_estop_clean_disjoint": [
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
    # "seq_sft_both_estop_clean_logconf": [
    #     {
    #         "modules_with_grad": "all",
    #         "type": "weak",
    #         "sampling": "random",
    #         "warmup_steps": 40,
    #         "val_frac": 0.2,
    #         "load_best_model_at_end": True,
    #         "loss": "logconf",
    #     },
    #     {
    #         "modules_with_grad": "all",
    #         "type": "oracle",
    #         "sampling": "random",
    #         "warmup_steps": 40,
    #         "val_frac": 0.2,
    #         "load_best_model_at_end": True,
    #         "reuse_optimizer_checkpoint": False,
    #     },
    # ],
    # "seq_sft_oracle_estop_2x": [
    #     {
    #         "modules_with_grad": "all",
    #         "type": "weak",
    #         "sampling": "random",
    #         "warmup_steps": 40,
    #         "val_frac": 0.2,
    #         "load_best_model_at_end": False,
    #         # go for 2x as long as your stopping criterion would tell you to do
    #         "early_stopping_multiplier": 2,
    #     },
    #     {
    #         "modules_with_grad": "all",
    #         "type": "oracle",
    #         "sampling": "random",
    #         "warmup_steps": 40,
    #         "val_frac": 0.2,
    #         "load_best_model_at_end": True,
    #         "reuse_optimizer_checkpoint": False,
    #     },
    # ],
    # "seq_sft_weak_estop_oracle_active_100steps": [
    #     {
    #         "modules_with_grad": "all",
    #         "type": "weak",
    #         "sampling": "random",
    #         "warmup_steps": 40,
    #         "val_frac": 0.2,
    #         "load_best_model_at_end": True,
    #     },
    #     {
    #         "modules_with_grad": "all",
    #         "type": "oracle",
    #         "sampling": "least_confident_pred",
    #         "sample_temp": 0.25,
    #         "warmup_steps": 40,
    #         "load_best_model_at_end": False,
    #         "reuse_optimizer_checkpoint": False,
    #     },
    # ],
    # "seq_sft_both_estop_active_oracle": [
    #     {
    #         "modules_with_grad": "all",
    #         "type": "weak",
    #         "sampling": "random",
    #         "warmup_steps": 40,
    #         "val_frac": 0.2,
    #         "load_best_model_at_end": True,
    #     },
    #     {
    #         "modules_with_grad": "all",
    #         "type": "oracle",
    #         "sampling": "least_confident_pred",
    #         "sample_temp": 0.25,
    #         "warmup_steps": 40,
    #         "load_best_model_at_end": True,
    #         "val_frac": 0.2,
    #         "reuse_optimizer_checkpoint": False,
    #     },
    # ],
    # "seq_sft_weak_estop_active_oracle_3x": [
    #     {
    #         "modules_with_grad": "all",
    #         "type": "weak",
    #         "sampling": "random",
    #         "warmup_steps": 40,
    #         "val_frac": 0.2,
    #         "load_best_model_at_end": True,
    #     },
    #     {
    #         "modules_with_grad": "all",
    #         "type": "oracle",
    #         "sampling": "least_confident_pred",
    #         "sample_temp": 0.25,
    #         "warmup_steps": 40,
    #         "load_best_model_at_end": False,
    #         "reuse_optimizer_checkpoint": False,
    #         "val_frac": 0.2,
    #         "early_stopping_multiplier": 3,
    #     },
    # ],
}

root = "/mnt/ssd-1/alexm/w2s/results"
# root = "/home/fslcollab366/w2s/results"

weak_models = [
    # "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-4B",
    # "Qwen/Qwen1.5-7B",
]
ds_names = [
    "boolq",
    # "anli-r2",
    # "ethics-virtue",
    # "ethics-utilitarianism",
    # "ethics-justice",
    # "ethics-deontology",
    "hellaswag",
    # "amazon_polarity",
    # "paws",
    # "sciq_with_support",
    "sciq",
    "cola",
    "cosmos_qa",
    "quail",
    "social_i_qa",
    "dream",
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
        for sweep_name, stages in cfgs.items():
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
                f"--per_device_train_batch_size {mbs} "
                "--per_device_eval_batch_size 3 "
                f"--gradient_accumulation_steps {bs // mbs} "
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
                for stage in stages:
                    is_weak = stage["type"] == "weak"
                    total_points = (
                        30_000  # NOTE: total number of datapoints, including repetions
                    )
                    # over epochs
                    num = num_weak if is_weak else num_oracle
                    num_epochs = max(total_points / num, 1)
                    stage["size"] = num
                    steps_per_epoch = int(np.ceil(stage["size"] / bs))
                    eval_every = min(
                        default_eval_every, steps_per_epoch
                    )  # eval at least every epoch
                    stage["eval_steps"], stage["save_steps"] = eval_every, eval_every
                    # set num warmup steps to no more than the number of steps per epoch
                    if "warmup_steps" in stage:
                        stage["warmup_steps"] = max(
                            min(stage["warmup_steps"], steps_per_epoch), 2
                        )
                    if stage.get("load_best_model_at_end"):
                        assert "val_frac" in stage
                    if "val_frac" in stage:
                        stage["n_val"] = max(int(num * stage["val_frac"]), 2)
                        del stage["val_frac"]
                    stage["num_train_epochs"] = num_epochs

                seed = 101  # random.randint(0, 100)
                model_last = strong_model_name.split("/")[-1]
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
                    model_name=strong_model_name,
                )

                # if os.path.exists(f"{root}/{weak_ds}/{run_name}/results.json"):
                #     raise ValueError(f"Results already exist for {run_name}")

                for j, stage in enumerate(stages):
                    prefix = f"stage{j}_"
                    for key, value in stage.items():
                        if isinstance(value, bool):
                            if value:
                                command += f"--{prefix}{key} "
                        else:
                            command += f"--{prefix}{key} {value} "

                return command

            pairs = [
                # weak, oracle
                (80, 0),
                (70, 1),
                (50, 3),
                (40, 4),
                (20, 6),
                (10, 7),
                (0, 8),
                (640, 0),
                (610, 3),
                (520, 12),
                (390, 25),
                (260, 38),
                (130, 51),
                (40, 60),
                (0, 64),
                (2560, 0),
                (2440, 12),
                (2050, 51),
                (1540, 102),
                (1030, 153),
                (520, 204),
                (130, 243),
                (0, 256),
                (10240, 0),
                (9730, 51),
                (8200, 204),
                (6150, 409),
                (4100, 614),
                (2050, 819),
                (520, 972),
                (0, 1024),
                # (5000, 0),
                # (4750, 250),
                # (4000, 1000),
                # (3000, 2000),
                # (2000, 3000),
                # (999, 4000),
                # (250, 4750),
                # (0, 5000),
                # (4750, 62),
                # (4000, 250),
                # (3000, 500),
                # (2000, 750),
                # (999, 1000),
                # (250, 1187),
                # (0, 1250),
                # (4750, 15),
                # (4000, 62),
                # (3000, 125),
                # (2000, 187),
                # (999, 250),
                # (250, 296),
                # (0, 312),
                # (4750, 3),
                # (4000, 15),
                # (3000, 31),
                # (2000, 46),
                # (999, 62),
                # (250, 74),
                # (0, 78),
                # (4000, 3),
                # (3000, 7),
                # (2000, 11),
                # (999, 15),
                # (250, 18),
                # (0, 19),
                # (3000, 2000),
                # (2000, 3000),
                # (999, 4000),
                # (250, 4750),
                # (0, 5000),
                # (999, 400),
                # (0, 500),
                # (5000, 0),
                # (4750, 250),
                # (4000, 1000),
                # (3000, 2000),
                # (2000, 3000),
                # (999, 4000),
                # (250, 4750),
                # (0, 5000),
                # (4750, 25),
                # (4000, 100),
                # (3000, 200),
                # (2000, 300),
                # (999, 400),
                # (250, 475),
                # (0, 500),
                # (4750, 2),
                # (4000, 10),
                # (3000, 20),
                # (2000, 30),
                # (999, 40),
                # (250, 47),
                # (0, 50),
                # (4000, 1),
                # (3000, 2),
                # (2000, 3),
                # (999, 4),
                # (250, 4),
                # (0, 5),
                # (50000, 0),
                # (45000, 5000),
                # (25000, 25000),
                # (4999, 45000),
                # (0, 50000),
                # (50000, 0),
                # (45000, 1250),
                # (25000, 6250),
                # (4999, 11250),
                # (0, 12500),
                # (50000, 0),
                # (45000, 500),
                # (25000, 2500),
                # (4999, 4500),
                # (0, 5000),
                # (50000, 0),
                # (45000, 50),
                # (25000, 250),
                # (4999, 450),
                # (0, 500),
                # (50000, 0),
                # (45000, 5),
                # (25000, 25),
                # (4999, 45),
                # (0, 50),
            ]
            # pairs += [
            #     (0, num_oracle) for num_oracle in [10, 100, 300, 1000, 3000, 10_000]
            # ]
            # pairs = [(10**i - 1, 10**j - 1) for i in range(5) for j in range(5)]
            # pairs.remove((0, 0))
            # pairs += [(num_weak, 0) for num_weak in [10, 100, 600, 3000, 10_000]]

            # pairs = [generate_random_pair() for _ in range(200)]
            for num_weak, num_oracle in pairs:
                cmd = get_command(stages, num_weak, num_oracle)
                if cmd:
                    print(cmd)
