import copy

# CFG 1: LP(weak), FT(GT), FT(weak) with new head, FT(GT)
cfgs = {
    "w2s_then_lp_gt": [
        {
            "modules_with_grad": "all",
            "type": "weak",
            "sampling": "random",
            "warmup_steps": 40,
            "val_frac": 0.2,
            "load_best_model_at_end": True,
        },
        {
            "modules_with_grad": "head",
            "type": "oracle",
            "sampling": "random",
            "warmup_steps": 40,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "per_device_eval_batch_size": 8,
            "load_best_model_at_end": False,
            "reuse_optimizer_checkpoint": False,
        },
    ]
    # "seq_sft_both_estop": [
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
    #         "sampling": "random",
    #         "warmup_steps": 0,
    #         "val_frac": 0.2,
    #         "load_best_model_at_end": True,
    #         "reuse_optimizer_checkpoint": True,
    #     },
    # ]
}

root = "/mnt/ssd-1/alexm/w2s/results"
# root = "/home/fslcollab366/w2s/results"

weak_models = [
    "Qwen/Qwen1.5-0.5B",
    # "Qwen/Qwen1.5-4B",
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
]
weak_ds_list = [
    [
        f"{ds_name}_{'Meta-Llama-3-8B'}_stopped_at_{model_name.split('/')[-1]}",
        f"{ds_name}_{model_name.split('/')[-1]}",
        f"{ds_name}_{model_name.split('/')[-1]}_shuffled_err",
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
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
    "meta-llama/Meta-Llama-3-8B",
]

for i, strong_model_name in list(enumerate(strong_model_names))[::-1][:1]:  # NOTE
    for sweep_name, stages in cfgs.items():
        for weak_ds in weak_ds_list:
            # skip = False
            # for ii in range(i, len(strong_model_names)):
            #     larger_model = strong_model_names[ii].split("/")[-1]
            #     if larger_model in weak_ds:
            #         # NOTE: this shouldn't necessarily be skipped for non-vanilla weak labels
            #         skip = True
            #         break
            # if skip:
            #     continue

            base_command = (
                "python train_transformer_reporter.py "
                "{weak_ds_path} "
                "{oracle_ds_path} "
                "{test_ds_path} "
                "10_000 10_000 1000 "
                "--seed {seed} "
                "--strong_model_name {model_name} "
                "--reporter_stages {reporter_stages} "
                "--eval_steps 50 "
                "--save_steps 50 "
                "--save_total_limit 1 "
                "--per_device_train_batch_size 2 "
                "--per_device_eval_batch_size 3 "
                "--gradient_accumulation_steps 16 "
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
                total_points = 20_000  # total number of datapoints, including repetions over epochs
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

            # pairs = [
            #     # weak, oracle
            #     (10, 0),
            #     (0, 15),
            #     (10, 10),
            #     (0, 12),
            #     (12, 0),
            #     (15, 0),
            #     (900, 900),
            #     (900, 600),
            #     (600, 900),
            #     (10, 100),
            #     (10, 15),
            # ]
            # pairs += [
            #     (0, num_oracle) for num_oracle in [10, 100, 300, 1000, 3000, 10_000]
            # ]
            pairs = [(10**i - 1, 10**j - 1) for i in range(5) for j in range(5)]
            pairs.remove((0, 0))
            # pairs += [(num_weak, 0) for num_weak in [10, 100, 600, 3000, 10_000]]

            # def generate_random_pair():
            #     choice = np.random.random()
            #     if choice < 0.25:
            #         return (loguniform.rvs(1, 1e4) - 1, 0)
            #     elif choice < 0.5:
            #         return (0, loguniform.rvs(1, 1e4) - 1)
            #     else:
            #         return (loguniform.rvs(1, 1e4) - 1, loguniform.rvs(1, 1e4) - 1)

            # pairs = [generate_random_pair() for _ in range(200)]
            for num_weak, num_oracle in pairs:
                cmd = get_command(stages, num_weak, num_oracle)
                if cmd:
                    print(cmd)
