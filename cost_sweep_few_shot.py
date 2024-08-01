import random

root = "/mnt/ssd-1/alexm/w2s/results"

weak_models = [
    "Qwen/Qwen1.5-0.5B",
    # "Qwen/Qwen1.5-4B",
    # "Qwen/Qwen1.5-7B",
]
sweep_name = "few_shot_prompt"
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

for i, strong_model_name in list(enumerate(strong_model_names)):  # NOTE
    for weak_ds in weak_ds_list:
        base_command = (
            "python few_shot_prompt.py "
            "{weak_ds_path} "
            "{oracle_ds_path} "
            "{test_ds_path} "
            "{num_weak} {num_oracle} 1000 "
            "--seed {seed} "
            "--strong_model_name {model_name} "
            f"--results_folder {root}/{weak_ds} "
            '--run_name "{run_name}" '
        )

        weak_ds_path = f"{root}/{weak_ds}/weak_train"
        oracle_ds_path = f"{root}/{weak_ds}/weak_train"
        test_ds_path = f"{root}/{weak_ds}/weak_test"

        def get_command(num_weak, num_oracle):
            seed = random.randint(0, 100)
            model_last = strong_model_name.split("/")[-1]
            run_name = (
                f"nw={num_weak}_no={num_oracle}_m={model_last}_{sweep_name}_s{seed}"
            )
            command = base_command.format(
                weak_ds_path=weak_ds_path,
                oracle_ds_path=oracle_ds_path,
                test_ds_path=test_ds_path,
                seed=seed,
                num_weak=num_weak,
                num_oracle=num_oracle,
                run_name=run_name,
                model_name=strong_model_name,
            )

            # if os.path.exists(f"{root}/{weak_ds}/{run_name}/results.json"):
            #     raise ValueError(f"Results already exist for {run_name}")

            return command

        pairs = [
            # weak, oracle
            (3000, 2000),
            (2000, 3000),
            (999, 4000),
            (250, 4750),
            (0, 5000),
            (999, 400),
            (0, 500),
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
        ]

        for num_weak, num_oracle in pairs:
            cmd = get_command(num_weak, num_oracle)
            if cmd:
                print(cmd)
