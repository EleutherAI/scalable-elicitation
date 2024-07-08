# python salience.py results/amazon_polarity_title_only_both_amplified 2000 200 --use_weak_label
#  --eval_steps 10 --save_steps 10 --num_train_epochs 1 --per_device_train_batch_size 1
#  --per_device_eval_batch_size 3 --gradient_accumulation_steps 32
base_command = (
    "python salience.py "
    "{weak_ds_path} "
    "6000 400 "
    "--strong_model_name {strong_model_name} "
    "--seed {seed} "
    "--eval_steps 10 "
    "--save_steps 10 "
    "--num_train_epochs 1 "
    "--per_device_train_batch_size 1 "
    "--per_device_eval_batch_size 3 "
    "--gradient_accumulation_steps 32 "
    "--max_ctx 256 "
)

weak_ds_paths = ["results/boolq_Qwen1.5-0.5B"]
weak_ds_paths += [
    f"results/{ds_name}_{prompt}"
    for ds_name in [
        "paws_consistency",
        "ethics_deontology_excuse_only",
        "amazon_polarity_title_only",
        "sciq_support_contains",
    ]
    for prompt in [
        "weak_amplified",
        "both_amplified",
        "neither_amplified",
        "gt_amplified",
    ]
]
strong_model_names = [
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
]
seeds = range(3)
for weak_ds_path in weak_ds_paths:
    for strong_model_name in strong_model_names:
        for seed in seeds:
            cmd = base_command.format(
                weak_ds_path=weak_ds_path,
                strong_model_name=strong_model_name,
                seed=seed,
            )
            print(cmd)
            cmd += " --use_weak_label "
            print(cmd)
