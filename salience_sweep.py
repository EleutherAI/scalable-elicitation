# python salience.py results/amazon_polarity_title_only_both_amplified 2000 200 --use_weak_label
#  --eval_steps 10 --save_steps 10 --num_train_epochs 1 --per_device_train_batch_size 1
#  --per_device_eval_batch_size 3 --gradient_accumulation_steps 32
base_command = (
    "python salience.py "
    "{weak_ds_path} "
    "2500 400 "
    "--strong_model_name {strong_model_name} "
    "--seed {seed} "
    "--run_name salienceV2 "  # NOTE: change this for each sweep
    "--eval_steps 10 "
    "--save_steps 10 "
    "--num_train_epochs 1 "
    "--per_device_train_batch_size 1 "
    "--per_device_eval_batch_size 3 "
    "--gradient_accumulation_steps 32 "
    "--max_ctx 1024 "
)

models = [
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-4B",
    # "Qwen/Qwen1.5-7B",
]
ds_names = [
    "boolq",
    "ethics-virtue",
    "hellaswag",
    "sciq",
]

# ds_names = [
#     "boolq", "anli-r2", "ethics-virtue", "ethics-utilitarianism", "ethics-justice",
#     "hellaswag", "amazon_polarity", "ethics_deontology", "paws", "sciq_with_support"
# ]
weak_ds_list = [
    f"{ds_name}_{model_name.split('/')[-1]}"
    for ds_name in ds_names
    for model_name in models
]
weak_ds_list += [f"{weak_ds}_shuffled_err" for weak_ds in weak_ds_list]
weak_ds_list += [
    f"{ds_name}_{'Meta-Llama-3-8B'}_stopped_at_{model_name.split('/')[-1]}"
    for ds_name in ds_names
    for model_name in models
]
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
seeds = range(3)
for seed in seeds:
    for weak_ds_name in weak_ds_list:
        for i, strong_model_name in enumerate(strong_model_names):
            # remove runs where strong is worse than or equal to weak
            # skip = False
            # for ii in range(i, len(strong_model_names)):
            #     larger_model = strong_model_names[ii].split("/")[-1]
            #     if larger_model in weak_ds_name:
            #         # NOTE: this shouldn't be skipped for non-vanilla weak labels
            #         skip = True
            #         break
            # if skip:
            #     continue

            cmd = base_command.format(
                weak_ds_path=f"results/{weak_ds_name}",
                strong_model_name=strong_model_name,
                seed=seed,
            )
            print(cmd)
            cmd += " --use_weak_label "
            print(cmd)
