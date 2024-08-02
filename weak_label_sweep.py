# models = [
#     "Qwen/Qwen1.5-0.5B",
#     "Qwen/Qwen1.5-4B",
# ]
# ds_names = [
#     "sciq", "boolq", "anli-r2", "ethics-virtue", "ethics-utilitarianism",
#     "ethics-justice", "hellaswag", "amazon_polarity", "ethics-deontology",
#     "paws", "sciq_with_support",
# ]
# base_ds_names = ["boolq", "sciq", "ethics-virtue", "hellaswag"]
base_ds_names = ["cola", "quail", "social_i_qa", "cosmos_qa", "dream"]
weak_models = ["Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-4B"]

base_command = "python vanilla_weak_labels.py {ds_name} {model} "

for weak_model in weak_models:
    for ds_name in base_ds_names:
        command = base_command.format(ds_name=ds_name, model=weak_model)
        print(command)

# base_command = (
#     "python vanilla_weak_labels.py {ds_name} {model} --target_accuracy {target_accuracy} "
#     "--run_name {name} --eval_steps 5 --n_val 500"
# )
# model = "meta-llama/Meta-Llama-3-8B"

# for original_weak_model in weak_models:
#     for ds_name in base_ds_names:
#         weak_name = f"{ds_name}_{original_weak_model.split('/')[-1]}"
#         ds = load_from_disk(f"results/{weak_name}/weak_train")
#         acc = sum(
#             (ex["soft_pred"][1] > 0.5) == ex["hard_label"] for ex in ds  # type: ignore
#         ) / len(ds)
#         name = f"{ds_name}_{model.split('/')[-1]}_stopped_at_{original_weak_model.split('/')[-1]}"
#         command = base_command.format(
#             ds_name=ds_name, model=model, name=name, target_accuracy=acc
#         )
#         print(command)
