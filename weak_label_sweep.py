from datasets import load_from_disk

# models = [
#     "Qwen/Qwen1.5-0.5B",
#     "Qwen/Qwen1.5-4B",
# ]
# ds_names = [
#     "sciq", "boolq", "anli-r2", "ethics-virtue", "ethics-utilitarianism",
#     "ethics-justice", "hellaswag", "amazon_polarity", "ethics-deontology",
#     "paws", "sciq_with_support",
# ]
base_ds_names = ["boolq", "sciq", "ethics-virtue", "hellaswag"]
weak_models = ["Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-4B"]
model = "meta-llama/Meta-Llama-3-8B"

base_command = (
    "python vanilla_weak_labels.py {ds_name} {model} --target_accuracy {target_accuracy} "
    "--run_name {name} --eval_steps 5 --n_val 500"
)

for ds_name in base_ds_names:
    for original_weak_model in weak_models:
        weak_name = f"{ds_name}_{original_weak_model.split('/')[-1]}"
        ds = load_from_disk(f"results/{weak_name}/weak_train")
        acc = sum(
            (ex["soft_pred"][1] > 0.5) == ex["hard_label"] for ex in ds  # type: ignore
        ) / len(ds)
        name = f"{ds_name}_{model.split('/')[-1]}_stopped_at_{original_weak_model.split('/')[-1]}"
        command = base_command.format(
            ds_name=ds_name, model=model, name=name, target_accuracy=acc
        )
        print(command)
