models = [
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
]
ds_names = [
    "sciq"
]  # "boolq", "anli-r2", "ethics-virtue", "ethics-utilitarianism", "ethics-justice", "hellaswag",
# "amazon_polarity", "ethics-deontology", "paws", "sciq_with_support",

base_command = (
    "python vanilla_weak_labels.py {ds_name} {model} --also_save_shuffled_error_labels"
)

for model in models:
    for ds_name in ds_names:
        command = base_command.format(ds_name=ds_name, model=model)
        print(command)
