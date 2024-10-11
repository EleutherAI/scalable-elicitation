base_ds_names = [
    "boolq",
    "hellaswag",
    "paws",
    "sciq",
    "cola",
    "cosmos_qa",
    "quail",
    "social_i_qa",
]
weak_models = ["Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-4B"]

base_command = "python vanilla_weak_labels.py {ds_name} {model} "

for weak_model in weak_models:
    for ds_name in base_ds_names:
        command = base_command.format(ds_name=ds_name, model=weak_model)
        print(command)
