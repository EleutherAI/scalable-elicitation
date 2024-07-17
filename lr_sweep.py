models = {
    "meta-llama/Meta-Llama-3-8B": [8e-5, 2e-4, 5e-4],
    "Qwen/Qwen1.5-0.5B": [5e-4, 1e-3, 3e-3],
    "Qwen/Qwen1.5-4B": [2e-4, 5e-4, 1e-3],
    "Qwen/Qwen1.5-7B": [8e-5, 2e-4, 5e-4],
}
seeds = [
    1,
]
ds_names = ["amazon_polarity", "boolq"]

base_command = "python run_simple_sft.py {ds_name} {model} 10000 1000 0 --results_folder results/lr_sweep/{name} --load_best_model_at_end --seed {seed} --run_name {name} --learning_rate {lr} --num_train_epochs 2 "  # noqa

for seed in seeds:
    for model in models:
        for ds_name in ds_names:
            for lr in models[model]:
                name = f"{ds_name}_{model.split('/')[-1]}_s{seed}_lr{lr}"
                command = base_command.format(
                    ds_name=ds_name, model=model, seed=seed, name=name, lr=lr
                )
                print(command)
