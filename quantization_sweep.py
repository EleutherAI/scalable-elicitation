models = [
    "meta-llama/Meta-Llama-3-8B",
    "qwen/Qwen1.5-7B",
]
seeds = [1, 2]
ds_names = ["amazon_polarity", "boolq"]

base_command = "python run_simple_sft.py {ds_name} {model} 8000 1000 1000 --results_folder results/quantization/{name} --load_best_model_at_end --seed {seed} --run_name {name}"  # noqa

for model in models:
    for seed in seeds:
        for ds_name in ds_names:
            for quantize in [True, False]:
                name = f"{ds_name}_{model.split('/')[-1]}_s{seed}{'_quant' if quantize else ''}"
                command = base_command.format(
                    ds_name=ds_name, model=model, seed=seed, name=name
                )
                if quantize:
                    command += " --quantize"
                print(command)
