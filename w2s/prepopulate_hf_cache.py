from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

models = [
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
    "meta-llama/Meta-Llama-3-8B",
]

for model in tqdm(models):
    model = AutoModelForSequenceClassification.from_pretrained(model)
