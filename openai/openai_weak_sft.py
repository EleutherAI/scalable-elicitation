import json
import os
import time

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_from_disk

from openai import OpenAI, RateLimitError  # type: ignore

weak_marginal_costs = [1 / 10]
oracle_spending_fracs = [0.0, 0.05, 0.5, 0.95, 1.0]
oracle_affordables = [16, 64, 256, 1024, 4096]

pairs = []
for weak_marginal_cost in weak_marginal_costs:
    for oracle_affordable in oracle_affordables:
        accs = []
        actual_osfs = []
        for osf in oracle_spending_fracs:
            n_oracle = int(osf * oracle_affordable)
            n_weak = int((oracle_affordable - n_oracle) / weak_marginal_cost)
            n_oracle = min(n_oracle, 23_000)
            pairs.append((n_weak, n_oracle))
pairs.append((0, 8192))
pairs = list(set(pairs))


def upload_files(is_weak, num_weak, num_oracle, num_test, source_dict):
    weak_oracle = "weak" if is_weak else "oracle"
    name = f"{num_weak}-{num_oracle}-{weak_oracle}"
    shuffled_train = source_dict["train"].shuffle(seed=0)
    weak = shuffled_train.select(range(num_weak))
    oracle = shuffled_train.select(range(num_weak, num_weak + num_oracle))
    train = weak if is_weak else oracle
    if is_weak:
        train = train.add_column(
            "label", (np.array(train["weak_prob"]) > 0.5).astype(int).tolist()
        )  # type: ignore
    else:
        train = train.add_column("label", train["hard_label"])  # type: ignore

    def modify_messages(ex, mes_choices=["No", "Yes"], compl_choices=[" No", " Yes"]):
        """
        Modify the messages to have the requested label
        """
        label = mes_choices[ex["label"]]
        mes = ex["messages"]["messages"][-1]
        assert mes["role"] == "assistant"
        assert mes["content"] in mes_choices
        mes["content"] = label
        ex["completion"]["completion"] = compl_choices[ex["label"]]
        return ex

    train = train.map(modify_messages)

    test = source_dict["test"].shuffle(seed=0).select(range(num_test))

    os.makedirs(f"openai/{name}", exist_ok=True)
    with open(f"openai/{name}/{weak_oracle}_train.jsonl", "w") as f:
        f.write("\n".join([json.dumps(d) for d in train["messages"]]))
    with open(f"openai/{name}/{weak_oracle}_test.jsonl", "w") as f:
        f.write("\n".join([json.dumps(d) for d in test["messages"]]))

    train_file = client.files.create(
        file=open(f"openai/{name}/{weak_oracle}_train.jsonl", "rb"), purpose="fine-tune"
    )
    test_file = client.files.create(
        file=open(f"openai/{name}/{weak_oracle}_test.jsonl", "rb"), purpose="fine-tune"
    )
    return train_file, test_file, name


num_test = 1000

splits = dict()
for split in ["train", "val", "test"]:
    splits[split] = load_from_disk(f"openai/weak_{split}")
weak_dict = DatasetDict(
    {
        "train": concatenate_datasets([splits["train"], splits["val"]]).shuffle(seed=0),
        "test": splits["test"],
    }
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("openai/jobs.json", "r") as f:
    jobs = json.load(f)
for num_weak, num_oracle in pairs:
    print(f"TRYING {num_weak} {num_oracle}")
    if num_weak == 0 or num_weak > len(weak_dict["train"]):
        print(f"SKIPPING because num_weak is 0 or too large: {num_weak}")
        continue
    if f"{num_weak}-{num_oracle}-weak" in jobs:
        print(f"SKIPPING because {num_weak}-{num_oracle}-weak was already started")
        continue

    train_file, test_file, name = upload_files(
        True, num_weak, num_oracle, num_test, weak_dict
    )
    while True:
        try:
            job = client.fine_tuning.jobs.create(
                training_file=train_file.id,
                validation_file=test_file.id,
                model="gpt-4o-mini-2024-07-18",
                suffix=name,
                integrations=[{"type": "wandb", "wandb": {"project": "openai-sft"}}],
            )
            job_id = job.id
            jobs[name] = {
                "job": json.loads(job.model_dump_json()),
                "train_file_id": train_file.id,
                "val_file_id": test_file.id,
            }
            break
        except RateLimitError as e:
            print(f"Rate limit error at {time.time()}: {e}")
            time.sleep(120)
            continue
        except Exception as e:
            print(f"Error: {e}")

            time.sleep(120)
            continue
    with open("openai/jobs.json", "w") as f:
        f.write(json.dumps(jobs))
