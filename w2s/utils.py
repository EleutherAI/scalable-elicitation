from typing import Any, Literal, Type, TypeVar, cast

import torch
from datasets import Dataset

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        else:
            return str(value)

    config = flatten_dict(config)
    return "-".join(
        f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items())
    )


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def split_args_by_prefix(args: dict, prefixs: tuple) -> dict[str, dict]:
    """Split a dictionary of arguments into sub-dictionaries based on prefixes.
    Keys without a prefix are placed in all sub-dictionaries.
    """
    base_args = {
        k: v for k, v in args.items() if not any(k.startswith(p) for p in prefixs)
    }
    prefix_args = {p: base_args.copy() for p in prefixs}
    for prefix in prefixs:
        prefix_args[prefix].update(
            {k[len(prefix) :]: v for k, v in args.items() if k.startswith(prefix)}
        )
    return prefix_args


def ds_with_labels(ds: Dataset, labels_column: str = "soft_label"):
    if "labels" in ds.column_names:
        ds = ds.remove_columns("labels")
    return ds.add_column(
        "labels", torch.as_tensor(ds[labels_column])[:, 1].tolist()
    )  # type: ignore


def uncertainty_sample(
    probs, n, method: Literal["sample", "hard"], most_confident=False
):
    assert probs.ndim == 2
    entropies = -(probs * torch.log2(probs)).sum(dim=-1)
    if method == "hard":
        idxs = (1 - entropies if most_confident else entropies).argsort()[:n][
            torch.randperm(n)
        ]
    elif method == "sample":
        # hard uncertainty sampling sucks. We want to sample with a bias
        # towards confidently labeled examples. We can do this by sampling with weight 1 - entropy

        # get n_train random indices with replacement, weighted by p_correct
        idxs = torch.multinomial(
            1 - entropies if most_confident else entropies, n, replacement=False
        )
    else:
        raise ValueError(f"Invalid method: {method}")
    return idxs
