from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Works for Llama, Mistral, and Qwen architectures
DEFAULT_LORA_MODULES = [
    "gate_proj",
    "down_proj",
    "up_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


@dataclass
class PredictorConfig(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        ...


@dataclass
class ModelConfig(PredictorConfig):
    name: str
    enable_lora: bool
    model_class: type
    lora_modules: Optional[List[str]] = None
    num_heads: int = 1
    quantize: bool = False
    max_ctx: int = 8192

    def to_dict(self):
        d = vars(self).copy()
        d["model_class"] = self.model_class.__name__
        return d

    def initialize_model(self):
        return self.model_class(self)


class AutoCastingScore(torch.nn.Module):
    def __init__(
        self, score: torch.nn.Linear, output_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        # make a leaf tensor with the same data as score
        self.weight = torch.nn.Parameter(score.weight.to(torch.float32).data)
        self.output_dtype = output_dtype

    def forward(self, hiddens):
        return torch.nn.functional.linear(
            hiddens.to(self.weight.dtype), self.weight, None
        ).to(self.output_dtype)


class MultiHeadAutoCastingScore(torch.nn.Module):
    def __init__(
        self,
        score: torch.nn.Linear,
        output_dtype: torch.dtype = torch.bfloat16,
        num_heads: int = 12,
    ):
        super().__init__()
        # shuffle the weights to make sure the heads aren't identical, but to
        # keep approximately the same initialization distribution
        weight = torch.stack(
            [
                score.weight[torch.randperm(score.weight.size(0))]
                .to(torch.float32)
                .data
                for _ in range(num_heads)
            ],
            dim=1,
        )

        self.weight = torch.nn.Parameter(weight)

        self.output_dtype = output_dtype

    def forward(self, hiddens):
        return torch.einsum(
            "btd,khd->bthk", hiddens.to(self.weight.dtype), self.weight
        ).to(self.output_dtype)


class Predictor(torch.nn.Module, ABC):
    """
    The strong "predictor", using the terminology of the original ELK report
    https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/edit#heading=h.kkaua0hwmp1d
    this is the model we would like to elicit latent knowledge from using the reporter
    """

    cfg: PredictorConfig

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg

    def __call__(
        self, inputs, output_hidden_states=False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor]]]:
        """
        This takes in a batch of inputs and returns the logodds of the model's predictions.
        If output_hidden_states is True, it also returns the hidden states of the model (second)
        Each of the `num_layers` hiddens are tensors of shape [n, hidden_size]
        """
        ...

    def to_dict(self) -> dict[str, str | int | float]:
        """A summary of the method that approximately uniquely identifies it.
        It should include a name and all the important hyperparameters."""
        ...


class TransformerPredictor(Predictor):
    cfg: ModelConfig

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.name,
            device_map={"": "cuda"},
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if cfg.quantize
            else None,
            ignore_mismatched_sizes=True,
        )
        if cfg.quantize and cfg.enable_lora:
            model = prepare_model_for_kbit_training(model)

        if cfg.lora_modules is None and cfg.enable_lora:
            cfg.lora_modules = MODEL_REGISTRY.get(cfg.name, {}).get(
                "lora_modules", DEFAULT_LORA_MODULES
            )

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.name, model_max_length=cfg.max_ctx
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore
        model.score.weight.data *= 0.01
        model.config.problem_type = "single_label_classification"

        if cfg.enable_lora:
            lora_cfg = LoraConfig(
                target_modules=cfg.lora_modules,
                task_type=TaskType.SEQ_CLS,
            )

            # NOTE: adding task_type causes dtype errors, but is necessary for proper module saving
            # and for making the lm head trainable, so we need to wrap it in an AutoCastingScore
            for attr in ["score", "classifier"]:
                if hasattr(model, attr):
                    score = (
                        MultiHeadAutoCastingScore(
                            getattr(model, attr),
                            output_dtype=model.dtype,
                            num_heads=cfg.num_heads,
                        )
                        if cfg.num_heads > 1
                        else AutoCastingScore(
                            getattr(model, attr), output_dtype=model.dtype
                        )
                    )

                    setattr(model, attr, score)
                    break
            else:
                raise ValueError("Could not find classifier head in model.")
            model = get_peft_model(model, lora_cfg)

        # put all the trainable (e.g. LoRA) parameters in float32
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.transformer = model
        self.tokenizer = tokenizer
        self.cfg = cfg

    def to_dict(self):
        return self.cfg.to_dict()


class LMPredictor(Predictor):
    cfg: ModelConfig

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        model = AutoModelForCausalLM.from_pretrained(
            cfg.name,
            device_map={"": "cuda"},
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if cfg.quantize
            else None,
            ignore_mismatched_sizes=True,
        )
        if cfg.quantize and cfg.enable_lora:
            model = prepare_model_for_kbit_training(model)

        if cfg.lora_modules is None and cfg.enable_lora:
            cfg.lora_modules = MODEL_REGISTRY.get(cfg.name, {}).get(
                "lora_modules", DEFAULT_LORA_MODULES
            )

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.name, model_max_length=cfg.max_ctx
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore

        if cfg.enable_lora:
            lora_cfg = LoraConfig(
                target_modules=cfg.lora_modules,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_cfg)

        # put all the trainable (e.g. LoRA) parameters in float32
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.transformer = model
        self.tokenizer = tokenizer
        self.cfg = cfg


# TODO: make a legitimate model registry
# for now we just have a map from model name to learning rate and lora modules
MODEL_REGISTRY = {
    "meta-llama/Meta-Llama-3-8B": {
        "lr": 8e-5,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "meta-llama/Llama-2-7B-hf": {
        "lr": 8e-5,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "meta-llama/Meta-Llama-3-70B": {
        "lr": 4e-5,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "mistralai/Mistral-7B-v0.1": {
        "lr": 8e-5,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "gemma/gemma-7b": {
        "lr": 8e-5,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "Qwen/Qwen1.5-0.5B": {
        "lr": 5e-4,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "Qwen/Qwen1.5-4B": {
        "lr": 2e-4,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "Qwen/Qwen1.5-7B": {
        "lr": 8e-5,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "meta-llama/Meta-Llama-3.1-8B": {
        "lr": 2e-4,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "meta-llama/Meta-Llama-3.1-70B": {
        "lr": 2e-5,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
}
