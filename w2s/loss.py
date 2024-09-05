from typing import Optional

import torch
from einops import rearrange
from transformers import Trainer


class CustomLossTrainer(Trainer):
    def __init__(
        self,
        loss_name: str,
        *args,
        resume_from_optimizer_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.resume_from_optimizer_checkpoint = resume_from_optimizer_checkpoint

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        This loss function does a hacky workaround to optionally
        load in an Adam optimizer state at the start of training runs
        """
        if (
            self.state.global_step == 0
            and self.resume_from_optimizer_checkpoint is not None
            and self.optimizer is not None
        ):
            # check if adam exp buffer is empty, and then load the optimizer state if it is
            if not isinstance(self.optimizer, torch.optim.AdamW):
                assert isinstance(self.optimizer.optimizer, torch.optim.AdamW)
            self.optimizer: torch.optim.AdamW
            state = self.optimizer.state[self.optimizer.param_groups[0]["params"][0]]
            if "exp_avg" not in state:
                # update the step, exp_avg, and exp_avg_sq of the optimizer state
                print(
                    "Loading optimizer state from",
                    self.resume_from_optimizer_checkpoint,
                )
                state_dict = torch.load(
                    self.resume_from_optimizer_checkpoint,
                    map_location=self.model.device,
                )["state"]
                trainable_params = (
                    p for p in self.model.parameters() if p.requires_grad
                )
                for state, p in zip(state_dict.values(), trainable_params):  # type: ignore
                    self.optimizer.state[p] = state  # type: ignore
                self.resume_from_optimizer_checkpoint = None

        if self.state.global_step == 1 and self.optimizer is not None:
            state = self.optimizer.state[self.optimizer.param_groups[0]["params"][1]]
            if "exp_avg" in state and state["exp_avg"].dtype != torch.float32:
                print(f"Adam buffer dtype: {state['exp_avg'].dtype}")

        return self.compute_loss_custom(model, inputs, return_outputs)

    def compute_loss_custom(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()

        outputs = model(**inputs)
        if self.loss_name in {"xent", "kl"}:
            aux_weight = 0
        elif self.loss_name == "logconf":
            aux_weight = 0.5
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

        loss = log_confidence_loss(
            outputs.logits,
            labels,
            self.state.global_step,
            aux_coef=aux_weight,
            subtract_label_ent=self.loss_name == "kl",
        )

        return (loss, outputs) if return_outputs else loss


def mutual_info_loss(probs):
    """Input: predicted probabilites on target batch."""
    B, H, D = probs.shape  # B=batch_size, H=heads, D=pred_dim
    marginal_p = probs.mean(dim=0)  # H, D
    marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)  # H, H, D, D
    marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # H^2, D^2
    joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(dim=0)  # H, H, D, D
    joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # H^2, D^2
    kl_divs = joint_p * (joint_p.log() - marginal_p.log())
    kl_grid = rearrange(kl_divs.sum(dim=-1), "(h g) -> h g", h=H)  # H, H
    pairwise_mis = torch.triu(
        kl_grid, diagonal=1
    )  # Get only off-diagonal KL divergences
    return pairwise_mis.mean()


def log_confidence_loss(
    logits,
    labels,
    step: int,
    # preds_buffer: list,
    # labels_buffer: list,
    warmup_steps: int = 200,
    aux_coef: float = 0.5,
    subtract_label_ent: bool = False,
    preds_buffer_size: int = 32,
):
    """
    logits: [B, 2]
    labels: [B]
    """
    logits = logits.float()
    labels = labels.float()
    # Note that we accumulate all the labels, not just `preds_buffer_size` of them
    prior = labels.mean().item() if len(labels) > 1 else 0.5

    coef = aux_coef * min(1.0, step / warmup_steps)
    preds = torch.softmax(logits, dim=-1)
    threshold = torch.quantile(preds[:, 0], prior)
    strong_preds = torch.cat(
        [(preds[:, 0] >= threshold)[:, None], (preds[:, 0] < threshold)[:, None]],
        dim=1,
    )
    labels_one_hot = torch.stack([1.0 - labels, labels], dim=1)
    target = labels_one_hot * (1 - coef) + strong_preds.detach() * coef
    loss = torch.nn.functional.cross_entropy(logits, target)
    if subtract_label_ent:
        avg_label_ent = -torch.sum(
            labels_one_hot * torch.log(labels_one_hot + 1e-10), dim=1
        ).mean()
        loss = loss - avg_label_ent
    return loss
