"""Steering vector training loss (Dunefsky & Cohan 2025).

Given:
    base model B, thinking model T, both sharing tokenizer
    a category-c training example: (context_tokens, continuation_tokens)
    a candidate steering vector v ∈ R^{d_model}
    layer L where v is injected

Loss = − log p_T(continuation | context, steering=v at L)
       + λ · log p_B(continuation | context, steering=v at L)

We MAXIMIZE the thinking-model likelihood while MINIMIZING the base-model
likelihood of the same continuation. The optimization is over v with both
model weights frozen.

Implementation: register a forward hook on layer L that adds α·v to
residual stream, run both models, compute teacher-forced NLL on continuation tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SteeringHook:
    """Adds a vector to the residual stream output of a target decoder layer."""

    def __init__(self, vector: torch.Tensor, scale: float = 1.0):
        self.vector = vector
        self.scale = scale
        self.handle = None

    def install(self, layer: nn.Module) -> None:
        def hook(_module, _inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            # vector: (d_model,) → broadcasts over (B, T, d_model)
            hs = hs + self.scale * self.vector.to(hs.dtype).to(hs.device)
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs
        self.handle = layer.register_forward_hook(hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def teacher_forced_nll(
    model, input_ids: torch.Tensor, continuation_start: int,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return mean negative log-likelihood over continuation tokens.

    input_ids: (B, T)
    continuation_start: index where the continuation begins.
    Returns None when there are no valid continuation tokens (so the caller can skip).
    """
    T = input_ids.shape[1]
    if continuation_start >= T or continuation_start < 1:
        return None
    logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
    pred = logits[:, continuation_start - 1:T - 1, :]   # (B, L, V)
    target = input_ids[:, continuation_start:T]         # (B, L)
    if pred.shape[1] == 0 or target.shape[1] == 0:
        return None
    log_probs = torch.log_softmax(pred.float(), dim=-1)
    nll = -log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


class SteeringVector(nn.Module):
    """Trainable steering vector for one category."""

    def __init__(self, d_model: int, init_std: float = 1e-3):
        super().__init__()
        self.v = nn.Parameter(torch.randn(d_model) * init_std)

    def forward(self) -> torch.Tensor:
        return self.v
