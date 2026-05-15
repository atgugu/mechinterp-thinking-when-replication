"""Top-K Sparse Autoencoder with restricted (unit-normalized) decoder.

Used for unsupervised reasoning-category discovery as in
Venhoff et al. 2025 (arxiv 2510.07364). The restricted decoder space
forces decoder columns to lie on the unit sphere; combined with Top-K=3
sparsity, the SAE behaves as a soft clustering of input activations,
where the strongest-firing feature per input is its category assignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """Top-K SAE with unit-norm decoder columns.

    encoder:  z = relu( W_enc @ (x - b_pre) )
    sparsify: z' = top_k(z, k)
    decoder:  x_hat = b_pre + W_dec @ z'   with  ||W_dec[:, i]||_2 == 1

    Loss: ||x - x_hat||^2  (plus optional aux loss to revive dead features).
    """

    def __init__(self, d_input: int, dict_size: int, k: int = 3,
                 normalize_decoder: bool = True,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_input = d_input
        self.dict_size = dict_size
        self.k = k
        self.normalize_decoder = normalize_decoder

        # Encoder: d_input -> dict_size
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_input, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(dict_size, dtype=dtype))
        # Decoder: dict_size -> d_input
        self.W_dec = nn.Parameter(torch.empty(d_input, dict_size, dtype=dtype))
        # Centering bias
        self.b_pre = nn.Parameter(torch.zeros(d_input, dtype=dtype))

        self._init_weights()

    def _init_weights(self) -> None:
        # Kaiming for encoder
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        # Decoder = encoder^T initially (Anthropic SAE recipe)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            if self.normalize_decoder:
                self._normalize_decoder_inplace()

    def _normalize_decoder_inplace(self) -> None:
        with torch.no_grad():
            norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-6)
            self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return Top-K sparse latent activations. x: (B, d_input) → (B, dict_size)."""
        x_c = x - self.b_pre
        pre = F.linear(x_c, self.W_enc, self.b_enc)
        z = F.relu(pre)
        if self.k < self.dict_size:
            topk = torch.topk(z, self.k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, topk.indices, 1.0)
            z = z * mask
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.W_dec) + self.b_pre  # (B, d_input)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        if self.normalize_decoder:
            self._normalize_decoder_inplace()

    @torch.no_grad()
    def assign_category(self, x: torch.Tensor) -> torch.Tensor:
        """Hard assignment: top-1 feature index per input. x: (B, d_input) → (B,) int64."""
        z = self.encode(x)
        return z.argmax(dim=-1)
