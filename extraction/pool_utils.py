"""Map sentence spans in raw text to token-index ranges given a tokenizer + raw IDs."""

from typing import Sequence


def map_sentences_to_token_ranges(
    sentences_with_spans: list[tuple[str, int, int]],
    offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Convert character spans to (token_start, token_end_exclusive) using HF offset_mapping.

    Tokens are included if their (char_start, char_end) intersects the sentence span.
    """
    out: list[tuple[int, int]] = []
    for _s, c_start, c_end in sentences_with_spans:
        tok_start = None
        tok_end = None
        for i, (t_start, t_end) in enumerate(offsets):
            if t_end <= c_start:
                continue
            if t_start >= c_end:
                break
            if tok_start is None:
                tok_start = i
            tok_end = i + 1
        if tok_start is not None and tok_end is not None and tok_end > tok_start:
            out.append((tok_start, tok_end))
        else:
            # If a sentence falls entirely in a single token (rare), record a one-tok span
            out.append((0, 0))
    return out


def mean_pool(activations, token_ranges: Sequence[tuple[int, int]]):
    """Mean-pool a (T, D) activation tensor over each token range.

    Returns a (n_sentences, D) tensor. Zero-length ranges yield a zero vector.
    """
    import torch
    if activations.dim() == 3:
        activations = activations[0]  # drop batch dim
    T, D = activations.shape
    out = torch.zeros(len(token_ranges), D, dtype=activations.dtype, device=activations.device)
    for i, (s, e) in enumerate(token_ranges):
        if e > s and s < T:
            e_ = min(e, T)
            out[i] = activations[s:e_].mean(dim=0)
    return out
