"""Split reasoning traces into sentences with math-aware regex.

Preserves $...$, \\(...\\), and \\begin{align}...\\end{align} blocks
so we don't fragment equations.
"""

import re
from typing import Iterable

# Protect math blocks before sentence splitting.
_MATH_PATTERNS = [
    (re.compile(r"\$\$.+?\$\$", re.DOTALL), "DISPLAYMATH"),
    (re.compile(r"\$[^\$]+\$"), "INLINEMATH"),
    (re.compile(r"\\\(.+?\\\)", re.DOTALL), "PARMATH"),
    (re.compile(r"\\\[.+?\\\]", re.DOTALL), "BRMATH"),
    (re.compile(r"\\begin\{align\*?\}.+?\\end\{align\*?\}", re.DOTALL), "ALIGN"),
    (re.compile(r"\\begin\{equation\*?\}.+?\\end\{equation\*?\}", re.DOTALL), "EQN"),
    (re.compile(r"\\boxed\{[^{}]*\}"), "BOXED"),
]

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])|\n\n+")


def _protect(text: str) -> tuple[str, dict[str, str]]:
    """Replace math blocks with placeholders. Return (text, restore_map)."""
    restore: dict[str, str] = {}
    counter = [0]
    for pat, tag in _MATH_PATTERNS:
        def repl(m, tag=tag):
            key = f"\x00{tag}{counter[0]}\x00"
            restore[key] = m.group(0)
            counter[0] += 1
            return key
        text = pat.sub(repl, text)
    return text, restore


def _unprotect(text: str, restore: dict[str, str]) -> str:
    for k, v in restore.items():
        text = text.replace(k, v)
    return text


def split_sentences(text: str, min_len: int = 8) -> list[str]:
    """Split text into sentences, preserving math blocks. Drops very short fragments."""
    if not text or not text.strip():
        return []
    protected, restore = _protect(text)
    parts = _SENT_BOUNDARY.split(protected)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        s = _unprotect(p, restore)
        if len(s) < min_len:
            continue
        out.append(s)
    return out


def extract_think_block(rollout_text: str) -> str:
    """Return the content inside <think>...</think>, or the full text if no tags."""
    m = re.search(r"<think>(.*?)</think>", rollout_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # DeepSeek-R1 sometimes omits closing tag → take everything before final answer
    m = re.search(r"<think>(.*)", rollout_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return rollout_text.strip()


def iter_sentences_with_spans(text: str, min_len: int = 8) -> list[tuple[str, int, int]]:
    """Return [(sentence_text, start_char, end_char), ...] aligned to `text`."""
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for s in split_sentences(text, min_len=min_len):
        start = text.find(s, cursor)
        if start < 0:
            # fallback if exact substring not found (very rare)
            continue
        end = start + len(s)
        out.append((s, start, end))
        cursor = end
    return out


if __name__ == "__main__":
    sample = (
        "Let me think about this. The problem says $f(x) = x^2 + 3x$. "
        "First, I should compute $f(2)$. That gives $4 + 6 = 10$. "
        "Wait, let me check that again. Yes, $f(2) = 10$. So the answer is \\boxed{10}."
    )
    sents = split_sentences(sample)
    print(f"Got {len(sents)} sentences:")
    for s in sents:
        print(f"  - {s}")
