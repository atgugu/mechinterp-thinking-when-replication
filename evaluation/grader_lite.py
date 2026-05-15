"""Lightweight math-answer grader for MATH500 / GSM8K.

Avoids latex2sympy2 (broken on Python 3.13). Handles:
  - integers, floats, fractions a/b, percentages
  - \\frac{a}{b}, \\sqrt{a}, \\pi, simple symbolic
  - \\boxed{} extraction
  - tuple / list answers like (2, 3)
"""

from __future__ import annotations

import re
from fractions import Fraction


_REPL_MAP = {
    "\\dfrac": "\\frac",
    "\\tfrac": "\\frac",
    "\\cdot": "*",
    "\\times": "*",
    "\\div": "/",
    "\\left": "",
    "\\right": "",
    "^\\circ": "",
    "\\degree": "",
    "\\%": "/100",
    "%": "/100",
    "\\$": "",
    "$": "",
    "\\,": "",
    "\\!": "",
    "\\;": "",
    "\\ ": "",
    "\\quad": "",
    "\\qquad": "",
    "{,}": ",",
    "\\pi": "pi",
    "\\sqrt": "sqrt",
    "\\theta": "theta",
}


def _strip(s: str) -> str:
    s = str(s).strip()
    # extract last \boxed{} if present
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", s)
    if m:
        s = m[-1]
    # remove latex artifacts
    for k, v in _REPL_MAP.items():
        s = s.replace(k, v)
    # remove \text{...}
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    # remove leading/trailing parens
    s = s.strip().strip("()").strip()
    return s


def _try_frac(s: str) -> Fraction | None:
    s = s.strip()
    # \\frac{a}{b}
    m = re.fullmatch(r"\\?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if m:
        try:
            return Fraction(m.group(1)) / Fraction(m.group(2))
        except Exception:
            return None
    # a/b
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        try:
            return Fraction(m.group(1)) / Fraction(m.group(2))
        except Exception:
            return None
    return None


def _try_float(s: str) -> float | None:
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        pass
    try:
        return float(Fraction(s))
    except Exception:
        pass
    f = _try_frac(s)
    if f is not None:
        return float(f)
    return None


def _try_tuple(s: str) -> list[str] | None:
    if "," not in s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if 2 <= len(parts) <= 6:
        return parts
    return None


def math_equal_lite(pred: str, target: str, tol: float = 1e-4) -> bool:
    if pred is None or target is None:
        return False
    p, t = _strip(str(pred)), _strip(str(target))
    if not p or not t:
        return False
    if p == t:
        return True

    # numeric
    fp, ft = _try_float(p), _try_float(t)
    if fp is not None and ft is not None:
        return abs(fp - ft) <= tol * max(1.0, abs(ft))

    # fraction
    fp2, ft2 = _try_frac(p), _try_frac(t)
    if fp2 is not None and ft2 is not None:
        return fp2 == ft2

    # tuple — match element-wise
    tp, tt = _try_tuple(p), _try_tuple(t)
    if tp is not None and tt is not None and len(tp) == len(tt):
        return all(math_equal_lite(a, b, tol) for a, b in zip(tp, tt))

    # symbolic via sympy (best effort, no latex)
    try:
        from sympy import simplify, sympify
        sp = sympify(p.replace("sqrt", "sqrt"))
        st = sympify(t.replace("sqrt", "sqrt"))
        if simplify(sp - st) == 0:
            return True
    except Exception:
        pass

    return p.lower() == t.lower()


if __name__ == "__main__":
    cases = [
        ("5/2", "2.5", True),
        ("\\frac{1}{2}", "0.5", True),
        ("42", "42", True),
        ("\\boxed{42}", "42", True),
        ("0.5", "1/2", True),
        ("(2, 3)", "(2, 3)", True),
        ("1.0", "2", False),
    ]
    for p, t, exp in cases:
        got = math_equal_lite(p, t)
        print(f"  {p!r:>20} == {t!r:<20} → {got} (expected {exp}) {'✓' if got == exp else '✗'}")
