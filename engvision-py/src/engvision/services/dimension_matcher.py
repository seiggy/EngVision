"""Fuzzy dimension matching, ported from .NET DimensionMatcher."""

from __future__ import annotations

import re


def are_similar(a: str, b: str) -> bool:
    return confidence_score(a, b) >= 0.75


def confidence_score(a: str | None, b: str | None) -> float:
    if a is None or b is None:
        return 0.0

    na = _normalize_whitespace(a).upper()
    nb = _normalize_whitespace(b).upper()

    if na == nb:
        return 1.0

    max_len = max(len(na), len(nb))
    if max_len == 0:
        return 1.0

    distance = _levenshtein_distance(na, nb)
    return round(1.0 - distance / max_len, 4)


def _is_numeric_token(s: str) -> bool:
    if not s:
        return False
    digit_like = sum(
        1 for c in s
        if c.isdigit() or c in "./-°OoØ⌀"
    )
    return digit_like / len(s) > 0.6


def _normalize_numeric(s: str) -> str:
    chars = list(s)
    for i, c in enumerate(chars):
        if c in ("O", "o"):
            chars[i] = "0"
        elif c in ("I", "l"):
            chars[i] = "1"
        elif c == "S" and _is_digit_context(chars, i):
            chars[i] = "5"
        elif c == "B" and _is_digit_context(chars, i):
            chars[i] = "8"
        elif c in ("Ø", "⌀"):
            chars[i] = "0"
        elif c in ("°", "º"):
            chars[i] = "°"
    return "".join(chars)


def _normalize_mixed(s: str) -> str:
    result = s.upper()
    chars = list(result)
    for i, c in enumerate(chars):
        if c == "O":
            adj_digit = (
                (i > 0 and (chars[i - 1].isdigit() or chars[i - 1] == "."))
                or (i < len(chars) - 1 and (chars[i + 1].isdigit() or chars[i + 1] == "."))
            )
            if adj_digit:
                chars[i] = "0"
    return "".join(chars)


def _is_digit_context(chars: list[str], i: int) -> bool:
    prev = i > 0 and (chars[i - 1].isdigit() or chars[i - 1] == ".")
    nxt = i < len(chars) - 1 and (chars[i + 1].isdigit() or chars[i + 1] == ".")
    return prev and nxt


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _levenshtein_distance(s: str, t: str) -> int:
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m]
