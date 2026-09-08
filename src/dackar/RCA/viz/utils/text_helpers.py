"""Truncation and safe excerpt rendering."""

from __future__ import annotations


def truncate(text: str, max_len: int = 600, suffix: str = "…") -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix
