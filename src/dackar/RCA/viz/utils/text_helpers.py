"""Truncation and safe excerpt rendering."""

from __future__ import annotations


def truncate(text: str, max_len: int = 600, suffix: str = "…") -> str:
    """Truncate ``text`` to ``max_len`` characters, appending ``suffix`` if cut.

    @ In, text, str, the text to truncate (falsy input returns "")
    @ In, max_len, int, maximum length of the returned string including suffix
    @ In, suffix, str, appended when the text is truncated
    @ Out, out, str, the (possibly truncated) string
    """
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix
