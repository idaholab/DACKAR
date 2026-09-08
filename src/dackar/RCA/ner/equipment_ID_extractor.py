import re
from typing import List, Optional, Pattern, Tuple


def extract_equipment_ids(
    text: str,
    *,
    patterns: Optional[List[str]] = None,
    normalize: bool = True,
    unique: bool = True,
    max_ids: int = 500,
) -> List[str]:
    """
    Extract likely nuclear plant equipment / tag identifiers from free text.

    This helper is intended to populate `NERSeed.equipment_ids` and/or chunk metadata
    for filtering and retrieval (e.g., in Chroma).

    Inputs
    ------
    text:
        Arbitrary text (string). Can be SOP/CR/WO/ECA chunk text.
    patterns:
        Optional list of regex patterns (strings). If omitted, a reasonable default set
        is used. Patterns should contain a single capturing group for the tag itself
        OR match the tag as the full match.
    normalize:
        If True, normalize extracted tags:
          - uppercase
          - collapse whitespace
          - convert underscores/spaces to hyphens where appropriate
          - strip trailing punctuation
    unique:
        If True, return unique tags in first-seen order.
    max_ids:
        Safety limit; stops collecting after this many matches.

    Output
    ------
    List[str]:
        Extracted equipment IDs/tags as strings (e.g., "P-101", "MOV-204A", "PT-1102").

    Notes
    -----
    - Tag naming conventions vary by plant/site. You should tune `patterns` to your
      org's conventions once you see real data.
    - This function is conservative by default and tries to avoid grabbing years or
      generic hyphenated numbers.
    - Recommended usage:
        eq_ids = extract_equipment_ids(chunk_text)
        seed = NERSeed(..., equipment_ids=eq_ids, ...)
    """
    if not text:
        return []

    # Defaults cover common patterns:
    # - Equipment tags like P-101, MOV-204A, HX-10, PT-1102, CV-12, FCV-100, etc.
    # - Optional train suffixes (A/B) and trailing letter suffixes.
    #
    # You can extend with site-specific conventions:
    # - Loop IDs, panel IDs, cable IDs, breaker IDs, etc.
    default_patterns = patterns or [
        # Common equipment tag: 1-6 letters + '-' + 1-6 digits + optional trailing letter(s)
        r"\b([A-Z]{1,6}-\d{1,6}[A-Z]{0,2})\b",

        # With optional middle segment: e.g., AFW-P-101, RHR-MOV-204A
        r"\b([A-Z]{2,6}-[A-Z]{1,6}-\d{1,6}[A-Z]{0,2})\b",

        # Instrument loop-ish tags: e.g., PT-1102, TT-301, DPIT-12, LT-0045
        r"\b([A-Z]{2,6}T-\d{1,6}[A-Z]{0,2})\b",

        # Valve tags commonly: MOV-204A, AOV-10, FCV-100, PCV-22
        r"\b((?:M|A|F|P)?CV-\d{1,6}[A-Z]{0,2})\b",
        r"\b((?:M|A)OV-\d{1,6}[A-Z]{0,2})\b",

        # Breaker / relay-ish tags (site dependent; keep conservative)
        r"\b(BKR-\d{1,6}[A-Z]{0,2})\b",
        r"\b(RLY-\d{1,6}[A-Z]{0,2})\b",
    ]

    compiled: List[Pattern[str]] = [re.compile(p) for p in default_patterns]

    results: List[str] = []
    seen = set()

    def _norm(tag: str) -> str:
        tag = tag.strip()
        # strip surrounding punctuation
        tag = tag.strip(".,;:()[]{}<>\"'`")
        # normalize separators
        tag = re.sub(r"[_\s]+", "-", tag)
        tag = re.sub(r"-{2,}", "-", tag)
        tag = tag.upper()
        return tag

    # Avoid obvious false positives:
    # - years like 2024-01, ranges like 10-15, dates, etc.
    false_positive_patterns: List[Tuple[str, Pattern[str]]] = [
        ("date_like", re.compile(r"^\d{1,4}-\d{1,2}(-\d{1,2})?$")),  # 2024-01-31, 10-15
        ("range_like", re.compile(r"^\d{1,4}-\d{1,4}$")),            # 10-15
    ]

    def _is_false_positive(tag: str) -> bool:
        t = tag.upper()
        for _, fp in false_positive_patterns:
            if fp.match(t):
                return True
        _DOC_REF_PREFIXES = {"CR", "WO", "SOP", "ECA", "MR", "PM", "AR", "CAP", "PER", "ALM", "ANN"}
        prefix = t.split("-")[0]
        if prefix in _DOC_REF_PREFIXES:
            return True
        # Single-letter prefix with tiny numbers can be noisy (e.g., A-1)
        if re.match(r"^[A-Z]-\d{1,2}$", t):
            return True
        return False

    count = 0
    for rx in compiled:
        for m in rx.finditer(text):
            # If pattern has capturing groups, take the first non-empty group; else full match.
            tag = ""
            if m.groups():
                for g in m.groups():
                    if g:
                        tag = g
                        break
            else:
                tag = m.group(0)

            if not tag:
                continue

            tag = _norm(tag) if normalize else tag

            if _is_false_positive(tag):
                continue

            if unique:
                if tag in seen:
                    continue
                seen.add(tag)

            results.append(tag)
            count += 1
            if count >= max_ids:
                break

    # Containment filter: drop shorter tags that are trailing segments of a longer tag.
    # E.g. if "AFW-P-101" is found, drop "P-101" since it's a suffix component of it.
    filtered: List[str] = []
    result_set = set(results)
    for tag in results:
        absorbed = any(
            longer != tag and longer.endswith("-" + tag)
            for longer in result_set
        )
        if not absorbed:
            filtered.append(tag)
    return filtered
