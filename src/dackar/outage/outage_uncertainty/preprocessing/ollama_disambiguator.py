"""
LLM-backed abbreviation disambiguator for nuclear outage activity descriptions.

Role in the preprocessing pipeline
------------------------------------
This component sits *after* :class:`AbbreviationResolver` and handles the
residual tokens that the rule-based system could not expand — typically
plant-specific or outage-specific shorthand that is absent from both the
DACKAR Excel file and the nuclear supplement.

A token is considered *unresolved* if, after rule-based expansion, it still
matches the pattern of a P6-scheduler abbreviation:

* 2–8 characters
* All alphabetic (no digits, no hyphens → not a component tag like "PT-455A")
* All-uppercase (P6 convention for abbreviations)

For each such token the disambiguator builds a one-shot prompt that includes
the *full original description* as context (so the LLM can infer meaning from
surrounding words) and a small candidate set drawn from the known nuclear
vocabulary.  The resolved expansion replaces the uppercase token in the output.

Design choices
--------------
* **Conservative triggering** — only uppercase 2–8-char alpha tokens trigger
  an LLM call.  Mixed-case or already-expanded tokens are never sent.
* **Two-level cache** — in-memory dict + optional JSON file.  Cache key is the
  raw token (case-insensitive).  Context is NOT part of the key: the same
  abbreviation always expands to the same term in a nuclear outage context.
* **Disabled by default** — the facade enables this only when
  ``AppConfig.llm_disambiguation_enabled=True`` so the service remains fully
  functional without Ollama running.
* **Graceful degradation** — any Ollama failure (timeout, unreachable) returns
  the original token unchanged and logs a warning.

Model recommendation
--------------------
``mistral:latest`` (7B) correctly resolves standard nuclear abbreviations
(MOV, RHR, EDGR, …) and outperforms ``llama3.2:3b`` (57 %) on the benchmark
disambiguation cases.  ``phi4:14b`` or ``gpt-oss:20b`` would give higher
accuracy at the cost of ~3–10× latency per call.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex for tokens that look like unresolved P6 abbreviations:
# all-uppercase, 2–8 alphabetic characters only.
_ABBR_RE = re.compile(r"^[A-Z]{2,8}$")

_PROMPT_TEMPLATE = """\
You are a nuclear power plant maintenance expert.
An activity description from a P6 outage schedule contains the abbreviation "{token}".
The full description is: "{description}"

Select the most appropriate expansion of "{token}" from the candidates below.
Reply with ONLY the chosen expansion text, nothing else.

Candidates: {candidates}
"""

# Candidate pool drawn from the nuclear vocabulary — offered to the LLM as
# multiple-choice options so it does not hallucinate free-form expansions.
_CANDIDATE_POOL: list[str] = [
    "motor operated valve", "air operated valve", "check valve",
    "safety injection", "residual heat removal", "reactor coolant pump",
    "main coolant pump", "emergency core cooling system",
    "pressurizer", "containment", "feedwater", "transmitter",
    "transformer", "pressure relief valve",
    "preventive maintenance", "corrective maintenance",
    "work order", "work control order",
    "technical specification", "surveillance",
    "emergency diesel generator", "diesel generator",
    "radiation protection", "non-destructive examination",
    "auxiliary feedwater", "main feedwater",
    "component cooling water", "service water pump",
    "heat exchanger", "instrumentation and controls",
    "calibrate", "replace", "inspect", "overhaul",
    "mechanical", "electrical", "instrumentation",
]

# How many candidates to include per prompt (top-N most likely given the token)
_N_CANDIDATES = 6


def _select_candidates(token: str) -> list[str]:
    """Pick the N most plausible candidates for *token* using simple overlap."""
    tok_lower = token.lower()

    def score(c: str) -> int:
        # Prefer candidates whose words start with letters from the abbreviation
        words = c.split()
        initials = "".join(w[0] for w in words if w)
        overlap = sum(1 for ch in tok_lower if ch in initials)
        return overlap

    ranked = sorted(_CANDIDATE_POOL, key=score, reverse=True)
    return ranked[:_N_CANDIDATES]


class LLMAbbreviationDisambiguator:
    """Expand residual uppercase abbreviations using a local Ollama LLM.

    This component wraps :class:`AbbreviationResolver` and fires an LLM call
    only for tokens that remain unresolved after rule-based expansion.

    Args:
        abbreviation_resolver: The upstream rule-based resolver.  Its
            :meth:`transform` method is called first on every input.
        model: Ollama model name.  Default ``"mistral:latest"``.
        base_url: Ollama HTTP API base URL.
        cache_path: Optional path to a JSON file for persistent token cache.
        timeout: HTTP timeout in seconds per LLM call.
    """

    def __init__(
        self,
        abbreviation_resolver,
        model: str = "mistral:latest",
        base_url: str = "http://localhost:11434",
        cache_path: str | None = None,
        timeout: int = 30,
    ) -> None:
        self._resolver = abbreviation_resolver
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache: dict[str, str] = {}   # token.upper() → expansion
        self._cache_path = cache_path

        if cache_path and Path(cache_path).exists():
            self._load_cache(cache_path)

        logger.info(
            "LLMAbbreviationDisambiguator: model=%s, cache_entries=%d",
            model,
            len(self._cache),
        )

    # ------------------------------------------------------------------
    # Public interface (same as AbbreviationResolver)
    # ------------------------------------------------------------------

    def transform(self, text: str) -> str:
        """Return *text* with abbreviations expanded.

        First applies rule-based expansion via the upstream resolver, then
        scans the result for remaining uppercase abbreviation tokens and
        resolves them using the LLM.
        """
        if not text:
            return text

        # Step 1: rule-based expansion (fast, always applied)
        expanded = self._resolver.transform(text)

        # Step 2: find remaining uppercase tokens that look like abbreviations
        tokens = expanded.split()
        resolved: list[str] = []
        for token in tokens:
            if _ABBR_RE.match(token):
                resolved.append(self._resolve_with_llm(token, text))
            else:
                resolved.append(token)

        return " ".join(resolved)

    def cache_size(self) -> int:
        return len(self._cache)

    def save_cache(self) -> None:
        """Flush the in-memory cache to the JSON file specified at construction."""
        if not self._cache_path:
            return
        path = Path(self._cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)
        logger.info(
            "LLMAbbreviationDisambiguator: saved %d cache entries to %s",
            len(self._cache),
            self._cache_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_with_llm(self, token: str, original_text: str) -> str:
        """Return the LLM expansion of *token*, or *token* unchanged on failure."""
        cache_key = token.upper()
        if cache_key in self._cache:
            return self._cache[cache_key]

        candidates = _select_candidates(token)
        prompt = _PROMPT_TEMPLATE.format(
            token=token,
            description=original_text,
            candidates=", ".join(f'"{c}"' for c in candidates),
        )

        expansion = self._call_llm(prompt, token)
        self._cache[cache_key] = expansion

        if expansion != token:
            logger.debug(
                "LLMAbbreviationDisambiguator: '%s' → '%s'", token, expansion
            )

        return expansion

    def _call_llm(self, prompt: str, fallback: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
                answer = data.get("response", "").strip().strip('"').lower()
                # Validate: answer must be one of our candidates (or a prefix match)
                candidates = _select_candidates(fallback)
                for c in candidates:
                    if answer == c or c.startswith(answer) or answer.startswith(c[:6]):
                        return c
                # If LLM went off-script, return fallback unchanged
                logger.debug(
                    "LLMAbbreviationDisambiguator: LLM response '%s' not in "
                    "candidate set for token '%s'; keeping original",
                    answer,
                    fallback,
                )
                return fallback
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "LLMAbbreviationDisambiguator: Ollama call failed for token '%s' (%s); "
                "keeping original",
                fallback,
                exc,
            )
            return fallback
        except (KeyError, json.JSONDecodeError) as exc:
            logger.warning(
                "LLMAbbreviationDisambiguator: unexpected response for token '%s' (%s); "
                "keeping original",
                fallback,
                exc,
            )
            return fallback

    def _load_cache(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as fh:
                self._cache = json.load(fh)
            logger.info(
                "LLMAbbreviationDisambiguator: loaded %d cache entries from %s",
                len(self._cache),
                path,
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "LLMAbbreviationDisambiguator: could not load cache from %s (%s)",
                path,
                exc,
            )
