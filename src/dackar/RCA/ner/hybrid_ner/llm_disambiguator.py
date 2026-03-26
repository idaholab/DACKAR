# file: hybrid_ner/llm_disambiguator.py
from __future__ import annotations
import requests
import subprocess
import json
import time
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import urljoin

# Import your project's LabelHypothesis, CandidateSpan, SchemaIndex
from .models import LabelHypothesis, CandidateSpan
from .schema import SchemaIndex

@dataclass
class LLMConfig:
    use_cli: bool = False                    # Use Ollama CLI (subprocess) instead of HTTP
    cli_binary: str = "ollama"               # CLI binary name
    http_url: str = "http://localhost:11434/v1/chat/completions"  # Example Ollama HTTP endpoint
    model: str = "ollama/gpt-oss:20B"            # Model name to call
    timeout: int = 10
    temperature: float = 0.0                # deterministic
    max_tokens: int = 64
    stop_sequences: Optional[List[str]] = None
    # Controls
    min_confidence: float = 0.15            # attach hypotheses with at least this soft score (if LLM emits a score)
    cache_ttl_seconds: int = 3600
    dry_run: bool = False                   # if true, do not call the model; return NO_LABEL

class LLMDisambiguator:
    def __init__(self, schema: SchemaIndex, config: LLMConfig = LLMConfig()):
        self.schema = schema
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}  # key -> {"time":ts, "resp":...}
        self.llm_ok = self.health_check()
        if not self.llm_ok:
            return

    # Decide if we should call the LLM for candidate `c` (you can adjust heuristics)
    def should_call(self, c: CandidateSpan) -> bool:
        # call if no proposed labels or all hypotheses lack a known group
        if not getattr(c, "proposed_labels", None):
            return True
        if all(getattr(h, "group", None) is None for h in c.proposed_labels):
            return True
        # or ambiguity: multiple groups and no clear winner (simple heuristic)
        groups = {getattr(h, "group", None) for h in c.proposed_labels if getattr(h, "group", None)}
        if len(groups) > 1:
            return True
        return False

    # Build a concise prompt. Keep it short and constrained.
    def _build_prompt(self, doc_text: str, c: CandidateSpan, candidate_labels: List[str]) -> str:
        ctx = doc_text[max(0, c.start - 200): c.end + 200]  # short surrounding context
        label_info = "\n".join([f"- {lbl}: {self._label_short(lbl)}" for lbl in candidate_labels])
        prompt = f"""
        You are a domain expert restricted to choosing one label from a provided list.
        Context: "{ctx}"
        Span: "{c.text}"
        Possible labels:
        {label_info}

        Instruction: Choose exactly one label id from the list above that best fits the span in this context, or reply EXACTLY "NO_LABEL" if none apply. Answer with a JSON object only:
        {{"label":"<label_id_or_NO_LABEL>", "score":<0.0-1.0 optional>, "rationale":"<short explanation>"}}
        Do NOT invent labels. Do NOT provide additional text.
        """
        return prompt.strip()

    def _label_short(self, lbl: str) -> str:
        # Provide a short description from schema if available (fallback to label)
        # The schema in your repo stores descriptions in the original json loaded by DescriptionEmbedGenerator.
        # If not present, return label.
        entry = getattr(self.schema, "label_descriptions", {}) or {}
        if lbl in entry:
            return entry[lbl].get("short_description") or entry[lbl].get("name") or lbl
        return lbl

    def _cache_key(self, c: CandidateSpan) -> str:
        return f"{c.doc_id}:{c.start}:{c.end}:{c.text}"

    def _extract_json_object(self, s: str) -> Optional[Dict[str, Any]]:
        """
        Some models occasionally wrap JSON with extra text.
        Try to recover the first {...} object.
        """
        if not s:
            return None
        s = s.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                pass
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    # Main entry: disambiguate a list of candidates, augment their proposed_labels with llm hypotheses
    def disambiguate(self, doc_text: str, candidates: List[CandidateSpan]) -> None:
        for c in candidates:
            if not self.should_call(c):
                continue
            key = self._cache_key(c)
            now = time.time()
            if key in self._cache and now - self._cache[key]["time"] < self.config.cache_ttl_seconds:
                resp = self._cache[key]["resp"]
            else:
                resp = self._call_llm_for_candidate(doc_text, c)
                self._cache[key] = {"time": now, "resp": resp}
            # parse & attach
            hypothesis = self._parse_llm_response(resp, c)
            if hypothesis:
                # Only attach if label is known in schema
                if hypothesis.label in self.schema.label_to_group:
                    # assign group
                    hypothesis.group = self.schema.label_to_group[hypothesis.label]
                    c.proposed_labels.append(hypothesis)

    def _call_llm_for_candidate(self, doc_text: str, c: CandidateSpan) -> Optional[Dict[str, Any]]:
        if self.config.dry_run:
            return {"label": "NO_LABEL", "score": 0.0, "rationale": "dry_run"}

        # Candidate label set: use schema labels that are plausible (all labels in schema)
        candidate_labels = list(self.schema.label_to_group.keys())  # or narrow set if desired
        prompt = self._build_prompt(doc_text, c, candidate_labels)

        if self.config.use_cli:
            # example: ollama generate MODEL --prompt '...' --json
            cmd = [self.config.cli_binary, "generate", self.config.model, "--prompt", prompt, "--json"]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.config.timeout)
                text = p.stdout.strip()
                return json.loads(text) if text else {"label": "NO_LABEL", "score": 0.0, "rationale": "empty"}
            except Exception as e:
                return {"label": "NO_LABEL", "score": 0.0, "rationale": f"error:{e}"}
        else:
            try:
                url = (self.config.http_url or "").strip()
                # OpenAI-compatible chat endpoint
                if url.endswith("/chat/completions"):
                    payload = {
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": "Return ONLY valid JSON. No extra text."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    }
                    r = requests.post(url, json=payload, timeout=self.config.timeout)
                    r.raise_for_status()
                    raw = r.json()
                    content = (
                        raw.get("choices", [{}])[0]
                           .get("message", {})
                           .get("content", "")
                    )
                    parsed = self._extract_json_object(content)
                    return parsed if parsed is not None else {"label": "NO_LABEL", "score": 0.0, "rationale": "unparseable_chat_json"}

                # OpenAI-compatible legacy completions endpoint
                if url.endswith("/completions"):
                    payload = {
                        "model": self.config.model,
                        "prompt": prompt,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    }
                    r = requests.post(url, json=payload, timeout=self.config.timeout)
                    r.raise_for_status()
                    raw = r.json()
                    text = (raw.get("choices", [{}])[0].get("text", "") or "")
                    parsed = self._extract_json_object(text)
                    return parsed if parsed is not None else {"label": "NO_LABEL", "score": 0.0, "rationale": "unparseable_completion_json"}

                # Fallback: unknown endpoint; try prompt-style (your old behavior)
                payload = {
                    "model": self.config.model,
                    "prompt": prompt,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                r = requests.post(url, json=payload, timeout=self.config.timeout)

                r.raise_for_status()
                return r.json()
            except Exception as e:
                return {"label": "NO_LABEL", "score": 0.0, "rationale": f"error:{e}"}

    def _parse_llm_response(self, resp: Any, c: CandidateSpan) -> Optional[LabelHypothesis]:
        # Expected resp is a dict containing {"label":..., "score":..., "rationale":...}
        if not isinstance(resp, dict):
            # attempt hydration if resp is string
            try:
                resp = json.loads(str(resp))
            except Exception:
                return None
        label = resp.get("label") or resp.get("label_id")
        if not label or label == "NO_LABEL":
            return None
        score = float(resp.get("score", 0.5))
        rationale = str(resp.get("rationale", "")).strip()
        # Build a LabelHypothesis (adapt to your models.py constructor)
        lh = LabelHypothesis(label=label, score=score, rationale=rationale)
        return lh


    def health_check(self) -> bool:
        try:
            # derive base API URL from configured endpoint
            base = self.config.http_url.rsplit("/", 2)[0]  # -> http://localhost:11434/v1
            models_url = urljoin(base + "/", "models")

            r = requests.get(models_url, timeout=3)
            r.raise_for_status()

            models = [m["id"] for m in r.json().get("data", [])]
            if self.config.model not in models:
                raise RuntimeError(
                    f"Model '{self.config.model}' not found. Available: {models}"
                )
            return True
        except Exception as e:
            import logging
            logging.error(f"LLM health check failed: {e}")
            return False
