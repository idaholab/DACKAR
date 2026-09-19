"""
reliability_summarizer.py
================================

Doc-type aware summarization utilities for nuclear plant reliability documents.

Designed to integrate with your existing pipeline:
- pdfParser.py creates document_index.json (text_md_path, tables_paths, figures, provenance) :contentReference[oaicite:3]{index=3}
- mdParser.py parses Markdown into sections and emits chunks.jsonl :contentReference[oaicite:4]{index=4}

This module provides:
1) detect_doc_type(): Identify SOP/CR/WO/ECA/OTHER from doc_name/source_path and early text
2) section_role_from_title(): Tag sections as purpose/steps/evidence/etc. based on doc type
3) build_ollama_prompt_*(): Exact JSON-output prompts per doc type and view type
4) ollama_generate_json(): Call Ollama and parse JSON robustly
5) quality gates: validate_summary_json(), validate_rca_json()

All outputs are strict JSON dicts, suitable for:
- storing alongside chunk records in chunks.jsonl
- feeding to Chroma embedding pipeline (e.g., embedding flattened JSON)

No external dependencies beyond 'requests' and the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal
import json
import os
import re
import time
import requests


DocType = Literal["SOP", "CR", "WO", "ECA", "OTHER"]
ViewType = Literal["retrieval_summary", "rca_frame"]


# -----------------------------------------------------------------------------
# Data models (lightweight, JSON-first)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkContext:
    """
    Context for a chunk being summarized.

    Input format
    ------------
    - doc_id: str
    - doc_type: DocType
    - chunk_id: str
    - section_path: str (e.g., "7 Procedure > 7.2 Stroke Time Verification")
    - page_start: int
    - page_end: int
    - authority_level: str in {"mandatory","guidance","informational","unknown"}
    - section_role: str (derived label such as "steps", "evidence", "analysis"...)

    Output format
    -------------
    Used to populate the "citations" block of JSON outputs.
    """
    doc_id: str
    doc_type: DocType
    chunk_id: str
    section_path: str
    page_start: int
    page_end: int
    authority_level: str = "unknown"
    section_role: str = "unknown"


@dataclass(frozen=True)
class NERSeed:
    """
    Seed signals derived from your NER layer and tag regex extraction.

    Input format
    ------------
    {
      "systems": [...],              # from NER label "syst" :contentReference[oaicite:5]{index=5}
      "equipment_ids": [...],        # regex-derived (recommend you add)
      "components": [...],           # from ast_*, comp_* labels :contentReference[oaicite:6]{index=6}
      "mechanisms": [...],           # from "deg_mech" :contentReference[oaicite:7]{index=7}
      "outcomes": [...],             # from "fail_type_n" + "event" :contentReference[oaicite:8]{index=8}
      "surveillance_actions": [...], # from surv_ops_v / surv_ops_n :contentReference[oaicite:9]{index=9}
      "maintenance_actions": [...],  # from mnt_ops :contentReference[oaicite:10]{index=10}
      "properties": [...],           # from prop :contentReference[oaicite:11]{index=11}
      "tools": [...]                 # from surv_tool + mnt_tool :contentReference[oaicite:12]{index=12}
    }

    Output format
    -------------
    A JSON-serializable dict used inside prompts. The model is instructed to
    ONLY include items if supported by CHUNK_TEXT.
    """
    systems: List[str]
    equipment_ids: List[str]
    components: List[str]
    mechanisms: List[str]
    outcomes: List[str]
    surveillance_actions: List[str]
    maintenance_actions: List[str]
    properties: List[str]
    tools: List[str]
    fm_ids: List[str] = field(default_factory=list)
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    doc_refs: List[str] = field(default_factory=list)
    alarm_ids: List[str] = field(default_factory=list)
    temporal_refs: List[str] = field(default_factory=list)
    temporal_relations: List[Dict[str, str]] = field(default_factory=list)
    temporal_qualifiers: List[str] = field(default_factory=list)
    locations: List[Dict[str, str]] = field(default_factory=list)
    conjectures: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "systems": self.systems or [],
            "equipment_ids": self.equipment_ids or [],
            "components": self.components or [],
            "mechanisms": self.mechanisms or [],
            "outcomes": self.outcomes or [],
            "surveillance_actions": self.surveillance_actions or [],
            "maintenance_actions": self.maintenance_actions or [],
            "properties": self.properties or [],
            "tools": self.tools or [],
            "doc_refs": self.doc_refs or [],
            "alarm_ids": self.alarm_ids or [],
            "measurements": self.measurements or [],
            "temporal_refs": self.temporal_refs or [],
            "temporal_relations": self.temporal_relations or [],
            "temporal_qualifiers": self.temporal_qualifiers or [],
            "locations": self.locations or [],
            "conjectures": self.conjectures or [],
        }


# -----------------------------------------------------------------------------
# Document type detection (heuristics)
# -----------------------------------------------------------------------------

_DOC_TYPE_HINTS: List[Tuple[DocType, List[str]]] = [
    ("SOP", ["standard operating procedure", "procedure", "operating procedure", "sop"]),
    ("CR", ["condition report", "cr-"]),
    ("WO", ["work order", "wo-"]),
    ("ECA", ["causal evaluation", "event causal analysis", "root cause", "eca-"]),
]


def detect_doc_type(doc_name: Optional[str], source_path: Optional[str], early_text: str) -> DocType:
    """
    Detect document type using filename/path hints + early extracted text.

    Inputs
    ------
    doc_name: optional filename from document_index["doc_name"] :contentReference[oaicite:13]{index=13}
    source_path: optional path from document_index["source_path"] :contentReference[oaicite:14]{index=14}
    early_text: first 2-5k chars of cleaned text from the first parsed section (string)

    Output
    ------
    DocType: "SOP"|"CR"|"WO"|"ECA"|"OTHER"
    """
    hay = " ".join([(doc_name or ""), (source_path or ""), (early_text or "")]).lower()

    # Strong filename/path tokens
    if re.search(r"\b(sop|op)\b[-_ ]?\d+", hay):
        return "SOP"
    if re.search(r"\bcr\b[-_ ]?\d+", hay) or "condition report" in hay:
        return "CR"
    if re.search(r"\bwo\b[-_ ]?\d+", hay) or "work order" in hay:
        return "WO"
    if re.search(r"\beca\b[-_ ]?\d+", hay) or "causal evaluation" in hay or "event causal" in hay:
        return "ECA"

    # General text hints
    for dt, terms in _DOC_TYPE_HINTS:
        for t in terms:
            if t in hay:
                return dt
    return "OTHER"


# -----------------------------------------------------------------------------
# Section role mapping (doc-type specific)
# -----------------------------------------------------------------------------

_ROLE_PATTERNS: Dict[DocType, List[Tuple[str, str]]] = {
    "SOP": [
        ("purpose", r"\bpurpose\b"),
        ("scope", r"\bscope\b|\bapplicab"),
        ("references", r"\breferences?\b"),
        ("definitions", r"\bdefinitions?\b|\bacronyms?\b"),
        ("preconditions", r"\bpreconditions?\b|\bprerequisites?\b|\binitial conditions?\b"),
        ("constraints", r"\bcaution\b|\bwarning\b|\bhold\b|\blimitations?\b"),
        ("steps", r"\bprocedure\b|\bsteps?\b|\binstructions?\b|\bmethod\b"),
        ("acceptance", r"\bacceptance\b|\bverification\b|\bcriteria\b|\btolerances?\b"),
        ("records", r"\brecords?\b|\blogs?\b"),
        ("attachments", r"\battachments?\b|\bappendix\b"),
    ],
    "CR": [
        ("header", r"\bcondition report\b|\bcr\b|\binitiator\b|\bdate\b"),
        ("description", r"\bproblem\b|\bcondition\b|\bdescription\b|\bissue\b"),
        ("immediate_actions", r"\bimmediate actions?\b|\bcompensatory\b"),
        ("impact", r"\bimpact\b|\boperability\b|\bsafety significance\b"),
        ("cause_statement", r"\bapparent cause\b|\bevaluation\b|\bdiscussion\b"),
        ("corrective_actions", r"\bcorrective actions?\b|\bplanned actions?\b"),
        ("evidence", r"\battachments?\b|\bevidence\b|\bdata\b|\bphotos?\b"),
    ],
    "WO": [
        ("header", r"\bwork order\b|\bwo\b|\bpriority\b"),
        ("task", r"\btask\b|\bproblem\b|\bdescription\b|\brequest\b"),
        ("steps", r"\bwork steps?\b|\binstructions?\b|\bprocedure\b"),
        ("as_found", r"\bas[- ]found\b"),
        ("work_performed", r"\bwork performed\b|\bperformed\b|\bcompleted\b"),
        ("as_left", r"\bas[- ]left\b"),
        ("verification", r"\btest\b|\bverification\b|\bresults?\b"),
        ("parts_tools", r"\bparts?\b|\bmaterials?\b|\btools?\b"),
        ("closeout", r"\bcloseout\b|\bnotes?\b"),
    ],
    "ECA": [
        ("event_summary", r"\bevent summary\b|\bsummary\b|\bbackground\b"),
        ("timeline", r"\btimeline\b|\bsequence of events\b"),
        ("evidence_items", r"\bevidence\b|\bdata\b|\binspection\b|\btest results\b"),
        ("rationale", r"\banalysis\b|\bdiscussion\b|\bevaluation\b"),
        ("causal_factors", r"\bcausal factors\b|\broot cause\b|\bcontributing\b"),
        ("actions", r"\bcorrective actions?\b|\bpreventive actions?\b|\brecommendations?\b"),
        ("lessons", r"\blessons learned\b"),
    ],
    "OTHER": [],
}


def section_role_from_title(doc_type: DocType, section_title: str) -> str:
    """
    Infer a section role from its title based on doc type.

    Inputs
    ------
    doc_type: SOP|CR|WO|ECA|OTHER
    section_title: the markdown heading title

    Output
    ------
    role: string (e.g., "steps", "evidence", "analysis", ...) or "unknown"
    """
    title = (section_title or "").strip().lower()
    for role, pat in _ROLE_PATTERNS.get(doc_type, []):
        if re.search(pat, title, flags=re.IGNORECASE):
            return role
    return "unknown"


# -----------------------------------------------------------------------------
# JSON output contracts (keys/types)
# -----------------------------------------------------------------------------

def empty_retrieval_summary(ctx: ChunkContext) -> Dict[str, Any]:
    return {
        "chunk_id": ctx.chunk_id,
        "doc_type": ctx.doc_type,
        "view_type": "retrieval_summary",
        "scope": "",
        "entities": {"systems": [], "equipment_ids": [], "components": []},
        "symptoms_outcomes": [],
        "mechanisms": [],
        "diagnostics": [],
        "corrective_actions": [],
        "numbers_limits": [],
        "keywords_synonyms": [],
        "unknowns": [],
        "citations": {
            "doc_id": ctx.doc_id,
            "section_path": ctx.section_path,
            "page_start": int(ctx.page_start),
            "page_end": int(ctx.page_end),
        },
    }


def empty_rca_frame(ctx: ChunkContext) -> Dict[str, Any]:
    return {
        "chunk_id": ctx.chunk_id,
        "doc_type": ctx.doc_type,
        "view_type": "rca_frame",
        "observed": [],
        "hypotheses": [],
        "tests_to_confirm": [],
        "candidate_actions": [],
        "constraints": [],
        "citations": {
            "doc_id": ctx.doc_id,
            "section_path": ctx.section_path,
            "page_start": int(ctx.page_start),
            "page_end": int(ctx.page_end),
        },
    }


# -----------------------------------------------------------------------------
# Prompt builders (EXACT JSON-output prompts)
# -----------------------------------------------------------------------------

_COMMON_SYSTEM_CONTRACT = """You are an information extraction engine for nuclear plant reliability documents.
Return ONLY valid JSON. Do not include markdown, comments, or extra text.

Rules:
- Use ONLY facts explicitly present in the provided CHUNK_TEXT.
- If a field cannot be filled from CHUNK_TEXT, use an empty array [] and add a short note to unknowns (retrieval_summary only).
- Preserve numbers, limits, units, and step numbers exactly as written.
- Prefer exact phrases from CHUNK_TEXT for technical terms.
- Do NOT invent causes, steps, thresholds, or conclusions.
- Output must match the requested JSON structure exactly (keys and types).
"""


def build_prompt(doc_type: DocType, view_type: ViewType, ctx: ChunkContext, ner_seed: NERSeed, chunk_text: str) -> str:
    """
    Build an Ollama prompt that produces STRICT JSON for the required view.

    Inputs
    ------
    doc_type: SOP|CR|WO|ECA|OTHER
    view_type: retrieval_summary|rca_frame
    ctx: ChunkContext (doc_id/doc_type/chunk_id/section_path/pages/authority/role)
    ner_seed: NERSeed (JSON-serializable grouped entity hints)
    chunk_text: string (cleaned text; include table text if relevant)

    Output
    ------
    prompt: string for Ollama (/api/chat or /api/generate)
    """
    seed_json = json.dumps(ner_seed.to_json(), ensure_ascii=False)

    # Doc-type + view specific task guidance
    if doc_type == "SOP":
        if view_type == "retrieval_summary":
            task = (
                "TASK: Create a retrieval_summary JSON for an SOP chunk.\n"
                "Emphasize procedure purpose, constraints/holds, and step-related diagnostics/actions.\n"
            )
        else:
            task = (
                "TASK: Create an rca_frame JSON for an SOP chunk.\n"
                "For SOP: observed describes what the SOP step requires; tests_to_confirm are checks/measurements; "
                "candidate_actions are allowed steps mentioned; constraints include holds, cautions, prerequisites.\n"
            )
    elif doc_type == "CR":
        if view_type == "retrieval_summary":
            task = (
                "TASK: Create a retrieval_summary JSON for a Condition Report (CR) chunk.\n"
                "Emphasize what happened (symptoms/outcomes), context, evidence mentioned, and any immediate/corrective actions described.\n"
            )
        else:
            task = (
                "TASK: Create an rca_frame JSON for a CR chunk.\n"
                "For CR: hypotheses ONLY if explicitly stated in CHUNK_TEXT; tests_to_confirm are checks suggested or performed; "
                "candidate_actions are actions taken or planned; constraints are any stated conditions or limitations.\n"
            )
    elif doc_type == "WO":
        if view_type == "retrieval_summary":
            task = (
                "TASK: Create a retrieval_summary JSON for a Work Order (WO) chunk.\n"
                "Emphasize work performed, as-found/as-left conditions, parts/tools, results, and any measurements/acceptance criteria recorded.\n"
            )
        else:
            task = (
                "TASK: Create an rca_frame JSON for a WO chunk.\n"
                "For WO: observed describes the problem/task; hypotheses only if stated; tests_to_confirm includes checks performed; "
                "candidate_actions are the work steps performed; constraints include prerequisites/holds/limitations mentioned.\n"
            )
    elif doc_type == "ECA":
        if view_type == "retrieval_summary":
            task = (
                "TASK: Create a retrieval_summary JSON for an ECA chunk.\n"
                "Emphasize evidence, analysis claims, causal factors (if stated), and recommended corrective actions. Keep phrasing extractive.\n"
            )
        else:
            task = (
                "TASK: Create an rca_frame JSON for an ECA chunk.\n"
                "For ECA: observed summarizes event/evidence; hypotheses can include stated causal hypotheses; "
                "tests_to_confirm are proposed verification steps; candidate_actions are recommended actions; constraints are assumptions/limitations.\n"
            )
    else:
        task = (
            "TASK: Create the requested JSON view for a technical reliability document chunk.\n"
            "Be extractive and conservative.\n"
        )

    # Output skeleton (forces keys)
    if view_type == "retrieval_summary":
        skeleton = json.dumps(empty_retrieval_summary(ctx), ensure_ascii=False, indent=2)
    else:
        skeleton = json.dumps(empty_rca_frame(ctx), ensure_ascii=False, indent=2)

    # Compose final prompt
    prompt = (
        _COMMON_SYSTEM_CONTRACT
        + "\n"
        + task
        + "\nCONTEXT:\n"
        + f"doc_id: {ctx.doc_id}\n"
        + f"doc_type: {ctx.doc_type}\n"
        + f"chunk_id: {ctx.chunk_id}\n"
        + f"section_path: {ctx.section_path}\n"
        + f"page_start: {ctx.page_start}\n"
        + f"page_end: {ctx.page_end}\n"
        + f"authority_level: {ctx.authority_level}\n"
        + f"section_role: {ctx.section_role}\n"
        + "\nNER_SEED (may include items not present; include only if supported by CHUNK_TEXT):\n"
        + seed_json
        + "\n\nCHUNK_TEXT:\n"
        + chunk_text
        + "\n\nOUTPUT JSON (exact keys/types):\n"
        + skeleton
    )
    return prompt


# -----------------------------------------------------------------------------
# Ollama JSON call + robust parsing
# -----------------------------------------------------------------------------

def ollama_generate_json(prompt: str, model: Optional[str] = None, timeout: int = 90) -> Dict[str, Any]:
    """
    Call Ollama and return a parsed JSON dict.

    Inputs
    ------
    prompt: str (must instruct "JSON only")
    model: optional model name; defaults to env OLLAMA_MODEL or "mistral:latest"
    timeout: request timeout seconds

    Output
    ------
    dict parsed from model output

    Behavior
    --------
    - Prefers /api/chat (non-stream for easier JSON parsing)
    - Falls back to /api/generate
    - If output contains extra text, attempts to extract the first JSON object via regex
    """
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = model or os.environ.get("OLLAMA_MODEL", "mistral:latest")
    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

    # 1) /api/chat (non-stream)
    try:
        r = requests.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"num_ctx": num_ctx},
                # Some Ollama builds accept "format": "json". If unsupported, it is ignored.
                "format": "json",
                "stream": False,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        content = ((data.get("message") or {}).get("content") or "").strip()
        return _parse_json_strict(content)
    except Exception:
        pass

    # 2) /api/generate (non-stream)
    r = requests.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {"num_ctx": num_ctx},
            "format": "json",
            "stream": False,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    content = (data.get("response") or "").strip()
    return _parse_json_strict(content)


_JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Parse JSON from the model output.
    If the output includes extra text, extract the first {...} block.

    Input: text (string)
    Output: dict
    Raises: ValueError on failure
    """
    if not text:
        raise ValueError("Empty model output; cannot parse JSON.")

    # Direct parse
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("Model output JSON is not an object.")
        return obj
    except Exception:
        pass

    # Try to extract a JSON object substring
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError(f"Could not locate JSON object in model output: {text[:200]}...")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object.")
    return obj


# -----------------------------------------------------------------------------
# Validation + quality gates (minimal, fast)
# -----------------------------------------------------------------------------

def validate_retrieval_summary_json(obj: Dict[str, Any]) -> List[str]:
    """
    Validate retrieval_summary shape and return flags (empty => pass).

    Input: dict (parsed JSON)
    Output: List[str] flags
    """
    flags: List[str] = []

    if obj.get("view_type") != "retrieval_summary":
        flags.append("bad_view_type")
    for k in ["chunk_id", "doc_type", "scope", "entities", "citations", "unknowns"]:
        if k not in obj:
            flags.append(f"missing_{k}")

    ent = obj.get("entities") or {}
    if not isinstance(ent, dict):
        flags.append("entities_not_object")
    else:
        for ek in ["systems", "equipment_ids", "components"]:
            if ek not in ent or not isinstance(ent.get(ek), list):
                flags.append(f"entities_{ek}_missing_or_not_list")

    # Basic quality checks
    scope = (obj.get("scope") or "").strip()
    if len(scope) < 15:
        flags.append("scope_too_short")
    kws = obj.get("keywords_synonyms") or []
    if isinstance(kws, list) and len(kws) < 3:
        flags.append("few_keywords")

    # Boilerplate guard
    bad = ["as an ai", "i cannot", "not provided"]
    low_scope = scope.lower()
    if any(b in low_scope for b in bad):
        flags.append("boilerplate")

    return flags


def validate_rca_frame_json(obj: Dict[str, Any]) -> List[str]:
    """
    Validate rca_frame shape and return flags (empty => pass).

    Input: dict (parsed JSON)
    Output: List[str] flags
    """
    flags: List[str] = []
    if obj.get("view_type") != "rca_frame":
        flags.append("bad_view_type")
    for k in ["chunk_id", "doc_type", "observed", "citations"]:
        if k not in obj:
            flags.append(f"missing_{k}")
    for lk in ["observed", "hypotheses", "tests_to_confirm", "candidate_actions", "constraints"]:
        if lk in obj and not isinstance(obj.get(lk), list):
            flags.append(f"{lk}_not_list")
    return flags


def summarize_with_retry(
    doc_type: DocType,
    view_type: ViewType,
    ctx: ChunkContext,
    ner_seed: NERSeed,
    chunk_text: str,
    model: Optional[str] = None,
    timeout: int = 90,
    max_tries: int = 3,
    sleep_sec: float = 0.25,
) -> Dict[str, Any]:
    """
    Robust summarize call with retries.

    Inputs
    ------
    doc_type/view_type/ctx/ner_seed/chunk_text: see build_prompt()
    model: optional Ollama model name
    max_tries: integer retry count

    Output
    ------
    Parsed JSON dict (retrieval_summary or rca_frame)

    Notes
    -----
    Retry ladder:
    1) normal prompt
    2) add "Your previous response was invalid JSON. Return valid JSON only."
    3) truncate chunk text to reduce failure risk
    """
    prompt = build_prompt(doc_type, view_type, ctx, ner_seed, chunk_text)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_tries + 1):
        try:
            time.sleep(sleep_sec)
            obj = ollama_generate_json(prompt, model=model, timeout=timeout)
            flags = validate_retrieval_summary_json(obj) if view_type == "retrieval_summary" else validate_rca_frame_json(obj)
            if not flags:
                return obj

            # If structure is wrong, treat as failure and retry
            raise ValueError(f"Validation flags: {flags}")
        except Exception as e:
            last_err = e
            if attempt == 1:
                prompt = "Your previous response was invalid JSON. Return valid JSON only.\n\n" + prompt
            elif attempt == 2:
                # truncate
                chunk_text2 = chunk_text[:4000]
                prompt = build_prompt(doc_type, view_type, ctx, ner_seed, chunk_text2)
            else:
                break

    raise RuntimeError(f"Ollama summarization failed after {max_tries} tries: {last_err}")


# -----------------------------------------------------------------------------
# Helpers: flatten JSON to embedding-friendly text
# -----------------------------------------------------------------------------

def flatten_retrieval_summary_for_embedding(summary: Dict[str, Any]) -> str:
    """
    Convert retrieval_summary JSON into a dense string for embedding.

    Input: retrieval_summary dict
    Output: string
    """
    ent = summary.get("entities") or {}
    lines = [
        f"SCOPE: {summary.get('scope','')}",
        "SYSTEMS: " + ", ".join(ent.get("systems") or []),
        "EQUIPMENT: " + ", ".join(ent.get("equipment_ids") or []),
        "COMPONENTS: " + ", ".join(ent.get("components") or []),
        "SYMPTOMS/OUTCOMES: " + ", ".join(summary.get("symptoms_outcomes") or []),
        "MECHANISMS: " + ", ".join(summary.get("mechanisms") or []),
        "DIAGNOSTICS: " + ", ".join(summary.get("diagnostics") or []),
        "ACTIONS: " + ", ".join(summary.get("corrective_actions") or []),
        "NUMBERS/LIMITS: " + ", ".join(summary.get("numbers_limits") or []),
        "KEYWORDS: " + ", ".join(summary.get("keywords_synonyms") or []),
    ]
    return "\n".join([ln for ln in lines if ln.strip()])


def flatten_rca_frame_for_embedding(rca: Dict[str, Any]) -> str:
    """
    Convert rca_frame JSON into an embedding-friendly string.

    Input: rca_frame dict
    Output: string
    """
    lines = [
        "OBSERVED: " + "; ".join(rca.get("observed") or []),
        "HYPOTHESES: " + "; ".join(rca.get("hypotheses") or []),
        "TESTS: " + "; ".join(rca.get("tests_to_confirm") or []),
        "ACTIONS: " + "; ".join(rca.get("candidate_actions") or []),
        "CONSTRAINTS: " + "; ".join(rca.get("constraints") or []),
    ]
    return "\n".join([ln for ln in lines if ln.strip()])
