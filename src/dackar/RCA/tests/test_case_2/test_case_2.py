# %%
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# %%
NOTEBOOK_ROOT = Path.cwd().resolve()

FIXTURE_DIR = NOTEBOOK_ROOT / "fixtures" 
OUTPUT_DIR = NOTEBOOK_ROOT / "rca_runs_case_002"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Python path setup
# ---------------------------------------------------------------------
# Update these if your notebook is in a different location.
rca_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if rca_root not in sys.path:
    sys.path.insert(0, rca_root)

dackar_root = os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))
if dackar_root not in sys.path:
    sys.path.insert(0, dackar_root)

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
from orchestrators.rca_reasoning_orchestrator import build_dev_orchestrator
from kg.py2neo_workflow import Py2Neo
from storage.chroma_store import ChromaRecordStore
from storage.processed_record_store import ProcessedRecordStore
from storage.lc_retriever_processed import LCProcessedRetriever
from storage.processed_evidence_store_adapter import ProcessedEvidenceStoreAdapter

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", None)

VALIDATOR_MODE = "compat"
STOP_ON_VALIDATION_ERROR = False

# Robust schema path from installed/imported package location
import orchestrators.rca_reasoning_orchestrator as orch_mod
SCHEMA_DIR = Path(orch_mod.__file__).resolve().parents[1] / "schemas"

print("Fixture dir:", FIXTURE_DIR)
print("Schema dir :", SCHEMA_DIR)
print("Schemas    :", sorted(p.name for p in SCHEMA_DIR.glob("*.json")))

# %% [markdown]
# ## Utilities

# %%
def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def maybe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    return load_json(path) if path.exists() else None

def safe_get(d: Optional[Dict[str, Any]], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required fixture file: {path}")

def print_block(title: str, obj: Any, max_chars: int = 5000) -> None:
    print(f"\n--- {title} ---")
    text = json.dumps(obj, indent=2, default=str)
    print(text[:max_chars])

# %% [markdown]
# ## Load fixture bundle

# %%
required_files = [
    "event.json",
    "telemetry_summary.json",
    "kg_context.json",
]

for name in required_files:
    require_file(FIXTURE_DIR / name)

event = load_json(FIXTURE_DIR / "event.json")
telemetry_summary = load_json(FIXTURE_DIR / "telemetry_summary.json")
kg_context = load_json(FIXTURE_DIR / "kg_context.json")

# Optional prebuilt artifacts
tskr_patterns = maybe_load_json(FIXTURE_DIR / "tskr_patterns.json")
operational_context = maybe_load_json(FIXTURE_DIR / "operational_context.json")
pm_compliance = maybe_load_json(FIXTURE_DIR / "pm_compliance.json")

# %% [markdown]
# ## Checks

# %%
assert event["event_id"] == telemetry_summary["event_id"], "event_id mismatch"
assert event["asset_id"] == telemetry_summary["asset_id"], "asset_id mismatch"
assert kg_context["event_id"] == event["event_id"], "kg_context.event_id mismatch"
assert kg_context["asset_id"] == event["asset_id"], "kg_context.asset_id mismatch"

if tskr_patterns is not None:
    assert tskr_patterns["event_id"] == event["event_id"], "tskr_patterns.event_id mismatch"
    assert tskr_patterns["asset_id"] == event["asset_id"], "tskr_patterns.asset_id mismatch"

print("Fixture sanity checks passed.")


# %%
print("Schema dir :", SCHEMA_DIR)
print("Schemas    :", sorted(p.name for p in SCHEMA_DIR.glob("*.json")))
assert (SCHEMA_DIR / "causality_candidates.json").exists(), "Missing causality_candidates.json in runtime schema dir"

# %%
# Chroma adapter for v32
# ## Build evidence store

# %%
PROCESSED_JSONL = FIXTURE_DIR / "processed_records.jsonl"  # adjust if needed
CHROMA_DIR = NOTEBOOK_ROOT / "chroma_case_002"

record_store = ProcessedRecordStore([str(PROCESSED_JSONL)])
chroma_manager = ChromaRecordStore(
    persist_directory=str(CHROMA_DIR),
)

# Ingest once for the local test corpus
chroma_manager.upsert_jsonl(str(PROCESSED_JSONL))

lc_retriever = LCProcessedRetriever(
    manager=chroma_manager,
    doc_store=record_store,
)

evidence_store = ProcessedEvidenceStoreAdapter(
    retriever=lc_retriever,
    k_final=10,
)

print("evidence_store:", evidence_store)
print("record count:", len(record_store))

# %% [markdown]
# ## Build orchestrator

# %%
client = Py2Neo(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

orchestrator_v31 = build_dev_orchestrator(
    output_dir=OUTPUT_DIR / "v31",
    client=client,
    database=NEO4J_DATABASE,
    evidence_store=evidence_store,
    schema_dir=SCHEMA_DIR,
    validator_mode=VALIDATOR_MODE,
    stop_on_validation_error=STOP_ON_VALIDATION_ERROR,
    causality_engine_version="v31",
)

orchestrator_v32 = build_dev_orchestrator(
    output_dir=OUTPUT_DIR / "v32",
    client=client,
    database=NEO4J_DATABASE,
    evidence_store=evidence_store,
    schema_dir=SCHEMA_DIR,
    validator_mode=VALIDATOR_MODE,
    stop_on_validation_error=STOP_ON_VALIDATION_ERROR,
    causality_engine_version="v32",
)

print("Orchestrators built.")
print("Validator schemas:", sorted(getattr(orchestrator_v31.validator, "schemas", {}).keys()))

# %%
print("\n=== DEBUG: Evidence store wiring ===")
print("orchestrator_v32.evidence_retriever.store:", orchestrator_v32.evidence_retriever.store)
print("type:", type(orchestrator_v32.evidence_retriever.store))

# Optional deeper inspection
store = orchestrator_v32.evidence_retriever.store
if hasattr(store, "__dict__"):
    print("store attributes:", list(store.__dict__.keys()))

# %% [markdown]
# ## Run orchestrator

# %%
try:
    result_v31 = orchestrator_v31.run(
        event=event,
        telemetry_summary=telemetry_summary,
        operational_context=operational_context,
        pm_compliance=pm_compliance,
        kg_context=kg_context,
        tskr_patterns=tskr_patterns,
    )
    result_v32 = orchestrator_v32.run(
        event=event,
        telemetry_summary=telemetry_summary,
        operational_context=operational_context,
        pm_compliance=pm_compliance,
        kg_context=kg_context,
        tskr_patterns=tskr_patterns,
    )

finally:
    client.close()

print("Runs completed.")

# %% [markdown]
# ## Inspect outputs

# %%
def build_comparison_summary(label: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": label,
        "run_id": safe_get(result, "run_context", "run_id"),
        "engine_version": safe_get(result, "run_manifest", "pipeline_config", "causality_engine_version"),
        "decision_status": safe_get(result, "rca_card", "executive_summary", "decision_status"),
        "primary_candidate_id": safe_get(result, "rca_card", "primary_hypothesis", "candidate_id"),
        "n_candidates": len(safe_get(result, "causality_candidates", "candidates", default=[]) or []),
        "n_evidence": len(safe_get(result, "evidence_bundle", "results", default=[]) or []),
        "supporting_count": safe_get(result, "run_manifest", "artifacts", "rca_card", "primary_supporting_evidence_count"),
        "contradicting_count": safe_get(result, "run_manifest", "artifacts", "rca_card", "primary_contradicting_evidence_count"),
        "fallback_used": safe_get(result, "rca_card", "validation_status", "fallback_used"),
        "writeback_ready": safe_get(result, "run_manifest", "review_hooks", "writeback_ready"),
        "generated_candidate_count": safe_get(result, "causality_candidates", "summary", "generated_candidate_count"),
        "retained_candidate_count": safe_get(result, "causality_candidates", "summary", "retained_candidate_count"),
        "filtered_out_candidate_count": safe_get(result, "causality_candidates", "summary", "filtered_out_candidate_count"),
    }

summary_v31 = build_comparison_summary("v31", result_v31)
summary_v32 = build_comparison_summary("v32", result_v32)

print_block("summary_v31", summary_v31, max_chars=3000)
print_block("summary_v32", summary_v32, max_chars=3000)

# %% [markdown]
# ## Candidate ranking

# %%
for label, result in [("v31", result_v31), ("v32", result_v32)]:
    cands = safe_get(result, "causality_candidates", "candidates", default=[]) or []
    cands_sorted = sorted(
        cands,
        key=lambda c: float(c.get("composite_score", 0.0)),
        reverse=True,
    )
    print(f"\n=== Candidate ranking: {label} ===")
    for i, c in enumerate(cands_sorted, start=1):
        print(
            f"{i}. {c.get('candidate_id')} | "
            f"{c.get('cause_label')} | "
            f"score={c.get('composite_score')} | "
            f"temporal={safe_get(c, 'temporal_evidence', 'relation')} | "
            f"signals={safe_get(c, 'telemetry_evidence', 'matching_signal_ids', default=[])}"
        )

# Ishikawa summary
cats = safe_get(result, "ishikawa_matrix", "categories", default=[]) or []
print("\n=== Ishikawa categories ===")
for cat in cats:
    category = cat.get("category")
    rows = cat.get("rows", []) or []
    print(f"{category}: {len(rows)} rows")
    for row in rows[:3]:
        print(
            "   -",
            row.get("label"),
            "| strength=",
            row.get("strength"),
            "| source=",
            row.get("source_artifact"),
        )

# %% [markdown]
# ## Full result bundle

# %%
for label, result in [("v31", result_v31), ("v32", result_v32)]:
    out_path = OUTPUT_DIR / f"{label}_full_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved {label} full result to:", out_path)


