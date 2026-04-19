# %% [markdown]
# # KG population demo from processed reliability documents
# 
# This notebook picks up **after** your Stage 1–6 parsing workflow and uses the workflow-oriented Neo4j classes to:
# 
# 1. load processed document outputs,
# 2. derive lightweight `document` records from the processed text records,
# 3. build a graph batch with `build_graph_from_workflow_artifacts`,
# 4. preview the nodes and edges,
# 5. optionally ingest the graph into Neo4j.
# 
# It is designed to be a companion to `stage1_6_existing_methods_demo.ipynb`.
# 

# %%
from pathlib import Path
print(Path().resolve())
print(list(Path().resolve().parents))

# %%
from __future__ import annotations

import sys
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

rca_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(rca_root)

dackar_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(dackar_root)

from kg.kg_schema_builder_workflow import (
    apply_schema_constraints,
    build_graph_from_workflow_artifacts,
    load_and_merge_schemas,
)
from kg.py2neo_workflow import Py2Neo
from kg_population_helpers import load_processed_records_from_output


# %%
# ----- configuration -----
# Point this to the output directory created by the Stage 1-6 document parsing notebook.
OUTPUT_ROOT = Path("./1-6pipeline_demo_existing_methods")

# The stage notebook stores enriched JSONL files under the output root. If your layout differs,
# update ENRICHED_GLOBS below.
ENRICHED_GLOBS = [
    "**/*enriched*.jsonl",
    "**/*processed*.jsonl",
]

SCHEMA_PATHS = [
    "../../knowledge_graph/schemas/mbseSchema.toml",
    "../../knowledge_graph/schemas/documentSchema.toml",
    "../../knowledge_graph/schemas/conditionReportSchema.toml",
    "../../knowledge_graph/schemas/workOrderSchema.toml",
    "../../knowledge_graph/schemas/causalSchema.toml",
    "../../knowledge_graph/schemas/fmeaSchema.toml",
]

# Neo4j connection: leave INGEST_TO_NEO4J = False for a dry run.
INGEST_TO_NEO4J = False
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "123456789")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or None
RESET_DATABASE_FIRST = False
CREATE_CONSTRAINTS = True


# %%
documents, processed_text_records, enriched_files = load_processed_records_from_output(OUTPUT_ROOT, ENRICHED_GLOBS)

print(f"Found {len(enriched_files)} enriched JSONL files")
print(f"Derived {len(documents)} document records")
print(f"Loaded {len(processed_text_records)} processed text records")

display(pd.DataFrame({"enriched_jsonl": [str(p) for p in enriched_files]}))


# %%
# Preview the derived document objects that will be passed into the workflow KG builder.
pd.DataFrame(documents).fillna("")


# %%
# Preview a sample processed text record.
sample_record = processed_text_records[0]
sample_record


# %%
schema = load_and_merge_schemas(SCHEMA_PATHS)

nodes, edges = build_graph_from_workflow_artifacts(
    SCHEMA_PATHS,
    documents=documents,
    processed_text_records=processed_text_records,
)

print(f"Graph batch built with {len(nodes)} nodes and {len(edges)} edges")


# %%
nodes_df = pd.DataFrame([
    {
        "id": n["id"],
        "label": n["label"],
        **{k: v for k, v in n["attrs"].items() if k in {"id", "doc_key", "doc_type", "record_key", "source_key", "kind", "label"}},
    }
    for n in nodes
])

nodes_df = pd.DataFrame(nodes) if nodes else pd.DataFrame(columns=["id", "label", "properties"])

normalized_edges = []
for e in edges or []:
    normalized_edges.append({
        "source": e.get("source") or e.get("from"),
        "target": e.get("target") or e.get("to"),
        "type": e.get("type") or e.get("relation") or e.get("edge_type"),
        "properties": e.get("properties", {}),
    })

edges_df = pd.DataFrame(normalized_edges, columns=["source", "target", "type", "properties"])

print("Node labels")
display(nodes_df["label"].value_counts().rename_axis("label").reset_index(name="count"))

print("Edge types")
display(edges_df["type"].value_counts().rename_axis("type").reset_index(name="count"))


# %%
display(nodes_df.head(25))
display(edges_df.head(25))


# %%
# Optional: export the graph batch so it can be shown without Neo4j.
EXPORT_DIR = OUTPUT_ROOT / "kg_population_demo"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

with (EXPORT_DIR / "nodes.json").open("w", encoding="utf-8") as f:
    json.dump(nodes, f, indent=2, ensure_ascii=False)
with (EXPORT_DIR / "edges.json").open("w", encoding="utf-8") as f:
    json.dump(edges, f, indent=2, ensure_ascii=False)

print("Exported graph batch to", EXPORT_DIR)


# %%
# Optional Neo4j ingestion.
# Set INGEST_TO_NEO4J = True in the configuration cell to enable this block.
if INGEST_TO_NEO4J:
    client = Py2Neo(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        if RESET_DATABASE_FIRST:
            client.reset(db=NEO4J_DATABASE)
        if CREATE_CONSTRAINTS:
            apply_schema_constraints(client, SCHEMA_PATHS, database=NEO4J_DATABASE)
        client.upsert_nodes_batch(nodes, db=NEO4J_DATABASE)
        client.upsert_edges_batch(edges, db=NEO4J_DATABASE)
        print(f"Ingested {len(nodes)} nodes and {len(edges)} edges into Neo4j")
    finally:
        client.close()
else:
    print("Dry run only. Set INGEST_TO_NEO4J = True to write to Neo4j.")


# %%
# Optional verification query after ingestion.
if INGEST_TO_NEO4J:
    client = Py2Neo(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        counts = client.query(
            "MATCH (n) RETURN labels(n) AS labels, count(*) AS count ORDER BY count DESC",
            db=NEO4J_DATABASE,
        )
        display(pd.DataFrame([dict(r) for r in counts]))
    finally:
        client.close()



