# %% [markdown]
# # Stage 1–6 demo using your developed methods
# 
# This notebook uses your existing modules where they are runnable: `pdfParser.py`, `mdParser.py`, `reliability_summarizer.py`, `equipment_ID_extractor.py`, and `chroma_store.py`. It prefers your implementations first and only falls back when a runtime dependency like Marker or Ollama is unavailable.
# 

# %%
from pathlib import Path
import pandas as pd
from stage1_6_existing_methods_helpers import run_many, maybe_upsert_with_chroma

PDFS = ['../examples/example_CR_2026_00123.pdf']
        #'../examples/example_ECA_2026_0007.pdf',
        #'../examples/example_SOP_AFW_P101A.pdf',
        #'../examples/example_WO_2026_04567.pdf']

OUTPUT_ROOT = './1-6pipeline_demo_existing_methods'


# %%
results = run_many(PDFS, output_root=OUTPUT_ROOT, model=None)
len(results)

# %%
rows = []
for r in results:
    rows.append({
        'pdf': Path(r.pdf_path).name,
        'doc_type': r.doc_type,
        'enriched_jsonl_path': r.enriched_jsonl_path,
        'processed_record_count': len(r.processed_records),
    })
pd.DataFrame(rows)


# %%
# Inspect one processed_text_record
sample = results[0].processed_records[0]
sample.keys(), sample['metadata'].keys()


# %%
sample


# %%
# Optional Chroma upsert if dependencies are installed and Ollama embeddings are available
enriched_paths = [r.enriched_jsonl_path for r in results]
maybe_upsert_with_chroma(enriched_paths, persist_directory = "./1-6pipeline_demo_existing_methods/chroma_store")



