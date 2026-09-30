"""
Embedding model benchmark for nuclear outage activity similarity.

Evaluates three Ollama embedding models on two tasks:

Task 1 — Corruption robustness (outage_cleaning_benchmark.csv)
    Each row has a clean_description and a contaminated_description for the
    same activity (with abbreviations, typos, style artefacts).
    A good embedder should give clean↔contaminated pairs HIGH similarity
    regardless of the text corruption.

    Metric: mean cosine similarity of same-activity pairs vs.
            mean cosine similarity of cross-category pairs
            → Discrimination Ratio = mean_within / mean_cross

Task 2 — Category retrieval (outage_cleaning_benchmark_severity.csv)
    Given each contaminated description as a query, retrieve the top-k
    most similar activities from the same CSV (leave-one-out) and check
    how many of the top-k share the same category.

    Metric: Precision@5 and Precision@10 (fraction of retrieved activities
            in the same category as the query)

Usage:
    python embedding_benchmark.py                   # all three models
    python embedding_benchmark.py --model nomic     # specific model
    python embedding_benchmark.py --no-llm          # skip LLM disambiguation test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://localhost:11434"

EMBED_MODELS = {
    "nomic":  "nomic-embed-text:latest",
    "mxbai":  "mxbai-embed-large:latest",
    "bge":    "bge-m3:567m",
}

LLM_MODEL = "llama3.2:latest"


def _post(endpoint: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def embed(model: str, text: str) -> list[float]:
    result = _post("/api/embeddings", {"model": model, "prompt": text})
    return result["embedding"]


def embed_batch(model: str, texts: list[str], label: str = "") -> list[list[float]]:
    embeddings = []
    for i, text in enumerate(texts):
        if label and (i % 20 == 0):
            print(f"  [{label}] {i}/{len(texts)} ...", end="\r", flush=True)
        embeddings.append(embed(model, text))
    if label:
        print(f"  [{label}] {len(texts)}/{len(texts)} done          ")
    return embeddings


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_BENCHMARK_DATA_DIR = Path(__file__).parent


def load_benchmark1() -> list[dict]:
    rows = []
    with open(_BENCHMARK_DATA_DIR / "outage_cleaning_benchmark.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_benchmark2() -> list[dict]:
    rows = []
    with open(_BENCHMARK_DATA_DIR / "outage_cleaning_benchmark_severity.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Task 1 — Corruption robustness
# ---------------------------------------------------------------------------

def task1_corruption_robustness(model_key: str, model_name: str, rows: list[dict]) -> dict:
    print(f"\n[Task 1] Corruption robustness — {model_key}")

    clean_texts       = [r["clean_description"] for r in rows]
    contaminated_texts = [r["contaminated_description"] for r in rows]
    categories        = [r["category"] for r in rows]

    t0 = time.time()
    clean_emb = embed_batch(model_name, clean_texts, f"{model_key}/clean")
    cont_emb  = embed_batch(model_name, contaminated_texts, f"{model_key}/contam")
    elapsed   = time.time() - t0

    n = len(rows)

    # Within-activity: clean_i vs contaminated_i (same activity, corrupted text)
    within = [cosine(clean_emb[i], cont_emb[i]) for i in range(n)]

    # Cross-category: random sample of (i, j) pairs where category_i != category_j
    cross = []
    for i in range(n):
        for j in range(i + 1, min(i + 20, n)):
            if categories[i] != categories[j]:
                cross.append(cosine(clean_emb[i], clean_emb[j]))

    mean_within = sum(within) / len(within)
    mean_cross  = sum(cross) / len(cross) if cross else 0.0
    disc_ratio  = mean_within / mean_cross if mean_cross > 0 else float("inf")

    # Per-category breakdown
    from collections import defaultdict
    cat_scores: dict[str, list[float]] = defaultdict(list)
    for i, row in enumerate(rows):
        cat_scores[row["category"]].append(within[i])

    print(f"  Embed time   : {elapsed:.1f}s ({2*n} calls)")
    print(f"  Mean within  : {mean_within:.4f}  (clean vs corrupted, same activity)")
    print(f"  Mean cross   : {mean_cross:.4f}  (different categories)")
    print(f"  Disc. ratio  : {disc_ratio:.3f}  (higher is better)")
    print(f"  Min within   : {min(within):.4f}  (worst-case corruption)")
    print(f"\n  Per-category mean similarity:")
    for cat, scores in sorted(cat_scores.items()):
        bar = "█" * int(sum(scores) / len(scores) * 40)
        print(f"    {cat:<30s} {sum(scores)/len(scores):.4f}  {bar}")

    return {
        "model": model_key,
        "mean_within": mean_within,
        "mean_cross": mean_cross,
        "disc_ratio": disc_ratio,
        "min_within": min(within),
        "embed_time_s": elapsed,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Task 2 — Category retrieval (Precision@k)
# ---------------------------------------------------------------------------

def task2_category_retrieval(model_key: str, model_name: str, rows: list[dict]) -> dict:
    print(f"\n[Task 2] Category retrieval — {model_key}")

    texts      = [r["contaminated_description"] for r in rows]
    categories = [r["category"] for r in rows]

    t0   = time.time()
    embs = embed_batch(model_name, texts, f"{model_key}/retrieval")
    elapsed = time.time() - t0

    n = len(rows)

    p_at_5  = []
    p_at_10 = []

    for i in range(n):
        # Score all other activities
        scored = []
        for j in range(n):
            if i == j:
                continue
            scored.append((cosine(embs[i], embs[j]), j))
        scored.sort(reverse=True)

        cat_i = categories[i]
        for k, label in [(5, p_at_5), (10, p_at_10)]:
            top_k = scored[:k]
            hits  = sum(1 for _, j in top_k if categories[j] == cat_i)
            label.append(hits / k)

    mean_p5  = sum(p_at_5)  / len(p_at_5)
    mean_p10 = sum(p_at_10) / len(p_at_10)

    print(f"  Embed time   : {elapsed:.1f}s ({n} calls)")
    print(f"  Precision@5  : {mean_p5:.4f}")
    print(f"  Precision@10 : {mean_p10:.4f}")

    return {
        "model": model_key,
        "precision_at_5":  mean_p5,
        "precision_at_10": mean_p10,
        "embed_time_s": elapsed,
        "n": n,
    }


# ---------------------------------------------------------------------------
# LLM disambiguation probe
# ---------------------------------------------------------------------------

DISAMBIGUATION_CASES = [
    {
        "description": "INSP reactor clnt pump seal in package the aux building",
        "token": "clnt",
        "candidates": ["coolant", "client", "clean"],
        "expected": "coolant",
    },
    {
        "description": "RPL PACKIIG on the MOV PT-455A in the SI sys",
        "token": "MOV",
        "candidates": ["motor operated valve", "move", "monitoring"],
        "expected": "motor operated valve",
    },
    {
        "description": "cal pressurizer pressure XMTR in containment during mid inspection outage window",
        "token": "XMTR",
        "candidates": ["transmitter", "transformer", "transfer"],
        "expected": "transmitter",
    },
    {
        "description": "repl mn fdtr vlv regulating actuator near the pressurizer after scaffold installation",
        "token": "fdtr",
        "candidates": ["feedwater", "filter", "finder"],
        "expected": "feedwater",
    },
    {
        "description": "Perform PM on RCP 1A per WCO-12345",
        "token": "PM",
        "candidates": ["preventive maintenance", "project manager", "prime mover"],
        "expected": "preventive maintenance",
    },
    {
        "description": "EDGR fuel oil transfer pump test per TS surveillance",
        "token": "EDGR",
        "candidates": ["emergency diesel generator", "edge router", "electrode grounding"],
        "expected": "emergency diesel generator",
    },
    {
        "description": "RHR HX tube side flushing after refuel outage",
        "token": "RHR",
        "candidates": ["residual heat removal", "right hand rotation", "reactor hazard rating"],
        "expected": "residual heat removal",
    },
]

PROMPT_TEMPLATE = """\
You are a nuclear power plant maintenance expert.
Given the following activity description and an ambiguous token, select the most appropriate meaning from the candidates.
Reply with ONLY the chosen candidate text, nothing else.

Activity: {description}
Token: {token}
Candidates: {candidates}
"""


def task3_llm_disambiguation() -> dict:
    print(f"\n[Task 3] LLM abbreviation disambiguation — {LLM_MODEL}")

    correct = 0
    results = []

    for case in DISAMBIGUATION_CASES:
        prompt = PROMPT_TEMPLATE.format(
            description=case["description"],
            token=case["token"],
            candidates=", ".join(f'"{c}"' for c in case["candidates"]),
        )
        t0 = time.time()
        resp = _post("/api/generate", {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        })
        elapsed = time.time() - t0
        answer = resp.get("response", "").strip().strip('"').lower()
        expected = case["expected"].lower()
        hit = expected in answer or answer in expected
        correct += int(hit)
        status = "✓" if hit else "✗"
        results.append({
            "token": case["token"],
            "expected": case["expected"],
            "got": resp.get("response", "").strip(),
            "correct": hit,
        })
        print(f"  {status} {case['token']:<8s} → expected '{case['expected']}' | got '{resp.get('response','').strip()}' ({elapsed:.1f}s)")

    accuracy = correct / len(DISAMBIGUATION_CASES)
    print(f"\n  Accuracy: {correct}/{len(DISAMBIGUATION_CASES)} = {accuracy:.0%}")

    return {"accuracy": accuracy, "n": len(DISAMBIGUATION_CASES), "results": results}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(t1_results: list[dict], t2_results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Model':<8s}  {'Disc.Ratio':>10s}  {'Min Within':>10s}  {'P@5':>6s}  {'P@10':>6s}  {'Embed(s)':>8s}")
    print("-" * 72)

    t1_by_model = {r["model"]: r for r in t1_results}
    t2_by_model = {r["model"]: r for r in t2_results}

    for key in EMBED_MODELS:
        if key not in t1_by_model:
            continue
        t1 = t1_by_model[key]
        t2 = t2_by_model.get(key, {})
        total_time = t1.get("embed_time_s", 0) + t2.get("embed_time_s", 0)
        print(
            f"{key:<8s}  {t1['disc_ratio']:>10.3f}  {t1['min_within']:>10.4f}"
            f"  {t2.get('precision_at_5', 0):>6.4f}  {t2.get('precision_at_10', 0):>6.4f}"
            f"  {total_time:>8.1f}"
        )

    print("=" * 72)
    print("Disc.Ratio = mean(clean↔corrupted same activity) / mean(cross-category)")
    print("P@k        = fraction of top-k retrieved activities in same category")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding benchmark for outage activity similarity")
    parser.add_argument("--model", choices=list(EMBED_MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM disambiguation test")
    parser.add_argument("--task", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    models_to_run = EMBED_MODELS if args.model == "all" else {args.model: EMBED_MODELS[args.model]}

    rows1 = load_benchmark1()
    rows2 = load_benchmark2()
    print(f"Loaded {len(rows1)} rows (benchmark1) and {len(rows2)} rows (benchmark2)")

    t1_results = []
    t2_results = []

    for key, name in models_to_run.items():
        if args.task in ("1", "all"):
            t1_results.append(task1_corruption_robustness(key, name, rows1))
        if args.task in ("2", "all"):
            t2_results.append(task2_category_retrieval(key, name, rows2))

    if args.task in ("1", "2", "all") and len(models_to_run) > 1:
        print_summary(t1_results, t2_results)

    if args.task in ("3", "all") and not args.no_llm:
        task3_llm_disambiguation()


if __name__ == "__main__":
    main()
