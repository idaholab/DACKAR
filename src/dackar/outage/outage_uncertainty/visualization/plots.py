"""
outage_uncertainty.visualization.plots — Reusable plot functions for outage uncertainty analysis.

All functions follow a consistent signature:
    plot_*(data, ..., *, title=None, ax=None or figsize=None) -> (fig, axes)

They can be used standalone (fig created internally) or composed into a
larger figure by passing an existing axes object.

Functions
---------
plot_finish_distribution        Histogram of MC project durations + percentile lines
plot_activity_risk_ranking      3-panel bar chart: criticality / drag / CP sensitivity
plot_cp_path_heatmap            Frequency heatmap of top-N critical-path routes
plot_duration_distributions     Grid of per-activity duration histograms
plot_analog_scatter             Planned vs actual duration scatter coloured by overrun
plot_duration_summary_bars      Horizontal bars: planned vs p50 ± p80 for all activities
plot_preprocessing_benchmark    NED and exact-match bar chart from the cleaning benchmark
plot_routine_vs_disruption      Two-panel: routine vs disruption histograms + CDF comparison
"""
from __future__ import annotations

import random as _random
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BLUE   = "#4C72B0"
_GREEN  = "#27ae60"
_ORANGE = "#e67e22"
_RED    = "#e74c3c"
_DARK   = "#2c3e50"
_GREY   = "#95a5a6"

_TIER_COLORS = {
    "high":           _GREEN,
    "data_supported": _GREEN,
    "medium":         _ORANGE,
    "sme_informed":   _ORANGE,
    "low":            _RED,
    "low_confidence": _RED,
}


def _fig_ax(ax, figsize):
    """Return (fig, ax): create new figure when ax is None."""
    if ax is not None:
        return ax.get_figure(), ax
    fig, ax = plt.subplots(figsize=figsize or (10, 4.5))
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Finish-time distribution
# ---------------------------------------------------------------------------

def plot_finish_distribution(
    cp_times: list[float],
    risk: dict,
    baseline_cp: float,
    *,
    title: str | None = None,
    ax=None,
) -> tuple:
    """Histogram of Monte Carlo project durations with baseline / percentile lines.

    Parameters
    ----------
    cp_times    : raw list of sampled project durations (hours)
    risk        : dict from CriticalPathRiskAnalyzer / ScenarioRunner with keys
                  p50_finish, p80_finish, p90_finish, robustness
    baseline_cp : deterministic baseline critical-path duration (hours)
    """
    fig, ax = _fig_ax(ax, (11, 4.5))

    n = len(cp_times)
    overrun_prob = 1.0 - risk.get("robustness", 0.0)

    ax.hist(cp_times, bins=60, color=_BLUE, alpha=0.72,
            edgecolor="white", linewidth=0.4, density=True, label="Simulated finish")

    overrun = [v for v in cp_times if v > baseline_cp]
    if overrun:
        ax.hist(overrun, bins=60, color=_RED, alpha=0.22,
                edgecolor="none", density=True, label="_overrun fill")

    ax.axvline(baseline_cp,         color=_DARK,   lw=2.2, ls="-",
               label=f"Baseline  {baseline_cp:.0f} h")
    ax.axvline(risk["p50_finish"],  color=_GREEN,  lw=2,   ls="--",
               label=f"P50  {risk['p50_finish']:.1f} h")
    ax.axvline(risk["p80_finish"],  color=_ORANGE, lw=1.8, ls="--",
               label=f"P80  {risk['p80_finish']:.1f} h")
    ax.axvline(risk["p90_finish"],  color=_RED,    lw=1.8, ls=":",
               label=f"P90  {risk['p90_finish']:.1f} h")

    ax.set_xlabel("Outage duration (h)", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    default_title = (
        f"Outage finish-time distribution  —  {n:,} MC iterations\n"
        f"Schedule overrun risk: {overrun_prob*100:.1f}%  "
        f"(prob > {baseline_cp:.0f} h baseline)"
    )
    ax.set_title(title or default_title, fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Activity risk ranking (3-panel)
# ---------------------------------------------------------------------------

def plot_activity_risk_ranking(
    risk: dict,
    activity_ids: list[str],
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """Three-panel bar chart: criticality index / expected drag / CP sensitivity.

    Parameters
    ----------
    risk         : dict from CriticalPathRiskAnalyzer with keys
                   criticality_index, expected_drag, cp_sensitivity
    activity_ids : ordered list of activity IDs to include
    """
    ci  = risk.get("criticality_index", {})
    drg = risk.get("expected_drag", {})
    sen = risk.get("cp_sensitivity", {})

    # Sort by CP sensitivity descending
    ranked = sorted(activity_ids, key=lambda k: sen.get(k, 0), reverse=True)
    palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(ranked)))

    fig, axes = plt.subplots(1, 3, figsize=figsize or (14, 4))

    def _barh(ax, vals, xlabel, subtitle):
        ax.barh(ranked, vals, color=palette)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v * 1.02 + 1e-9, i, f"{v:.1f}", va="center", fontsize=8)
        ax.yaxis.grid(False)

    crit_vals = [ci.get(k, 0) * 100 for k in ranked]
    drag_vals = [drg.get(k, 0) for k in ranked]
    sen_vals  = [sen.get(k, 0) for k in ranked]

    _barh(axes[0], crit_vals, "Criticality index (%)",     "% of simulations on CP")
    _barh(axes[1], drag_vals, "Expected drag (h)",         "Avg hours added when on CP")
    _barh(axes[2], sen_vals,  "CP sensitivity (ρ)",        "Correlation with overrun")

    fig.suptitle(
        title or "Activity risk ranking (sorted by CP sensitivity)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 3. CP path frequency heatmap
# ---------------------------------------------------------------------------

def plot_cp_path_heatmap(
    cp_paths: list[list[str]],
    *,
    title: str | None = None,
    top_n: int = 6,
    ax=None,
) -> tuple:
    """Frequency heatmap of the top-N most common critical-path routes.

    Parameters
    ----------
    cp_paths : list of lists, each being the ordered activity IDs on the CP
               for one simulation iteration (sim_result.cp_paths)
    top_n    : how many routes to show
    """
    fig, ax = _fig_ax(ax, (11, 3.5))
    n_runs = len(cp_paths)

    path_counts = Counter(tuple(p) for p in cp_paths)
    top_paths = path_counts.most_common(top_n)

    all_acts = sorted({a for path, _ in top_paths for a in path})
    matrix = np.zeros((len(top_paths), len(all_acts)))
    for i, (path, _) in enumerate(top_paths):
        for j, act in enumerate(all_acts):
            matrix[i, j] = 1.0 if act in path else 0.0

    ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    pct_labels = [f"{cnt / n_runs * 100:.0f}%" for _, cnt in top_paths]
    ax.set_yticks(range(len(top_paths)))
    ax.set_yticklabels(pct_labels, fontsize=9)
    ax.set_xticks(range(len(all_acts)))
    ax.set_xticklabels(all_acts, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Route frequency", fontsize=10)

    for i in range(len(top_paths)):
        for j in range(len(all_acts)):
            if matrix[i, j] > 0:
                ax.text(j, i, "●", ha="center", va="center",
                        color=_DARK, fontsize=13)

    ax.set_title(
        title or f"Top-{top_n} critical-path routes across {n_runs:,} simulations",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Per-activity duration distribution grid
# ---------------------------------------------------------------------------

def plot_duration_distributions(
    activities: list[dict],
    results: list[Any],
    *,
    max_cols: int = 3,
    title: str | None = None,
) -> tuple:
    """Grid of histograms showing sampled duration distributions.

    Parameters
    ----------
    activities : list of planned activity dicts (must have activity_id,
                 planned_duration_hours)
    results    : parallel list of ActivityDurationEstimate objects (from
                 service.estimate_activity); activities with no samples
                 (est.estimated_distribution is None) are skipped.
    max_cols   : columns in the subplot grid
    """
    plottable = [
        (act, est)
        for act, est in zip(activities, results)
        if est is not None
        and est.estimated_distribution is not None
        and len(est.estimated_distribution.samples or []) > 1
    ]

    if not plottable:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No distributions to plot", ha="center", va="center")
        return fig, ax

    n = len(plottable)
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                              squeeze=False)

    for idx, (act, est) in enumerate(plottable):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        dist = est.estimated_distribution
        samples = dist.samples or []

        color = _TIER_COLORS.get(est.confidence_tier, _BLUE)
        ax.hist(samples, bins=max(5, len(samples) // 3),
                color=color, alpha=0.75, edgecolor="white", linewidth=0.4)

        planned = act.get("planned_duration_hours", 0)
        ax.axvline(planned, color=_DARK,   lw=1.8, ls="--", label=f"Planned {planned:.0f}h")
        ax.axvline(dist.p50, color=_GREEN, lw=1.8, ls="-",  label=f"P50 {dist.p50:.1f}h")

        ax.set_title(f"{act['activity_id']}\n({est.confidence_tier})", fontsize=9)
        ax.set_xlabel("Duration (h)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7)

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(title or "Duration distributions from historical analogs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 5. Planned vs actual scatter
# ---------------------------------------------------------------------------

def plot_analog_scatter(
    activities: list[dict],
    results: list[Any],
    *,
    title: str | None = None,
    ax=None,
) -> tuple:
    """Planned vs P50 estimated duration, coloured by confidence tier.

    X-axis: planned_duration_hours
    Y-axis: estimated p50 duration (median of analog samples)
    Color:  confidence tier
    """
    fig, ax = _fig_ax(ax, (7, 5))

    for act, est in zip(activities, results):
        if est is None or est.estimated_distribution is None:
            continue
        planned = act.get("planned_duration_hours", 0)
        p50 = est.estimated_distribution.p50 or planned
        color = _TIER_COLORS.get(est.confidence_tier, _GREY)
        ax.scatter(planned, p50, color=color, s=90, zorder=3,
                   edgecolors="white", linewidth=0.8)
        ax.annotate(act["activity_id"], (planned, p50),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color=_DARK)

    # Perfect-estimate diagonal
    all_vals = [act.get("planned_duration_hours", 1) for act in activities]
    lim = max(all_vals) * 1.2
    ax.plot([0, lim], [0, lim], color=_GREY, ls="--", lw=1.2,
            label="P50 = planned (no overrun)")

    legend_patches = [
        mpatches.Patch(color=_GREEN,  label="High confidence (data-supported)"),
        mpatches.Patch(color=_ORANGE, label="Medium confidence (SME-informed)"),
        mpatches.Patch(color=_RED,    label="Low confidence"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    ax.set_xlabel("Planned duration (h)", fontsize=11)
    ax.set_ylabel("P50 estimated duration (h)", fontsize=11)
    ax.set_title(title or "Planned vs P50 estimated duration by confidence tier",
                 fontsize=11)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. Duration summary bars (planned vs p50 ± p80)
# ---------------------------------------------------------------------------

def plot_duration_summary_bars(
    activities: list[dict],
    results: list[Any],
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """Horizontal bar chart: planned duration vs P50 estimate ± P80 band.

    Each activity row shows:
      - grey bar   : planned duration
      - coloured bar: P50 estimate (coloured by confidence tier)
      - error bar  : P50 → P80 uncertainty band
    """
    n = len(activities)
    fig, ax = plt.subplots(figsize=figsize or (10, max(3, 0.6 * n + 1)))

    y = np.arange(n)
    act_ids  = [a["activity_id"] for a in activities]
    planned  = [a.get("planned_duration_hours", 0) for a in activities]
    p50_vals, p80_vals, colors = [], [], []

    for act, est in zip(activities, results):
        if est is not None and est.estimated_distribution is not None:
            dist = est.estimated_distribution
            p50_vals.append(dist.p50 or act.get("planned_duration_hours", 0))
            p80_vals.append(
                getattr(dist, "p80", None) or dist.p50 or act.get("planned_duration_hours", 0)
            )
            colors.append(_TIER_COLORS.get(est.confidence_tier, _BLUE))
        else:
            p50_vals.append(act.get("planned_duration_hours", 0))
            p80_vals.append(act.get("planned_duration_hours", 0))
            colors.append(_GREY)

    h = 0.35
    ax.barh(y + h / 2, planned, h, color=_GREY, alpha=0.55,
            label="Planned", zorder=2)
    ax.barh(y - h / 2, p50_vals, h, color=colors, alpha=0.85,
            label="P50 estimate", zorder=3)

    # P80 uncertainty whiskers
    for i, (p50, p80) in enumerate(zip(p50_vals, p80_vals)):
        if p80 > p50:
            ax.annotate(
                "", xy=(p80, y[i] - h / 2),
                xytext=(p50, y[i] - h / 2),
                arrowprops=dict(arrowstyle="-|>", color=_DARK,
                                lw=1.2, mutation_scale=8),
            )

    ax.set_yticks(y)
    ax.set_yticklabels(act_ids, fontsize=9)
    ax.set_xlabel("Duration (h)", fontsize=11)
    ax.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color=_GREY,   alpha=0.55, label="Planned"),
        mpatches.Patch(color=_GREEN,  alpha=0.85, label="P50 — high confidence"),
        mpatches.Patch(color=_ORANGE, alpha=0.85, label="P50 — medium confidence"),
        mpatches.Patch(color=_RED,    alpha=0.85, label="P50 — low confidence"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_title(
        title or "Duration estimate summary — planned vs P50 (arrow = P80)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 7. Pre-processing benchmark
# ---------------------------------------------------------------------------

def plot_preprocessing_benchmark(
    benchmark_rows: list[dict],
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """NED and exact-match rates before/after cleaning, broken down by category.

    Parameters
    ----------
    benchmark_rows : list of dicts with keys clean_description,
                     contaminated_description, category
    """
    from collections import defaultdict

    def _ned(a: str, b: str) -> float:
        a, b = a.strip().lower(), b.strip().lower()
        if not a and not b:
            return 0.0
        ratio = SequenceMatcher(None, a, b).ratio()
        return 1.0 - ratio

    def _clean(raw: str) -> str:
        try:
            from outage_uncertainty.preprocessing.abbreviations import AbbreviationResolver
            from outage_uncertainty.preprocessing.spell_checker import DomainSpellChecker
            resolver = AbbreviationResolver()
            checker  = DomainSpellChecker()
            return checker.transform(resolver.transform(raw))
        except Exception:
            return raw.lower()

    cat_data: dict[str, dict] = defaultdict(
        lambda: {"ned_raw": [], "ned_clean": [], "em_raw": 0, "em_clean": 0, "n": 0}
    )

    for row in benchmark_rows:
        ref   = row.get("clean_description", "")
        cont  = row.get("contaminated_description", "")
        cat   = row.get("category", "other")
        cleaned = _clean(cont)

        ned_raw   = _ned(ref, cont)
        ned_clean = _ned(ref, cleaned)

        d = cat_data[cat]
        d["ned_raw"].append(ned_raw)
        d["ned_clean"].append(ned_clean)
        d["em_raw"]   += int(ref.strip().lower() == cont.strip().lower())
        d["em_clean"] += int(ref.strip().lower() == cleaned.strip().lower())
        d["n"] += 1

    categories = sorted(cat_data)
    avg_ned_raw   = [np.mean(cat_data[c]["ned_raw"])   for c in categories]
    avg_ned_clean = [np.mean(cat_data[c]["ned_clean"]) for c in categories]
    em_raw_pct    = [cat_data[c]["em_raw"]   / cat_data[c]["n"] * 100 for c in categories]
    em_clean_pct  = [cat_data[c]["em_clean"] / cat_data[c]["n"] * 100 for c in categories]

    cat_labels = [c.replace("_", "\n") for c in categories]
    x = np.arange(len(categories))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (13, 4))

    ax1.bar(x - w / 2, avg_ned_raw,   w, color=_RED,    alpha=0.8, label="Before cleaning")
    ax1.bar(x + w / 2, avg_ned_clean, w, color=_GREEN,  alpha=0.8, label="After cleaning")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels, fontsize=8)
    ax1.set_ylabel("Avg NED (lower = better)", fontsize=10)
    ax1.set_title("Normalised edit distance", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, alpha=0.3)

    ax2.bar(x - w / 2, em_raw_pct,   w, color=_RED,   alpha=0.8, label="Before cleaning")
    ax2.bar(x + w / 2, em_clean_pct, w, color=_GREEN, alpha=0.8, label="After cleaning")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_labels, fontsize=8)
    ax2.set_ylabel("Exact-match rate (%)", fontsize=10)
    ax2.set_title("Exact-match recovery rate", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, alpha=0.3)

    fig.suptitle(
        title or "Pre-processing quality benchmark — NED and exact-match by category",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# 8. Routine vs disruption-driven separation
# ---------------------------------------------------------------------------

def plot_routine_vs_disruption(
    dist,
    activity_id: str = "",
    *,
    planned_hours: float | None = None,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """Two-panel figure visualising routine vs. disruption-driven execution modes.

    This corresponds to Slide 6 step 3: *'Separate routine execution (typical
    cases) from disruption-driven outliers (scope expansions, rework, parts
    delays).'*

    Left panel
        Overlaid histograms — routine pool (blue) and disruption-driven pool
        (red), with vertical marker lines for the planned baseline, routine
        P50, routine P80, and mixture-aware P80.  The IQR separation fence is
        annotated when available.

    Right panel
        Empirical CDF curves comparing the routine-only distribution against
        the full mixture (sampled from both pools with ``mixture_weight``
        probability for the disruption mode).  The gap between the two P80
        lines is the contingency underestimate when disruption is ignored.

    Parameters
    ----------
    dist          : DurationDistribution with ``samples``, optionally
                    ``extended_samples`` and ``mixture_weight``.
    activity_id   : Label shown in the title.
    planned_hours : Baseline planned duration to mark as a vertical line.
    """
    routine  = list(dist.samples or [])
    extended = list(dist.extended_samples or [])
    has_disruption = bool(extended) and dist.mixture_weight > 0.0

    fence     = dist.parameters.get("outlier_threshold")
    mix_p80   = dist.parameters.get("mixture_p80")
    ext_frac  = dist.parameters.get("extended_fraction", dist.mixture_weight)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (13, 5))

    # ------------------------------------------------------------------
    # Left: overlaid histograms
    # ------------------------------------------------------------------
    all_vals = routine + extended
    if not all_vals:
        ax1.text(0.5, 0.5, "No samples available", ha="center", va="center",
                 transform=ax1.transAxes)
    else:
        rng_min = min(all_vals) * 0.85
        rng_max = max(all_vals) * 1.12
        bins = np.linspace(rng_min, rng_max, min(30, max(10, len(all_vals) // 2)))

        ax1.hist(routine, bins=bins, color=_BLUE, alpha=0.78,
                 edgecolor="white", linewidth=0.4,
                 label=f"Routine execution  (n={len(routine)})")
        if has_disruption:
            ax1.hist(extended, bins=bins, color=_RED, alpha=0.65,
                     edgecolor="white", linewidth=0.4,
                     label=(f"Disruption-driven  "
                            f"(n={len(extended)},  {ext_frac * 100:.0f}% of jobs)"))

        # Vertical marker lines
        if planned_hours:
            ax1.axvline(planned_hours, color=_DARK, lw=2.0, ls="-",
                        label=f"Planned  {planned_hours:.0f} h")
        if dist.p50:
            ax1.axvline(dist.p50, color=_GREEN, lw=1.8, ls="--",
                        label=f"Routine P50  {dist.p50:.1f} h")
        if dist.p80:
            ax1.axvline(dist.p80, color=_ORANGE, lw=1.8, ls="--",
                        label=f"Routine P80  {dist.p80:.1f} h")
        if mix_p80 and has_disruption:
            ax1.axvline(mix_p80, color=_RED, lw=1.8, ls=":",
                        label=f"Mixture P80  {mix_p80:.1f} h  ← true contingency anchor")
        if fence:
            ax1.axvline(fence, color=_GREY, lw=1.2, ls=":", alpha=0.6,
                        label=f"IQR separation fence  {fence:.1f} h")

        ax1.set_xlabel("Duration (h)", fontsize=11)
        ax1.set_ylabel("Count", fontsize=11)
        ax1.legend(fontsize=8.5, loc="upper right")
        ax1.yaxis.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Right: CDF comparison — routine-only vs mixture-aware
    # ------------------------------------------------------------------
    if routine:
        sorted_r = sorted(routine)
        n_r = len(sorted_r)
        cdf_r = [(i + 1) / n_r for i in range(n_r)]
        ax2.step(sorted_r, cdf_r, color=_BLUE, lw=2.2, where="post",
                 label="Routine-only CDF")

        if has_disruption:
            # Build mixture CDF by drawing from both pools
            _random.seed(42)
            n_mix = max(2000, len(all_vals) * 50)
            mix_samples = sorted(
                _random.choice(extended) if _random.random() < dist.mixture_weight
                else _random.choice(routine)
                for _ in range(n_mix)
            )
            cdf_m = [(i + 1) / n_mix for i in range(n_mix)]
            ax2.step(mix_samples, cdf_m, color=_RED, lw=1.8, where="post",
                     alpha=0.75,
                     label=(f"Mixture CDF  "
                            f"({dist.mixture_weight * 100:.0f}% disruption weight)"))

        # Horizontal reference lines at P80 / P90
        for q, label in [(0.80, "80th pct"), (0.90, "90th pct")]:
            ax2.axhline(q, color=_GREY, lw=0.9, ls=":", alpha=0.7)
            ax2.text(ax2.get_xlim()[0] if ax2.get_xlim()[0] > 0 else 0,
                     q + 0.012, f" {label}", fontsize=7.5, color=_GREY)

        # Mark routine P80 and mixture P80 on the CDF
        if dist.p80:
            ax2.axvline(dist.p80, color=_ORANGE, lw=1.5, ls="--",
                        label=f"Routine P80 = {dist.p80:.1f} h")
        if mix_p80 and has_disruption:
            ax2.axvline(mix_p80, color=_RED, lw=1.5, ls=":",
                        label=f"Mixture P80 = {mix_p80:.1f} h")

        ax2.set_xlabel("Duration (h)", fontsize=11)
        ax2.set_ylabel("Cumulative probability", fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.yaxis.grid(True, alpha=0.3)
        ax2.legend(fontsize=8.5)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        label = f"  —  {activity_id}" if activity_id else ""
        disruption_note = (
            f"\n{ext_frac * 100:.0f}% of historical jobs ran in disrupted mode  "
            f"(mixture P80 − routine P80 = "
            f"{(mix_p80 or dist.p80 or 0) - (dist.p80 or 0):+.1f} h contingency gap)"
            if has_disruption and dist.p80 and mix_p80
            else ""
        )
        fig.suptitle(
            f"Routine vs. disruption-driven execution modes{label}{disruption_note}",
            fontsize=11, fontweight="bold",
        )

    fig.tight_layout()
    return fig, (ax1, ax2)
