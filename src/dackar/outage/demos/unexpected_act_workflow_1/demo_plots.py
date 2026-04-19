"""
demo_plots.py — Reusable plot functions for the DACKAR unexpected-activity triage demo.

All functions follow a consistent signature:

    plot_*(data, ..., *, title=None, ax=None or figsize=None) -> (fig, ax_or_axes)

They can be used standalone (figure created internally when ax is None) or composed
into a larger figure by passing an existing Axes object.  Every function returns
(fig, ax) so callers can save, annotate, or embed the result.

Functions
---------
draw_pipeline_architecture     A→B/D→C→E→F→G stage diagram with stub/live colouring
plot_stage_a_summary           3-panel: data-quality gauges / entities / regulatory drivers
plot_allen_timeline            Stage C temporal event chain (Allen interval relations)
plot_analog_distribution       Stage D: duration histogram + per-analog dot plot
plot_schedule_impact           Stage E: float waterfall + CP-drag / duration bar chart
plot_option_risk_scores        Stage F: horizontal risk-score bars with feasibility hatching
plot_recommendation_card       Stage G: full recommendation card (matplotlib figure)
plot_scenario_comparison       Radar + risk-bar + metrics table across two or three scenarios
plot_evidence_chain            Stage G: source-typed evidence items with confidence bars
"""
from __future__ import annotations

import math
import textwrap
from collections import Counter
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Shared colour palette
# ---------------------------------------------------------------------------

PALETTE: dict[str, str] = {
    "escalate":        "#D7263D",   # red
    "proceed":         "#06A77D",   # teal-green
    "defer":           "#F4A261",   # amber
    "monitor":         "#A8DADC",   # light blue
    "blocked":         "#E76F51",   # orange-red (reg-blocked option)
    "infeasible":      "#CCCCCC",   # light grey (infeasible option)
    "data_supported":  "#264653",   # dark teal   (≥5 analogues)
    "sme_informed":    "#2A9D8F",   # medium teal (1–4 analogues)
    "low_confidence":  "#E9C46A",   # gold        (0 analogues)
    "stage_live":      "#264653",   # dark teal   (stage runs real logic)
    "stage_stub":      "#94A3B8",   # slate grey  (pre-built stub)
    "stage_text":      "#FFFFFF",   # white text on stage tiles
}


def _fig_ax(ax, figsize: tuple | None) -> tuple:
    """Return (fig, ax): create a new figure when *ax* is None."""
    if ax is not None:
        return ax.get_figure(), ax
    fig, ax = plt.subplots(figsize=figsize or (10, 4.5))
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Pipeline architecture diagram
# ---------------------------------------------------------------------------

def draw_pipeline_architecture(
    *,
    highlight_stage: str | None = None,
    title: str | None = None,
    ax=None,
) -> tuple:
    """Draw the DACKAR A→G stage pipeline diagram.

    Parameters
    ----------
    highlight_stage : optional stage letter (e.g. ``"E"``) to outline in accent colour
    title           : override the default figure title
    ax              : existing Axes to draw into; a new figure is created when None
    """
    fig, ax = _fig_ax(ax, (14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(
        title or "DACKAR Outage Activity Analysis — Stage Pipeline",
        fontsize=13, fontweight="bold", pad=12,
    )

    stages = [
        # (x_center, y_center, letter, sub-label, is_stub)
        (1.0,  2.0, "A", "Activity\nIntake",          False),
        (3.0,  3.0, "B", "KG\nTimeline",              False),
        (3.0,  1.0, "D", "Historical\nAnalogs",       False),
        (5.5,  2.0, "C", "Temporal\nChain",           False),
        (8.0,  2.0, "E", "Schedule\nImpact",          True),
        (10.5, 2.0, "F", "Option\nGeneration",        False),
        (13.0, 2.0, "G", "Recommendation\nSynthesis", False),
    ]

    BOX_W, BOX_H = 1.6, 0.9
    boxes: dict[str, tuple[float, float]] = {}

    for cx, cy, lbl, sub, stub in stages:
        color = PALETTE["stage_stub"] if stub else PALETTE["stage_live"]
        edge_color = "#FFD700" if lbl == highlight_stage else "white"
        edge_lw = 3.0 if lbl == highlight_stage else 1.5
        rect = mpatches.FancyBboxPatch(
            (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.08", linewidth=edge_lw,
            edgecolor=edge_color, facecolor=color, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + 0.10, lbl, ha="center", va="center",
                fontsize=15, fontweight="bold", color="white", zorder=4)
        ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                fontsize=7.0, color="white", zorder=4, linespacing=1.3)
        boxes[lbl] = (cx, cy)

    def _arrow(x1, y1, x2, y2, *, style="arc3,rad=0"):
        ax.annotate(
            "", xy=(x2 - BOX_W / 2 - 0.05, y2),
            xytext=(x1 + BOX_W / 2 + 0.05, y1),
            arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5,
                            connectionstyle=style),
        )

    _arrow(*boxes["A"], *boxes["B"], style="arc3,rad=-0.35")
    _arrow(*boxes["A"], *boxes["D"], style="arc3,rad=0.35")
    _arrow(*boxes["B"], *boxes["C"], style="arc3,rad=0.30")
    _arrow(*boxes["D"], *boxes["C"], style="arc3,rad=-0.30")
    _arrow(*boxes["C"], *boxes["E"])
    _arrow(*boxes["E"], *boxes["F"])
    _arrow(*boxes["F"], *boxes["G"])

    leg = [
        mpatches.Patch(color=PALETTE["stage_live"], label="Real production logic"),
        mpatches.Patch(color=PALETTE["stage_stub"], label="Pre-built stub (LOGOS CPM)"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Stage A — intake quality summary
# ---------------------------------------------------------------------------

def plot_stage_a_summary(
    intake: dict,
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """3-panel Stage A summary: quality gauges / entity types / regulatory drivers.

    Parameters
    ----------
    intake : the ``r['intake']`` dict from :func:`run_pipeline`
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize or (13, 3.5))
    fig.suptitle(title or "Stage A — Activity Intake & Classification",
                 fontsize=12, fontweight="bold")

    # Panel 1: data-quality bars
    ax = axes[0]
    dq   = intake.get("data_quality_score", 0)
    abbr = intake.get("unknown_abbreviation_rate", 0)
    metrics = ["Data Quality", "Abbr Clarity"]
    values  = [dq, 1 - abbr]
    colors  = [PALETTE["proceed"] if v >= 0.6 else PALETTE["defer"] for v in values]
    bars = ax.barh(metrics, values, color=colors, edgecolor="white", height=0.45)
    ax.set_xlim(0, 1)
    ax.axvline(0.6, ls="--", color="#888", lw=1, label="threshold 0.6")
    ax.set_title("Input Quality Scores")
    ax.set_xlabel("Score [0–1]")
    for bar, v in zip(bars, values):
        ax.text(min(v + 0.02, 0.92), bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=9)
    ax.legend(fontsize=7)

    # Panel 2: entity type counts
    ax = axes[1]
    entities    = intake.get("extracted_entities", [])
    type_counts = Counter(e["entity_type"] for e in entities)
    if type_counts:
        types, counts = zip(*sorted(type_counts.items(), key=lambda x: -x[1]))
        ax.barh(types, counts, color=PALETTE["stage_live"], edgecolor="white", height=0.5)
        ax.set_xlabel("Count")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"Extracted Entities ({len(entities)})")

    # Panel 3: regulatory drivers
    ax = axes[2]
    drivers = intake.get("regulatory_drivers", [])
    if drivers:
        labels   = [d["driver_type"].replace("_", "\n") for d in drivers]
        colors_d = [
            PALETTE["escalate"] if d.get("defer_prohibited") else PALETTE["defer"]
            for d in drivers
        ]
        ax.barh(labels, [1] * len(drivers), color=colors_d, edgecolor="white", height=0.5)
        ax.set_xlim(0, 1.5)
        ax.set_xticks([])
        ax.set_title(f"Regulatory Drivers ({len(drivers)})")
        ax.legend(
            handles=[
                mpatches.Patch(color=PALETTE["escalate"], label="defer_prohibited"),
                mpatches.Patch(color=PALETTE["defer"],    label="defer_allowed"),
            ],
            fontsize=7, loc="lower right",
        )
    else:
        ax.text(0.5, 0.5, "No regulatory\nconstraints detected",
                ha="center", va="center", fontsize=10, color="#888")
        ax.set_title("Regulatory Drivers (0)")
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 3. Stage C — temporal event chain (Allen interval algebra)
# ---------------------------------------------------------------------------

_ALLEN_COLORS: dict[str, str] = {
    "precedes":     "#2A9D8F",
    "overlaps":     "#E76F51",
    "contains":     "#D7263D",
    "during":       "#A8DADC",
    "follows":      "#E9C46A",
    "simultaneous": "#264653",
    "unknown":      "#CCCCCC",
}


def plot_allen_timeline(
    temporal: dict,
    *,
    title: str | None = None,
    ax=None,
) -> tuple:
    """Stage C temporal event chain visualisation using Allen interval relations.

    Parameters
    ----------
    temporal : the ``r['temporal']`` dict from :func:`run_pipeline`
    """
    chain_links       = temporal.get("chain_links", [])
    activity_interval = temporal.get("emergent_activity_interval", {})
    summary           = temporal.get("summary", {})

    if not chain_links:
        fig, ax = _fig_ax(ax, (10, 3))
        ax.text(0.5, 0.5, "No chain links to visualise.",
                ha="center", va="center", fontsize=11, color="#888")
        ax.axis("off")
        return fig, ax

    fig, ax = _fig_ax(ax, (13, max(3, len(chain_links) * 0.9 + 2)))

    def _days_ago(ts_str: str) -> float:
        try:
            from datetime import datetime, timezone
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            act_ts = datetime.fromisoformat(
                activity_interval.get("start", ts_str)
            )
            if act_ts.tzinfo is None:
                act_ts = act_ts.replace(tzinfo=timezone.utc)
            return float((ts - act_ts).days)
        except Exception:
            return 0.0

    y_labels: list[str] = []
    for i, link in enumerate(chain_links):
        # Support both the legacy 'event_timestamp' key and the current schema
        ts_key    = "prior_event_timestamp" if "prior_event_timestamp" in link else "event_timestamp"
        type_key  = "prior_event_type"      if "prior_event_type"      in link else "event_type"
        x_event = _days_ago(link.get(ts_key, ""))
        rel     = link.get("allen_relation", "unknown")
        color   = _ALLEN_COLORS.get(rel, "#888")
        score   = link.get("relation_score", 0)
        conf    = link.get("confidence", 0)
        event_type = link.get(type_key, link.get("prior_event_id", "?"))

        ax.scatter(x_event, i, s=120, color=color, zorder=5,
                   edgecolors="white", linewidths=1)
        ax.annotate(
            "", xy=(0, i),
            xytext=(x_event + (0.2 if x_event < 0 else -0.2), i),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7),
        )
        y_labels.append(
            f"{event_type}  |  {rel.upper()}  |  "
            f"score={score:.2f}  conf={conf:.2f}"
        )

    ax.axvline(0, color=PALETTE["escalate"], lw=2.5, ls="-",
               label="Emergent activity detected", zorder=4)
    ax.set_yticks(range(len(chain_links)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Days relative to detection (negative = before detection)")
    ax.invert_yaxis()

    posture  = summary.get("causal_posture", "?")
    n_strong = summary.get("strong_link_count", 0)
    ax.set_title(
        (title or "Stage C — Temporal Event Chain") + "\n"
        f"Causal posture: {posture.upper()}  |  "
        f"Strong links: {n_strong}  |  "
        f"Temporal contradiction: {summary.get('has_temporal_contradiction', False)}",
        fontsize=10,
    )
    ax.legend(
        handles=[mpatches.Patch(color=c, label=r.upper())
                 for r, c in _ALLEN_COLORS.items()],
        fontsize=7, loc="lower right", title="Allen Relation", title_fontsize=7, ncol=2,
    )

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Stage D — analog duration distribution
# ---------------------------------------------------------------------------

def plot_analog_distribution(
    analogs_result: dict,
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """Stage D: histogram of analog durations + per-analog dot plot.

    Parameters
    ----------
    analogs_result : the ``r['analogs']`` dict from :func:`run_pipeline`
    """
    analogs  = analogs_result.get("analogs", [])
    dist     = analogs_result.get("duration_distribution", {})
    summary  = analogs_result.get("retrieval_summary", {})
    tier     = dist.get("confidence_tier", "low_confidence")
    p50, p80, p90 = dist.get("p50_hours"), dist.get("p80_hours"), dist.get("p90_hours")

    durations = [a["actual_duration_hours"] for a in analogs
                 if a.get("actual_duration_hours") is not None]

    if not durations:
        fig, ax = _fig_ax(None, figsize or (10, 4))
        ax.text(0.5, 0.5, "No analog durations available.",
                ha="center", va="center", fontsize=11, color="#888")
        ax.axis("off")
        return fig, ax

    tier_color = PALETTE.get(tier, "#888")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (12, 4))
    fig.suptitle(title or "Stage D — Historical Analog Duration Distribution",
                 fontsize=12, fontweight="bold")

    # Histogram
    ax1.hist(durations, bins=max(3, len(durations) // 2 + 1),
             color=tier_color, edgecolor="white", alpha=0.85, rwidth=0.8)
    if p50:
        ax1.axvline(p50, color="#F4A261", lw=2, ls="--", label=f"p50 = {p50:.0f} h")
    if p80:
        ax1.axvline(p80, color="#E76F51", lw=2, ls="--", label=f"p80 = {p80:.0f} h")
    if p90:
        ax1.axvline(p90, color="#D7263D", lw=2, ls="--", label=f"p90 = {p90:.0f} h")
    ax1.set_xlabel("Actual Duration (hours)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Duration Histogram  (n={len(durations)}, tier={tier})")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(fontsize=8)
    ax1.text(0.97, 0.97, tier.replace("_", " ").upper(),
             transform=ax1.transAxes, ha="right", va="top", fontsize=9,
             color="white", fontweight="bold",
             bbox=dict(facecolor=tier_color, edgecolor="none", pad=4, alpha=0.9))

    # Per-analog dot plot
    sorted_a = sorted(analogs, key=lambda x: x.get("actual_duration_hours") or 0)
    ys     = list(range(len(sorted_a)))
    xs     = [a.get("actual_duration_hours") or 0 for a in sorted_a]
    labels = [
        f"{a.get('outage_id', '?')} — {a.get('description', '')[:40]}"
        for a in sorted_a
    ]
    ax2.scatter(xs, ys, color=tier_color, s=80, zorder=5, edgecolors="white")
    if p50:
        ax2.axvline(p50, color="#F4A261", lw=2, ls="--", alpha=0.8)
    if p80:
        ax2.axvline(p80, color="#E76F51", lw=2, ls="--", alpha=0.8)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel("Actual Duration (hours)")
    ax2.set_title("Analog Records")

    fig.tight_layout()
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# 5. Stage E — schedule impact
# ---------------------------------------------------------------------------

def plot_schedule_impact(
    schedule: dict,
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """Stage E: float waterfall + CP-drag / duration percentile bars.

    Parameters
    ----------
    schedule : the ``r['schedule']`` dict from :func:`run_pipeline`
    """
    float_analysis = schedule.get("float_analysis", {})
    cp_impact      = schedule.get("cp_impact", {})
    dur_est        = schedule.get("duration_estimate", {})
    insertion_pt   = schedule.get("insertion_point", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (12, 4))
    fig.suptitle(title or "Stage E — Schedule Impact Assessment",
                 fontsize=12, fontweight="bold")

    # Float waterfall
    avail     = float_analysis.get("available_float_before_hours",
                  float_analysis.get("available_float_before", 0)) or 0
    consumed  = float_analysis.get("float_consumed_hours", 0) or 0
    remaining = float_analysis.get("remaining_float_after_hours",
                  float_analysis.get("remaining_float_after", 0)) or 0
    criticality = float_analysis.get("criticality_label", "unknown")

    bar_labels = ["Float\nAvailable", "Float\nConsumed", "Float\nRemaining"]
    bar_values = [avail, consumed, max(remaining, 0)]
    bar_colors = [
        PALETTE["proceed"],
        PALETTE["escalate"] if consumed > avail else PALETTE["defer"],
        PALETTE["proceed"] if remaining > 8
        else (PALETTE["defer"] if remaining > 0 else PALETTE["escalate"]),
    ]
    bars = ax1.bar(bar_labels, bar_values, color=bar_colors,
                   edgecolor="white", width=0.5)
    ax1.set_ylabel("Hours")
    ax1.set_title(f"Float Analysis  ({criticality.upper()})")
    for bar, val in zip(bars, bar_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 f"{val:.0f} h", ha="center", fontsize=9, fontweight="bold")
    if remaining < 0:
        ax1.text(2, 1, f"({remaining:.0f} h deficit)",
                 ha="center", fontsize=8, color=PALETTE["escalate"])

    # CP impact + duration percentiles
    cp_drag  = cp_impact.get("cp_drag_hours", 0) or 0
    baseline = cp_impact.get("baseline_cp_hours", 480) or 480
    sens     = cp_impact.get("cp_sensitivity_score", 0) or 0
    p50 = dur_est.get("p50_hours") or 0
    p80 = dur_est.get("p80_hours") or 0
    p90 = dur_est.get("p90_hours") or 0

    categories = ["p50 Duration", "p80 Duration", "p90 Duration", "CP Drag"]
    values     = [p50, p80, p90, cp_drag]
    colors_v   = [PALETTE["stage_live"]] * 3 + [
        PALETTE["escalate"] if cp_drag > 24 else PALETTE["defer"]
    ]
    ax2.barh(categories, values, color=colors_v, edgecolor="white", height=0.5)
    ax2.set_xlabel("Hours")
    ax2.set_title(
        f"Duration Estimates & CP Drag\n"
        f"Baseline CP: {baseline:.0f} h  |  Sensitivity: {sens:.2f}"
    )
    ax2.axvline(24, color="#888", lw=1, ls=":", label="Escalate threshold (24 h)")
    for i, v in enumerate(values):
        ax2.text(v + 0.5, i, f"{v:.0f} h", va="center", fontsize=9)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# 6. Stage F — insertion option risk scores
# ---------------------------------------------------------------------------

def plot_option_risk_scores(
    options_result: dict,
    *,
    title: str | None = None,
    ax=None,
) -> tuple:
    """Stage F: horizontal bar chart of option risk scores with feasibility hatching.

    Parameters
    ----------
    options_result : the ``r['options']`` dict from :func:`run_pipeline`
    """
    options = options_result.get("options", [])
    rec_id  = options_result.get("recommended_option_id")
    summary = options_result.get("ranking_summary", {})

    if not options:
        fig, ax = _fig_ax(ax, (10, 3))
        ax.text(0.5, 0.5, "No options generated.", ha="center", va="center",
                fontsize=11, color="#888")
        ax.axis("off")
        return fig, ax

    fig, ax = _fig_ax(ax, (11, max(3, len(options) * 0.8 + 1.5)))

    def _bar_color(opt: dict) -> str:
        if opt.get("option_id") == rec_id:
            return (PALETTE["escalate"]
                    if opt.get("option_type") == "escalate_to_management"
                    else PALETTE["proceed"])
        if not opt.get("feasible", True):
            return PALETTE["infeasible"]
        if not opt.get("regulatory_cleared", True):
            return PALETTE["blocked"]
        return "#A8DADC"

    labels: list[str] = []
    scores: list[float] = []
    colors: list[str] = []
    hatches: list[str] = []

    for opt in options:
        otype    = opt.get("option_type", "?").replace("_", " ")
        feasible = opt.get("feasible", True)
        cleared  = opt.get("regulatory_cleared", True)
        suffix = (
            "  ← RECOMMENDED"   if opt.get("option_id") == rec_id
            else "  [INFEASIBLE]"  if not feasible
            else "  [REG BLOCKED]" if not cleared
            else ""
        )
        labels.append(f"{otype}{suffix}")
        scores.append(opt.get("risk_score", 0))
        colors.append(_bar_color(opt))
        hatches.append("//" if not feasible or not cleared else "")

    y_pos = list(range(len(labels)))
    bars  = ax.barh(y_pos, scores, color=colors, edgecolor="white",
                    height=0.55, linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("#888")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Risk Score (lower = better)", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="#888", lw=1, ls=":", alpha=0.7)
    for i, score in enumerate(scores):
        ax.text(score + 0.01, i, f"{score:.3f}", va="center", fontsize=8)
    ax.invert_yaxis()

    ax.legend(
        handles=[
            mpatches.Patch(color=PALETTE["proceed"],    label="Recommended (PROCEED)"),
            mpatches.Patch(color=PALETTE["escalate"],   label="Recommended (ESCALATE)"),
            mpatches.Patch(color="#A8DADC",             label="Feasible + cleared"),
            mpatches.Patch(color=PALETTE["blocked"],    label="Regulatory blocked"),
            mpatches.Patch(color=PALETTE["infeasible"], label="Infeasible"),
        ],
        fontsize=7, loc="lower right",
    )
    ax.set_title(title or "Stage F — Insertion Option Risk Scores",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 7. Stage G — recommendation card (matplotlib figure)
# ---------------------------------------------------------------------------

_STATUS_BG: dict[str, str] = {
    "ESCALATE":     "#D7263D",
    "PROCEED":      "#06A77D",
    "DEFER":        "#F4A261",
    "MONITOR":      "#2A9D8F",
    "INCONCLUSIVE": "#94A3B8",
}


def plot_recommendation_card(
    result: dict,
    *,
    title: str | None = None,
    ax=None,
) -> tuple:
    """Render the Stage G recommendation as a matplotlib card figure.

    Parameters
    ----------
    result : the full dict returned by :func:`run_pipeline` for one scenario
    """
    rec    = result["recommendation"]
    summ   = rec.get("executive_summary", {})
    prim   = rec.get("primary_recommendation", {})
    hist   = rec.get("history_summary", {})
    rev    = rec.get("analyst_review", {})
    flags  = summ.get("analyst_attention_flags", rec.get("attention_flags", []))
    reg    = rec.get("regulatory_flags", [])
    status = rec["decision_status"]
    label  = result.get("scenario_label", "")

    bg = _STATUS_BG.get(status, "#888888")
    fig, ax = _fig_ax(ax, (11, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Header band
    ax.add_patch(mpatches.FancyBboxPatch((0, 8.5), 10, 1.5, boxstyle="square",
                                          facecolor=bg, edgecolor="none"))
    ax.text(5, 9.5, f"DACKAR — {label}", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold")
    ax.text(5, 8.85, f"DECISION: {status}", ha="center", va="center",
            fontsize=18, color="white", fontweight="bold")

    # Body background
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 10, 8.5, boxstyle="square",
                                          facecolor="#F8FAFC", edgecolor="#CBD5E1",
                                          linewidth=1))

    # Confidence tier pill
    tier = summ.get("confidence_tier", "?")
    tc   = PALETTE.get(tier, "#888")
    ax.text(0.3, 8.15, f"Confidence: {tier.replace('_', ' ').upper()}",
            fontsize=8.5, color="white", fontweight="bold",
            bbox=dict(facecolor=tc, edgecolor="none", pad=4, boxstyle="round"))

    # Analyst review flag
    if rev.get("required"):
        ax.text(9.7, 8.15, "⚑ ANALYST REVIEW REQUIRED",
                fontsize=8, color=PALETTE["escalate"], fontweight="bold", ha="right")

    # Primary conclusion
    conclusion = summ.get("primary_conclusion", "")
    ax.text(0.3, 7.7, "Primary Conclusion:", fontsize=9, fontweight="bold",
            color="#334155")
    ax.text(0.3, 7.3, textwrap.fill(conclusion, 80), fontsize=8.5, color="#475569",
            va="top",
            bbox=dict(facecolor="#E2E8F0", edgecolor="none", pad=6, boxstyle="round"))

    # Recommended option
    opt_type = prim.get("option_type", "?").replace("_", " ")
    opt_cp   = prim.get("cp_impact_hours", 0) or 0
    ax.text(0.3, 5.8,
            f"Recommended option: {opt_type.upper()}  |  CP impact: {opt_cp:.0f} h",
            fontsize=9, fontweight="bold", color="#334155")
    ax.text(0.3, 5.5, textwrap.fill(prim.get("rationale", ""), 80),
            fontsize=8, color="#475569", va="top")

    # Key metrics row
    ax.text(0.3, 3.9, "Key Metrics", fontsize=9, fontweight="bold", color="#334155")
    metrics = [
        ("Analogs",    f"{hist.get('analog_count', 0)} events / "
                       f"{hist.get('outages_represented', 0)} outages"),
        ("Median dur", f"{hist.get('median_actual_hours', '?')} h"),
        ("CP drag",    f"{opt_cp:.0f} h"),
        ("Reg flags",  str(len(reg))),
    ]
    for i, (k, v) in enumerate(metrics):
        xoff = 0.3 + i * 2.4
        ax.add_patch(mpatches.FancyBboxPatch(
            (xoff, 3.0), 2.0, 0.75, boxstyle="round,pad=0.05",
            facecolor=bg, edgecolor="none", alpha=0.15,
        ))
        ax.text(xoff + 1.0, 3.55, k, ha="center", fontsize=7.5, color="#475569")
        ax.text(xoff + 1.0, 3.15, v, ha="center", fontsize=9,
                fontweight="bold", color="#1E293B")

    # Attention flags
    ax.text(0.3, 2.7, "Attention Flags", fontsize=9, fontweight="bold", color="#334155")
    if flags:
        for i, flag in enumerate(flags[:4]):
            col = PALETTE["escalate"] if ("regulatory" in flag or "critical" in flag) else "#475569"
            ax.text(0.5, 2.35 - i * 0.35, f"⚑  {flag.replace('_', ' ')}",
                    fontsize=8, color=col)
    else:
        ax.text(0.5, 2.35, "No attention flags raised.", fontsize=8, color="#888")

    # Footer
    ax.text(9.7, 0.15, f"Run ID: {result.get('run_id', '?')}",
            ha="right", fontsize=7, color="#94A3B8")

    ax.set_title(title or "", fontsize=10)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 8. Multi-scenario comparison (radar + risk bars + metrics table)
# ---------------------------------------------------------------------------

def plot_scenario_comparison(
    *results: dict,
    scenario_labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """Radar + risk-bar + metrics table comparison across 2–3 scenario results.

    Parameters
    ----------
    *results         : two or three dicts returned by :func:`run_pipeline`
    scenario_labels  : override labels (default: uses each result's ``scenario_label``)
    """
    if len(results) < 2:
        raise ValueError("plot_scenario_comparison requires at least two results")
    labels = scenario_labels or [r.get("scenario_label", f"S{i+1}") for i, r in enumerate(results)]

    _COLORS = [PALETTE["escalate"], PALETTE["proceed"], PALETTE["monitor"],
               PALETTE["defer"], "#264653"]

    fig = plt.figure(figsize=figsize or (15, 6))
    fig.suptitle(title or "Scenario Comparison — DACKAR Pipeline Output",
                 fontsize=13, fontweight="bold")

    # ── Radar chart
    ax_radar = fig.add_subplot(1, 3, 1, polar=True)
    dimensions = ["Data Quality", "Analog\nCoverage", "Temporal\nCausality",
                  "CP Float\nRemaining", "Option\nConfidence"]
    N      = len(dimensions)
    angles = [n / float(N) * 2 * math.pi for n in range(N)] + [0]

    def _radar_values(r: dict) -> list[float]:
        intake  = r["intake"]
        analogs = r["analogs"]
        sched   = r["schedule"]
        options = r["options"]
        temporal = r["temporal"]

        dq           = intake.get("data_quality_score", 0) or 0
        analog_count = analogs.get("retrieval_summary", {}).get("analog_count", 0) or 0
        analog_score = min(1.0, analog_count / 10.0)
        strongest    = temporal.get("summary", {}).get("strongest_link_score", 0) or 0
        remaining    = sched.get("float_analysis", {}).get(
            "remaining_float_after_hours",
            sched.get("float_analysis", {}).get("remaining_float_after", 0)
        ) or 0
        baseline = sched.get("cp_impact", {}).get("baseline_cp_hours", 480) or 480
        float_score = max(0.0, min(1.0, remaining / baseline))
        rec_id = options.get("recommended_option_id")
        opt_conf = 0.5
        for opt in options.get("options", []):
            if opt.get("option_id") == rec_id:
                opt_conf = float(opt.get("confidence") or 0.5)
                break
        return [dq, analog_score, strongest, float_score, opt_conf]

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(dimensions, fontsize=8)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75])
    ax_radar.set_yticklabels(["0.25", "0.5", "0.75"], fontsize=6)

    for r, lbl, col in zip(results, labels, _COLORS):
        vals = _radar_values(r) + [_radar_values(r)[0]]
        ax_radar.plot(angles, vals, color=col, lw=2, label=lbl[:20])
        ax_radar.fill(angles, vals, color=col, alpha=0.10)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)
    ax_radar.set_title("Risk Profile Radar", fontsize=10, pad=12)

    # ── Risk score bar chart
    ax_bar = fig.add_subplot(1, 3, 2)

    def _feasible_scores(r: dict) -> dict[str, float]:
        return {
            o["option_type"].replace("_", " "): o["risk_score"]
            for o in r["options"].get("options", [])
            if o.get("feasible", True) and o.get("regulatory_cleared", True)
        }

    all_opts = sorted({k for r in results for k in _feasible_scores(r)})
    x = np.arange(len(all_opts))
    w = 0.8 / len(results)
    for i, (r, lbl, col) in enumerate(zip(results, labels, _COLORS)):
        scores_d = _feasible_scores(r)
        ax_bar.bar(
            x + (i - len(results) / 2 + 0.5) * w,
            [scores_d.get(o, 0) for o in all_opts],
            w, label=lbl[:15], color=col, alpha=0.85,
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(all_opts, rotation=30, ha="right", fontsize=7.5)
    ax_bar.set_ylabel("Risk Score")
    ax_bar.set_title("Option Risk Scores\n(feasible + cleared only)")
    ax_bar.axhline(0.5, color="#888", lw=1, ls=":")
    ax_bar.legend(fontsize=8)
    ax_bar.set_ylim(0, 1.0)

    # ── Metrics table
    ax_tbl = fig.add_subplot(1, 3, 3)
    ax_tbl.axis("off")
    ax_tbl.set_title("Key Metrics Comparison", fontsize=10)

    def _fmt(r: dict) -> list[str]:
        rec     = r["recommendation"]
        intake  = r["intake"]
        sched   = r["schedule"]
        analogs = r["analogs"]
        summ    = rec.get("executive_summary", {})
        return [
            rec["decision_status"],
            intake.get("emergence_type", "?"),
            str(intake.get("has_regulatory_constraint", False)),
            sched.get("float_analysis", {}).get("criticality_label", "?"),
            f"{sched.get('cp_impact', {}).get('cp_drag_hours', 0):.0f} h",
            f"{analogs.get('retrieval_summary', {}).get('analog_count', 0)}",
            summ.get("confidence_tier", "?"),
            str(rec.get("analyst_review", {}).get("required", False)),
        ]

    row_names = ["Decision", "Emergence type", "Regulatory",
                 "Criticality", "CP drag", "Analog count",
                 "Confidence", "Analyst review"]
    col_labels = ["Metric"] + [lbl[:14] for lbl in labels]
    table_data = [[rn] + [_fmt(r)[i] for r in results]
                  for i, rn in enumerate(row_names)]

    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                        loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

    decision_row_idx = 1
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == decision_row_idx:
            # colour each scenario column by its decision
            for j, r in enumerate(results):
                if col == j + 1:
                    status = r["recommendation"]["decision_status"]
                    cell.set_facecolor(_STATUS_BG.get(status, "#F1F5F9") + "44")
        elif row % 2 == 0:
            cell.set_facecolor("#F1F5F9")

    fig.tight_layout()
    return fig, (ax_radar, ax_bar, ax_tbl)


# ---------------------------------------------------------------------------
# 9. Stage G — evidence chain traceability
# ---------------------------------------------------------------------------

_EVIDENCE_COLORS: dict[str, str] = {
    "temporal_analysis":  PALETTE["stage_live"],
    "historical_analogs": "#2A9D8F",
    "schedule_analysis":  "#E76F51",
    "component_history":  "#F4A261",
    "regulatory":         PALETTE["escalate"],
}


def plot_evidence_chain(
    result: dict,
    *,
    title: str | None = None,
    ax=None,
) -> tuple:
    """Stage G evidence chain: source-typed items with inline confidence bars.

    Parameters
    ----------
    result : the full dict returned by :func:`run_pipeline` for one scenario
    """
    evidence = result["recommendation"].get("evidence_chain", [])

    if not evidence:
        fig, ax = _fig_ax(ax, (10, 3))
        ax.text(0.5, 0.5, "No evidence chain items.",
                ha="center", va="center", fontsize=11, color="#888")
        ax.axis("off")
        return fig, ax

    fig, ax = _fig_ax(ax, (13, max(3, len(evidence) * 0.8 + 1.5)))
    ax.set_title(title or "Stage G — Evidence Chain Traceability",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(evidence) - 0.5)
    ax.axis("off")

    for i, ev in enumerate(evidence):
        y     = len(evidence) - 1 - i
        src   = ev.get("source_type", "unknown")
        snip  = ev.get("snippet", "")
        conf  = ev.get("confidence", 0)
        color = _EVIDENCE_COLORS.get(src, "#888")

        ax.add_patch(mpatches.FancyBboxPatch(
            (0.1, y - 0.35), 9.8, 0.7, boxstyle="round,pad=0.05",
            facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.5,
        ))
        ax.text(0.25, y, f"[{src.replace('_', ' ').upper()}]",
                fontsize=7.5, va="center", color="white", fontweight="bold",
                bbox=dict(facecolor=color, edgecolor="none", pad=3, boxstyle="round"))
        ax.text(3.2, y, textwrap.shorten(snip, width=90, placeholder="…"),
                fontsize=8, va="center", color="#334155")
        ax.barh(y, conf * 1.5, left=8.3, height=0.35, color=color, alpha=0.7)
        ax.text(9.85, y, f"{conf:.2f}", fontsize=7.5, va="center", color="#334155")

    ax.text(9.85, len(evidence) - 0.05, "conf", fontsize=7, ha="center", color="#888")
    fig.tight_layout()
    return fig, ax
