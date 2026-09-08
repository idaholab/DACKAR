"""
demo_plots.py — Reusable plot functions for the DACKAR v2 pre-outage risk demo.

All functions follow a consistent signature:

    plot_*(stage_output, ..., *, title=None, figsize=None, ax=None) -> (fig, ax_or_axes)

They can be used standalone or composed into composite figures for reports and GUIs.
Every function returns (fig, ax) or (fig, axes) so callers can save or annotate.

Functions
---------
draw_pipeline_architecture_v2   A→B→C→D→E→F→G diagram with new-stage annotation
plot_stage_a_summary            3-panel: dataset sizes / emergence categories / regulatory flags
plot_stage_d_trends             4-panel: CR frequency / trend scores / category heatmap / WO overrun
plot_stage_e_causal_scores      2-panel: causal risk index bars + factor decomposition
plot_stage_f_float_history      Historical CP float consumed by flagged components
plot_risk_register              Pre-outage risk register table with tier colouring + recommended actions
plot_anchor_evidence_chain      Multi-outage evidence chain for the RHS pump anchor scenario
plot_recommendation_card_v2     Full recommendation card for one flagged component
plot_ground_truth_validation    Confusion matrix + predicted score vs actual CP impact
"""
from __future__ import annotations

import textwrap
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Shared colour palette (mirrors the notebook PALETTE dict)
# ---------------------------------------------------------------------------

PALETTE: dict[str, str] = {
    "data_supported":        "#264653",
    "sme_informed":          "#2A9D8F",
    "low_confidence_watch":  "#E9C46A",
    "not_flagged":           "#CBD5E1",
    "escalating":            "#D7263D",
    "moderate":              "#F4A261",
    "stable":                "#06A77D",
    "no_signal":             "#94A3B8",
    "critical_path":         "#E63946",
    "non_critical":          "#A8DADC",
    "observation":           "#A8DADC",
    "degradation":           "#E76F51",
    "stage_new":             "#D7263D",
    "stage_existing":        "#264653",
    "stage_text":            "#FFFFFF",
}

# Risk-tier → human label and foreground colour
_TIER_DISPLAY = {
    "data_supported":        "DATA-SUPPORTED",
    "sme_informed":          "SME-INFORMED",
    "low_confidence_watch":  "WATCH",
    None:                    "—",
}
_TIER_BG = {
    "data_supported":        PALETTE["data_supported"],
    "sme_informed":          PALETTE["sme_informed"],
    "low_confidence_watch":  PALETTE["low_confidence_watch"],
    None:                    "#F8FAFC",
}
_TIER_FG = {
    "data_supported":        "white",
    "sme_informed":          "white",
    "low_confidence_watch":  "#1E293B",
    None:                    "#1E293B",
}

# Recommended actions by risk tier (used in the risk register)
_TIER_ACTIONS: dict[str | None, str] = {
    "data_supported":        "Pre-order parts; allocate contingency crew",
    "sme_informed":          "Stage spare; SME walkdown on day 1",
    "low_confidence_watch":  "Monitor; inspector attention required",
    None:                    "No action required",
}

# Short display names for Millbrook demo components
COMP_NAMES: dict[str, str] = {
    "1RHS-P-001A": "Pump 1A\n(RHS)",
    "1RHS-E-001A": "HX 1A\n(RHS)",
    "1CSP-P-001B": "CSP Pump\n(CSP)",
    "1CCW-P-002A": "CCW Pump\n(CCW)",
    "1RHS-V-001A": "RHS Valve\n(RHS)",
}

COMP_SHORT: dict[str, str] = {k: v.replace("\n", " ") for k, v in COMP_NAMES.items()}


def _fig_ax(ax, figsize: tuple | None) -> tuple:
    """Return (fig, ax): create a new figure when *ax* is None."""
    if ax is not None:
        return ax.get_figure(), ax
    fig, ax = plt.subplots(figsize=figsize or (10, 4.5))
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Pipeline architecture diagram (v2 — pre-outage path)
# ---------------------------------------------------------------------------

def draw_pipeline_architecture_v2(
    *,
    highlight_stage: str | None = None,
    title: str | None = None,
    ax=None,
) -> tuple:
    """Draw the DACKAR v2 A→G pre-outage pipeline diagram.

    Stage D is highlighted as the new ★ stage (temporal trend analysis).

    Parameters
    ----------
    highlight_stage : stage letter to add a gold outline to (e.g. ``"D"``)
    """
    fig, ax = _fig_ax(ax, (15, 3.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title(
        title or "DACKAR v2 — Pre-Outage Risk Prediction Pipeline",
        fontsize=13, fontweight="bold", pad=12,
    )

    stages = [
        ("A", "Ingest &\nNormalize",    "Quality\ngate",       PALETTE["stage_existing"]),
        ("B", "NLP\nExtraction",        "Abbrev +\nNER",       PALETTE["stage_existing"]),
        ("C", "KG\nConstruction",       "In-memory\ngraph",    PALETTE["stage_existing"]),
        ("D", "Temporal\nTrend ★NEW",   "Escalation\nscoring", PALETTE["stage_new"]),
        ("E", "Causal\nChain",          "Causal\nformula",     PALETTE["stage_existing"]),
        ("F", "Schedule\nRisk",         "Float\nhistory",      PALETTE["stage_existing"]),
        ("G", "Recommend\n& Register",  "Tier\nassignment",    PALETTE["stage_existing"]),
    ]

    BOX_W, BOX_H = 1.6, 0.9
    x_positions = [1.0, 3.2, 5.4, 7.6, 9.8, 12.0, 14.2]
    cy = 1.6

    for lbl, sub, _, color in stages:
        pass  # will use enumerate below

    box_positions: dict[str, float] = {}
    for (lbl, sub, subsub, color), xc in zip(stages, x_positions):
        edge_color = "#FFD700" if lbl == highlight_stage else "white"
        edge_lw = 3.0 if lbl == highlight_stage else 1.5
        rect = mpatches.FancyBboxPatch(
            (xc - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.08", linewidth=edge_lw,
            edgecolor=edge_color, facecolor=color, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(xc, cy + 0.12, lbl, ha="center", va="center",
                fontsize=15, fontweight="bold", color="white", zorder=4)
        ax.text(xc, cy - 0.18, sub, ha="center", va="center",
                fontsize=6.8, color="white", zorder=4, linespacing=1.25)
        ax.text(xc, cy - 0.54, subsub, ha="center", va="center",
                fontsize=6.0, color="white", alpha=0.85, zorder=4, linespacing=1.2)
        box_positions[lbl] = xc

    # Arrows between consecutive stages
    for a, b in zip(list("ABCDEFG")[:-1], list("ABCDEFG")[1:]):
        x1 = box_positions[a] + BOX_W / 2 + 0.05
        x2 = box_positions[b] - BOX_W / 2 - 0.05
        ax.annotate(
            "", xy=(x2, cy), xytext=(x1, cy),
            arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5),
        )

    leg = [
        mpatches.Patch(color=PALETTE["stage_existing"], label="Existing stage"),
        mpatches.Patch(color=PALETTE["stage_new"],      label="★ New stage (temporal trend)"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Stage A — data ingestion quality summary
# ---------------------------------------------------------------------------

def plot_stage_a_summary(
    sa: dict,
    components: list[dict],
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """3-panel Stage A summary: dataset sizes / emergence categories / regulatory flags.

    Parameters
    ----------
    sa         : ``results['stage_a']`` dict
    components : component list (from ``demo_data.COMPONENTS``)
    """
    from collections import Counter

    fig, axes = plt.subplots(1, 3, figsize=figsize or (13, 3.5))
    fig.suptitle(title or "Stage A — Data Ingestion & Normalization",
                 fontsize=12, fontweight="bold")

    qs = sa.get("quality_summary", {})

    # Panel 1: record counts
    ax = axes[0]
    counts_d = qs.get("data_record_counts", {})
    labels  = ["Components", "Cond. Reports", "Work Orders", "Activities", "Schedule"]
    counts  = [counts_d.get(k, 0) for k in ("components", "crs", "wos", "activities", "schedule")]
    bars = ax.barh(labels, counts, color=PALETTE["stage_existing"], edgecolor="white", height=0.55)
    ax.set_xlabel("Count")
    ax.set_title("Dataset Sizes")
    for bar, v in zip(bars, counts):
        ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2, str(v), va="center", fontsize=9)

    # Panel 2: emergence categories pie
    ax = axes[1]
    ec_counts      = qs.get("emergence_category_counts", {})
    total_act      = counts_d.get("activities", 0)
    emergent_total = sum(ec_counts.values())
    planned_count  = total_act - emergent_total
    pie_labels     = list(ec_counts.keys()) + ["Planned"]
    pie_values     = list(ec_counts.values()) + [planned_count]
    pie_colors     = [PALETTE["degradation"]] * len(ec_counts) + [PALETTE["not_flagged"]]
    ax.pie(pie_values, labels=pie_labels, colors=pie_colors,
           autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
    ax.set_title("Emergence Categories")

    # Panel 3: regulatory flag count
    ax = axes[2]
    reg_true  = sum(1 for c in components if c.get("regulatory_constraint_flag"))
    reg_false = len(components) - reg_true
    ax.barh(["Regulated", "Not regulated"], [reg_true, reg_false],
            color=[PALETTE["escalating"], PALETTE["not_flagged"]], edgecolor="white", height=0.5)
    ax.set_xlabel("Component count")
    ax.set_title("Regulatory Constraint Flags")
    for i, v in enumerate([reg_true, reg_false]):
        ax.text(v + 0.05, i, str(v), va="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 3. Stage D — temporal trend analysis (4-panel)
# ---------------------------------------------------------------------------

def plot_stage_d_trends(
    sd: dict,
    comp_ids: list[str] | None = None,
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """4-panel Stage D temporal trend visualisation.

    Panels: (a) Degradation CR frequency by cycle  (b) Composite trend scores
            (c) CR category profile heatmap         (d) WO duration overrun trend

    Parameters
    ----------
    sd       : ``results['stage_d']`` dict — keyed by component_id
    comp_ids : list of component IDs to include (default: all keys in sd)
    """
    comp_ids = comp_ids or list(sd.keys())
    cycles   = ["RF-20 prep", "RF-21 prep", "RF-22 prep"]

    fig = plt.figure(figsize=figsize or (15, 8))
    fig.suptitle(
        title or "Stage D — Temporal Trend Analysis\n"
        "(★ New stage — detects escalating patterns before emergent work occurs)",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel (0,0): Degradation CR counts per cycle
    ax1 = fig.add_subplot(gs[0, 0])
    bar_width   = 0.22
    cycle_alphas = [0.5, 0.75, 1.0]
    x_pos = np.arange(len(comp_ids))
    for ci, (cycle, alpha) in enumerate(zip(cycles, cycle_alphas)):
        counts = [sd[cid].get("deg_counts_by_cycle", {}).get(cycle, 0)
                  for cid in comp_ids]
        ax1.bar(x_pos + ci * bar_width, counts, bar_width,
                label=cycle, color=PALETTE["degradation"], alpha=alpha, edgecolor="white")
    ax1.set_xticks(x_pos + bar_width)
    ax1.set_xticklabels(
        [COMP_NAMES.get(c, c).replace("\n", " ") for c in comp_ids], fontsize=7.5
    )
    ax1.set_ylabel("Degradation CR count")
    ax1.set_title("Degradation CR Frequency by Cycle")
    ax1.legend(fontsize=7.5)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Panel (0,1): Composite trend score bars
    ax2 = fig.add_subplot(gs[0, 1])
    trend_scores = [sd[cid].get("trend_score", 0) for cid in comp_ids]
    trend_labels = [sd[cid].get("trend_label", "no_signal") for cid in comp_ids]
    bar_colors   = [PALETTE.get(lbl, PALETTE["no_signal"]) for lbl in trend_labels]
    order = sorted(range(len(comp_ids)), key=lambda i: -trend_scores[i])
    sorted_names  = [COMP_NAMES.get(comp_ids[i], comp_ids[i]).replace("\n", " ") for i in order]
    sorted_scores = [trend_scores[i] for i in order]
    sorted_colors = [bar_colors[i] for i in order]
    sorted_tlabels = [trend_labels[i] for i in order]

    bars = ax2.barh(sorted_names, sorted_scores, color=sorted_colors, edgecolor="white", height=0.55)
    ax2.axvline(0.5, color="#334155", lw=1, linestyle="--", label="escalating ≥0.5")
    ax2.axvline(0.2, color="#94A3B8", lw=1, linestyle=":",  label="moderate ≥0.2")
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("Trend score")
    ax2.set_title("Composite Trend Score")
    ax2.legend(fontsize=7.5)
    for bar, score, lbl in zip(bars, sorted_scores, sorted_tlabels):
        ax2.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{score:.1f}  [{lbl}]", va="center", fontsize=7.5)

    # Panel (1,0): Category profile heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, len(cycles))
    ax3.set_ylim(0, len(comp_ids))
    ax3.axis("off")
    ax3.set_title("CR Category Profile by Cycle")

    for ci, cycle in enumerate(cycles):
        ax3.text(ci + 0.5, len(comp_ids) + 0.15, cycle.replace(" prep", ""),
                 ha="center", fontsize=8, fontweight="bold", color="#334155")
    for ri, cid in enumerate(reversed(comp_ids)):
        prof       = sd.get(cid, {})
        cyc_detail = prof.get("cycle_detail", {})
        ax3.text(-0.05, ri + 0.5, COMP_NAMES.get(cid, cid).replace("\n", " "),
                 ha="right", va="center", fontsize=7.5)
        for ci, cycle in enumerate(cycles):
            det   = cyc_detail.get(cycle, {})
            deg   = det.get("degradation_crs", 0)
            obs   = det.get("observation_crs", 0)
            total = deg + obs
            if total == 0:
                cell_color, cell_text = "#F8FAFC", "—"
            elif deg == 0:
                cell_color, cell_text = PALETTE["observation"], "Obs"
            elif obs == 0:
                cell_color, cell_text = PALETTE["degradation"], "Deg"
            else:
                cell_color, cell_text = "#FEF3C7", "Mixed"
            rect = mpatches.FancyBboxPatch(
                (ci + 0.05, ri + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.03",
                facecolor=cell_color, edgecolor="#CBD5E1", lw=0.8,
            )
            ax3.add_patch(rect)
            ax3.text(ci + 0.5, ri + 0.5, cell_text, ha="center", va="center", fontsize=8)
        if prof.get("category_escalation"):
            ax3.text(len(cycles) + 0.05, ri + 0.5, "← escalation",
                     va="center", fontsize=7, color=PALETTE["escalating"], fontweight="bold")

    # Panel (1,1): WO duration overrun trend
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axhline(1.0, color="#94A3B8", lw=1.2, linestyle="--", label="on-plan ratio = 1.0")
    plotted_any = False
    colours_by_comp = [PALETTE["escalating"], PALETTE["sme_informed"], PALETTE["moderate"]]
    for cid, col in zip(comp_ids[:3], colours_by_comp):
        ratios = sd.get(cid, {}).get("overrun_ratios", [])
        if ratios:
            x_vals = list(range(1, len(ratios) + 1))
            ax4.plot(x_vals, ratios, "o-", color=col, lw=1.8, ms=7,
                     label=COMP_SHORT.get(cid, cid))
            plotted_any = True
    if plotted_any:
        ax4.legend(fontsize=7.5)
    ax4.set_xlabel("WO sequence (chronological)")
    ax4.set_ylabel("Actual / planned duration")
    ax4.set_title("WO Duration Overrun Trend")

    fig.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)


# ---------------------------------------------------------------------------
# 4. Stage E — causal chain scores (2-panel)
# ---------------------------------------------------------------------------

def plot_stage_e_causal_scores(
    se: dict,
    sg: dict,
    comp_ids: list[str] | None = None,
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """2-panel Stage E visualisation: causal index + factor decomposition.

    Parameters
    ----------
    se       : ``results['stage_e']`` dict — keyed by component_id
    sg       : ``results['stage_g']`` dict (for tier assignments)
    comp_ids : component IDs to include (default: all keys in se)
    """
    comp_ids = comp_ids or list(se.keys())
    _TIER_COLORS = {
        "data_supported": PALETTE["data_supported"],
        "sme_informed":   PALETTE["sme_informed"],
        None:             PALETTE["not_flagged"],
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize or (13, 4))
    fig.suptitle(title or "Stage E — Causal Chain Scores", fontsize=12, fontweight="bold")

    # Panel 1: causal risk index
    ax = axes[0]
    scores = [se[cid].get("causal_score", 0) for cid in comp_ids]
    tiers  = [sg.get("recommendations", {}).get(cid, {}).get("confidence_tier")
              for cid in comp_ids]
    colors = [_TIER_COLORS.get(t, PALETTE["not_flagged"]) for t in tiers]
    bars = ax.barh(
        [COMP_SHORT.get(c, c) for c in comp_ids], scores,
        color=colors, edgecolor="white", height=0.55,
    )
    ax.axvline(2.0, color="#94A3B8", lw=1, linestyle=":", label="max possible (2.0)")
    ax.axvline(1.5, color=PALETTE["data_supported"], lw=1, linestyle="--",
               label="DATA-SUPPORTED ≥1.5")
    ax.set_xlabel("Causal risk index")
    ax.set_title("Causal Chain Risk Index")
    ax.legend(fontsize=7.5)
    for bar, score in zip(bars, scores):
        ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=8.5)

    # Panel 2: factor decomposition
    ax = axes[1]
    bar_w = 0.22
    x_pos = np.arange(len(comp_ids))
    factors = [
        ("deg_cr_frac",                  "Deg CR fraction",       0.5),
        ("emergent_frac",                "Emergent fraction",     0.75),
        ("criticality_weight_contrib",   "Criticality factor",    1.0),
    ]
    for fi, (factor_key, label, alpha) in enumerate(factors):
        vals = []
        for cid in comp_ids:
            d    = se.get(cid, {})
            n_tr = d.get("n_training_outages", 2)
            n_deg = d.get("n_outages_with_degradation_cr", 0)
            n_em  = d.get("n_outages_with_emergent_activity", 0)
            cw    = d.get("criticality_weight", 1.0)
            if factor_key == "deg_cr_frac":
                vals.append(n_deg / n_tr if n_tr else 0)
            elif factor_key == "emergent_frac":
                vals.append(n_em / n_deg if n_deg else 0)
            else:
                vals.append(min(1.0, (cw - 1.0)))
        ax.bar(x_pos + fi * bar_w, vals, bar_w, label=label,
               color=PALETTE["stage_existing"], alpha=alpha, edgecolor="white")

    ax.set_xticks(x_pos + bar_w)
    ax.set_xticklabels([COMP_SHORT.get(c, c) for c in comp_ids], fontsize=8)
    ax.set_ylabel("Factor value (0–1)")
    ax.set_title("Score Factor Decomposition")
    ax.legend(fontsize=7.5)
    ax.set_ylim(0, 1.3)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 5. Stage F — historical CP float consumed
# ---------------------------------------------------------------------------

def plot_stage_f_float_history(
    sf: dict,
    flagged_ids: list[str] | None = None,
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """Stage F: historical critical-path float consumed by each flagged component.

    Parameters
    ----------
    sf          : ``results['stage_f']`` dict — keyed by component_id
    flagged_ids : component IDs to show (default: all keys in sf)
    """
    flagged_ids = flagged_ids or list(sf.keys())
    fig, ax = _fig_ax(None, figsize or (10, 4))
    fig.suptitle(
        title or "Stage F — Historical Critical-Path Float Consumed by Emergent Work",
        fontsize=11, fontweight="bold",
    )

    bar_width = 0.3
    for i, cid in enumerate(flagged_ids):
        fdata   = sf.get(cid, {})
        impacts = fdata.get("historical_cp_impacts", [])
        if impacts:
            for j, imp in enumerate(impacts):
                hrs = imp.get("float_consumed_hrs", 0)
                color = PALETTE["critical_path"] if hrs > 0 else PALETTE["not_flagged"]
                ax.bar(i + (j - 0.5) * bar_width, hrs, bar_width * 0.9,
                       color=color, edgecolor="white")
                if hrs > 0:
                    ax.text(i + (j - 0.5) * bar_width, hrs + 0.3,
                            f"{hrs:.0f}h", ha="center", fontsize=8.5, fontweight="bold")
            mean_val = fdata.get("mean_cp_float_consumed", 0) or 0
            ax.hlines(mean_val, i - 0.4, i + 0.4, color="#1E293B", lw=1.5, linestyle="--")
            ax.text(i + 0.42, mean_val, f"mean\n{mean_val:.0f}h",
                    va="center", fontsize=7.5, color="#1E293B")
        else:
            ax.bar(i, 0, 0.5, color=PALETTE["not_flagged"], edgecolor="#CBD5E1")
            ax.text(i, 0.3, "no historical\ndata", ha="center", va="bottom",
                    fontsize=8, color="#94A3B8")

    ax.set_xticks(range(len(flagged_ids)))
    ax.set_xticklabels(
        [f"{cid}\n{COMP_SHORT.get(cid, '')}" for cid in flagged_ids], fontsize=9
    )
    ax.set_ylabel("Critical-path float consumed (hours)")
    ax.set_title("Historical CP Impact per Training Outage")
    ax.legend(
        handles=[
            mpatches.Patch(color=PALETTE["critical_path"], label="RF-20"),
            mpatches.Patch(color=PALETTE["critical_path"], alpha=0.6, label="RF-21"),
        ],
        fontsize=8,
    )

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. Stage G — pre-outage risk register table
# ---------------------------------------------------------------------------

def plot_risk_register(
    sg: dict,
    sd: dict,
    se: dict,
    components: list[dict],
    *,
    outage_label: str = "RF-22",
    plant_label: str = "Millbrook Nuclear Station Unit 1",
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """Pre-outage risk register table with tier colouring and recommended actions.

    Extends the base register with a **Recommended Action** column derived from
    the risk tier — this is the column that matters most to outage managers.

    Parameters
    ----------
    sg         : ``results['stage_g']`` dict
    sd         : ``results['stage_d']`` dict (for trend labels)
    se         : ``results['stage_e']`` dict (for causal scores)
    components : component list (from ``demo_data.COMPONENTS``)
    """
    register = sg.get("risk_register", [])
    fig, ax = _fig_ax(None, figsize or (16, max(3.5, len(register) * 0.85 + 1.2)))
    ax.axis("off")
    ax.set_title(
        title or f"Pre-Outage Risk Register — {outage_label}  ({plant_label})",
        fontsize=12, fontweight="bold", pad=10,
    )

    col_labels = ["Rank", "Component ID", "Description", "System",
                  "Tier", "Trend", "Reg ★", "Score", "Recommended Action"]
    col_widths  = [0.03,   0.10,           0.20,          0.15,
                   0.12,   0.07,    0.03,  0.05,   0.25]

    row_h       = 0.13
    header_y    = 0.90
    data_y_start = header_y - row_h

    # Header
    x = 0.01
    for label, w in zip(col_labels, col_widths):
        ax.text(x + w / 2, header_y, label, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#334155", edgecolor="none"))
        x += w

    for rank, row in enumerate(register):
        cid      = row.get("component_id", "")
        tier     = row.get("confidence_tier")
        comp     = next((c for c in components if c["component_id"] == cid), {})
        trend_lbl = sd.get(cid, {}).get("trend_label", "—")
        score_val = se.get(cid, {}).get("causal_score", 0.0)
        reg_flag  = "★" if comp.get("regulatory_constraint_flag") else ""
        desc      = comp.get("description", "")
        system    = comp.get("system", "")
        tier_note = " (T)" if row.get("tier_reason") == "escalating_trend_no_emergent_precedent" else ""
        action    = _TIER_ACTIONS.get(tier, _TIER_ACTIONS[None])

        row_y = data_y_start - rank * row_h
        bg    = _TIER_BG.get(tier, "#F8FAFC")
        fg    = _TIER_FG.get(tier, "#1E293B")

        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, row_y - row_h * 0.45), 0.98, row_h * 0.9,
            boxstyle="round,pad=0.01", facecolor=bg, edgecolor="white", lw=0.8, alpha=0.85,
        ))

        values = [str(rank + 1), cid, desc[:28], system[:20],
                  (_TIER_DISPLAY.get(tier, "—") + tier_note)[:20],
                  trend_lbl[:12], reg_flag, f"{score_val:.2f}", action]
        x = 0.01
        for val, w in zip(values, col_widths):
            ax.text(x + w / 2, row_y, val, ha="center", va="center",
                    fontsize=6 if len(val) > 30 else (7 if len(val) > 18 else 8),
                    color=fg)
            x += w

    ax.text(0.5, -0.05, "Millbrook synthetic holdout evaluation",
            ha="center", fontsize=8, color="#94A3B8", style="italic",
            transform=ax.transAxes)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 7. Evidence chain — anchor scenario (1RHS-P-001A across RF-20/21/22)
# ---------------------------------------------------------------------------

def plot_anchor_evidence_chain(
    *,
    comp_id: str = "1RHS-P-001A",
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """Multi-outage evidence chain for the RHS pump anchor scenario.

    This is an illustrated diagram showing the pattern of CRs, WOs, planned
    activities, and emergent work across RF-20 (training), RF-21 (training),
    and RF-22 (holdout prediction).  The chain data is hardcoded for the
    Millbrook synthetic demo — it is not drawn from stage output dicts.

    Parameters
    ----------
    comp_id : component ID label shown in the title (default ``"1RHS-P-001A"``)
    """
    fig, ax = _fig_ax(None, figsize or (14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title(
        title or f"Evidence Chain — {comp_id}  (DATA-SUPPORTED)",
        fontsize=12, fontweight="bold",
    )

    # Lane layout
    lanes_x = [0.4, 4.9, 9.4]
    lane_labels = ["RF-20 (training)", "RF-21 (training)", "RF-22 (holdout — prediction)"]
    lane_colors = ["#F0FDF4", "#F0FDF4", "#EFF6FF"]
    lane_border = ["#86EFAC", "#86EFAC", "#93C5FD"]
    lane_w = 4.0

    for lx, ll, lc, lb in zip(lanes_x, lane_labels, lane_colors, lane_border):
        ax.add_patch(mpatches.FancyBboxPatch(
            (lx - 0.1, 0.1), lane_w, 5.1,
            boxstyle="round,pad=0.05", facecolor=lc, edgecolor=lb, lw=1.5, alpha=0.5,
        ))
        ax.text(lx + lane_w / 2 - 0.1, 5.0, ll, ha="center", fontsize=9,
                fontweight="bold", color="#1E293B")

    _COLORS_MAP = {
        "cr_obs":   ("#DBEAFE", "#3B82F6"),
        "cr_deg":   ("#FEF3C7", "#F59E0B"),
        "wo":       ("#EDE9FE", "#7C3AED"),
        "planned":  ("#F1F5F9", "#94A3B8"),
        "emergent": ("#FEE2E2", "#DC2626"),
        "predict":  ("#D1FAE5", "#059669"),
    }

    def _box(x, y, w, h, label, btype):
        fc, ec = _COLORS_MAP.get(btype, ("#F8FAFC", "#CBD5E1"))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06", facecolor=fc, edgecolor=ec, lw=1.5,
        ))
        lines = label.split("\n")
        for li, line in enumerate(lines):
            ax.text(x + w / 2, y + h - (li + 0.6) * h / len(lines),
                    line, ha="center", va="center",
                    fontsize=7,
                    fontweight="bold" if btype == "emergent" else "normal",
                    color="#DC2626" if btype == "emergent" else "#1E293B")

    def _arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.2))

    bw, bh = 3.6, 0.65

    # RF-20 lane
    _box(0.5, 4.0, bw, bh, "CR-2019-04412\nvibration above baseline\n(observation)", "cr_obs")
    _box(0.5, 3.1, bw, bh, "CR-2019-06891\nslight leakage at mech seal\n(degradation)", "cr_deg")
    _box(0.5, 2.2, bw, bh, "WO-2019-52341\nSeal Insp  8h planned / 9.5h actual", "wo")
    _box(0.5, 1.3, bw, bh, "RF20-MECH-0042  Seal Insp (planned)", "planned")
    _box(0.5, 0.3, bw, bh, "RF20-MECH-0089  EMERGENT\nSeal Face Repl — 16h ON CP", "emergent")
    for ya, yb in [(4.0, 3.75), (3.1, 2.85), (2.2, 1.95), (1.3, 0.95), (1.3, 0.55)]:
        _arrow(0.5 + bw / 2, ya, 0.5 + bw / 2, yb)

    # RF-21 lane
    _box(5.0, 4.0, bw, bh, "CR-2021-00892 + CR-2021-02234\n+ CR-2021-07743\n(degradation — 3 CRs)", "cr_deg")
    _box(5.0, 2.9, bw, bh, "WO-2021-38471\nSeal+Align  16h planned / 24h actual", "wo")
    _box(5.0, 1.9, bw, bh, "RF21-MECH-0038  Seal Repl (planned)", "planned")
    _box(5.0, 0.9, bw, bh, "RF21-MECH-0079  EMERGENT\nImpeller Insp — 12h ON CP", "emergent")
    for ya, yb in [(4.0, 3.65), (2.9, 2.55), (1.9, 2.20), (1.9, 1.55), (1.9, 1.25)]:
        _arrow(5.0 + bw / 2, ya, 5.0 + bw / 2, yb)

    # RF-22 prediction lane
    _box(9.5, 4.0, bw, bh, "CR-2022-01142 + CR-2022-03387\nvib still elevated post RF-21 seal repl\n(observation + degradation)", "cr_deg")
    _box(9.5, 2.9, bw, bh, "WO-2022-31102\nEnhanced Insp  20h planned", "wo")
    _box(9.5, 1.9, bw, bh, "RF22-MECH-0041  Enhanced Insp (planned)", "planned")
    _box(9.5, 0.65, bw, 0.85, "PREDICTED: bearing + impeller scope\nPre-stage replacement parts\n→ Ground truth: 20h CP consumed", "predict")
    for ya, yb in [(4.0, 3.65), (2.9, 2.55), (1.9, 2.20), (1.9, 1.5)]:
        _arrow(9.5 + bw / 2, ya, 9.5 + bw / 2, yb)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 8. Stage G — recommendation card (per component)
# ---------------------------------------------------------------------------

def plot_recommendation_card_v2(
    comp_id: str,
    sg: dict,
    sf: dict,
    sd: dict,
    se: dict,
    components: list[dict],
    *,
    run_id: str = "",
    title: str | None = None,
) -> tuple:
    """Render a Stage G recommendation as a matplotlib card figure.

    Parameters
    ----------
    comp_id    : component ID to render
    sg / sf / sd / se : stage output dicts from :func:`run_pipeline`
    components : component metadata list (from ``demo_data.COMPONENTS``)
    run_id     : pipeline run ID (shown in footer)
    """
    rec   = sg.get("recommendations", {}).get(comp_id, {})
    fdata = sf.get(comp_id, {})
    tdata = sd.get(comp_id, {})
    tier  = rec.get("confidence_tier")
    comp  = next((c for c in components if c["component_id"] == comp_id), {})

    _TIER_COLORS_CARD = {
        "data_supported": PALETTE["data_supported"],
        "sme_informed":   PALETTE["sme_informed"],
        None:             "#94A3B8",
    }
    _TIER_LABELS_CARD = {
        "data_supported": "DATA-SUPPORTED",
        "sme_informed":   "SME-INFORMED",
        None:             "NOT FLAGGED",
    }

    bg = _TIER_COLORS_CARD.get(tier, "#888")
    fig, ax = _fig_ax(None, (12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Header band
    ax.add_patch(mpatches.FancyBboxPatch((0, 4.9), 12, 1.1, boxstyle="square",
                                          facecolor=bg, edgecolor="none"))
    ax.text(6, 5.65, f"{comp_id} — {comp.get('description', '')}",
            ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.text(6, 5.15, _TIER_LABELS_CARD.get(tier, "—"),
            ha="center", va="center", fontsize=14, color="white", fontweight="bold")

    # Body
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 12, 4.9, boxstyle="square",
                                          facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1))

    # Regulatory badge
    if comp.get("regulatory_constraint_flag"):
        ax.add_patch(mpatches.FancyBboxPatch((0.2, 4.55), 3.0, 0.32,
                                              boxstyle="round,pad=0.05",
                                              facecolor=PALETTE["escalating"], edgecolor="none"))
        ax.text(1.7, 4.71, f"★ REGULATORY — {comp.get('notes','')[:40]}",
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    cat = rec.get("category", "—") or "—"
    ax.text(11.8, 4.71, f"Category: {cat}", ha="right", va="center",
            fontsize=8, color="white", fontweight="bold")

    # Finding
    ax.text(0.3, 4.35, "Finding:", fontsize=9, fontweight="bold", color="#334155")
    for i, line in enumerate(textwrap.wrap(rec.get("finding", "N/A"), 90)[:4]):
        ax.text(0.3, 4.05 - i * 0.32, line, fontsize=8, color="#475569")

    # Recommendation
    ax.text(0.3, 2.85, "Recommendation:", fontsize=9, fontweight="bold", color="#334155")
    for i, line in enumerate(textwrap.wrap(rec.get("recommendation", "N/A"), 90)[:3]):
        ax.text(0.3, 2.55 - i * 0.30, line, fontsize=8, color="#475569")

    # Evidence chain
    ax.text(0.3, 1.65, "Evidence chain:", fontsize=9, fontweight="bold", color="#334155")
    for i, ev in enumerate(rec.get("evidence_chain", [])[:4]):
        snippet = (f"[{ev.get('record_type','?')}] {ev.get('record_id','')} — "
                   f"{ev.get('description','')[:55]}")
        ax.text(0.45, 1.38 - i * 0.28, snippet, fontsize=7.5, color="#475569")

    # Key metrics row
    mean_cp = fdata.get("mean_cp_float_consumed", 0) or 0
    trend   = tdata.get("trend_label", "—")
    causal  = se.get(comp_id, {}).get("causal_score", 0)
    metrics = [
        ("Causal score",   f"{causal:.2f}"),
        ("Trend",          trend),
        ("Mean CP impact", f"{mean_cp:.0f} h" if mean_cp else "no data"),
        ("Reg. flag",      "YES" if comp.get("regulatory_constraint_flag") else "NO"),
    ]
    for i, (k, v) in enumerate(metrics):
        xoff = 0.3 + i * 2.9
        ax.add_patch(mpatches.FancyBboxPatch((xoff, 0.05), 2.5, 0.55,
                                              boxstyle="round,pad=0.04",
                                              facecolor=bg, edgecolor="none", alpha=0.15))
        ax.text(xoff + 1.25, 0.48, k, ha="center", fontsize=7.5, color="#475569")
        ax.text(xoff + 1.25, 0.22, v, ha="center", fontsize=9, fontweight="bold",
                color="#1E293B")

    # Accept/Reject widgets
    ax.text(9.0, 1.55, "✓  Accept", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#D1FAE5", edgecolor="#059669"))
    ax.text(11.0, 1.55, "✗  Reject — reason: ___", ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEE2E2", edgecolor="#DC2626"))

    if run_id:
        ax.text(11.8, -0.08, f"Run ID: {run_id}   |   Synthetic illustrative dataset",
                ha="right", fontsize=6.5, color="#94A3B8", transform=ax.transAxes)

    ax.set_title(title or "", fontsize=10)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 9. Ground truth validation (RF-22 holdout)
# ---------------------------------------------------------------------------

def plot_ground_truth_validation(
    gt: dict,
    sg: dict,
    se: dict,
    rf22_ground_truth: list[dict],
    *,
    title: str | None = None,
    figsize: tuple | None = None,
) -> tuple:
    """Confusion matrix + predicted score vs actual CP hours for RF-22 holdout.

    Parameters
    ----------
    gt                 : ``results['ground_truth_comparison']`` dict
    sg                 : ``results['stage_g']`` dict
    se                 : ``results['stage_e']`` dict
    rf22_ground_truth  : ground truth records (from ``demo_data.RF22_GROUND_TRUTH``)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize or (12, 4))
    fig.suptitle(title or "Stage G — Prediction vs. Actual (RF-22 Holdout)",
                 fontsize=12, fontweight="bold")

    # Panel 1: confusion matrix
    ax = axes[0]
    tp = len(gt.get("true_positives", []))
    fn = len(gt.get("false_negatives", []))
    tn = len(gt.get("true_negatives_confirmed", []))
    fp = 0
    matrix = [[tp, fn], [fp, tn]]
    cell_labels = [
        ["True Pos\n(flagged + emergent)", "False Neg\n(missed)"],
        ["False Pos\n(over-flagged)",       "True Neg\n(correctly clear)"],
    ]
    cm_colors = [["#D1FAE5", "#FEE2E2"], ["#FEF3C7", "#DBEAFE"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(mpatches.FancyBboxPatch(
                (j, 1 - i), 1, 1, boxstyle="square",
                facecolor=cm_colors[i][j], edgecolor="white", lw=2,
            ))
            ax.text(j + 0.5, 1.5 - i, str(matrix[i][j]),
                    ha="center", va="center", fontsize=24, fontweight="bold", color="#1E293B")
            ax.text(j + 0.5, 1.15 - i, cell_labels[i][j],
                    ha="center", va="center", fontsize=7.5, color="#475569")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted:\nFlagged", "Predicted:\nNot Flagged"], fontsize=8)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual:\nNot Emergent", "Actual:\nEmergent"], fontsize=8)
    ax.set_title("Confusion Matrix")

    # Panel 2: actual CP hours vs predicted score
    ax = axes[1]
    tp_ids    = gt.get("true_positives", [])
    gt_records = {r["component_id"]: r for r in rf22_ground_truth}
    bar_w = 0.35
    for i, cid in enumerate(tp_ids):
        tier = sg.get("recommendations", {}).get(cid, {}).get("confidence_tier")
        tc   = _TIER_BG.get(tier, PALETTE["not_flagged"])
        gt_r = gt_records.get(cid, {})
        actual_hrs = gt_r.get("actual_duration_hrs", 0) or 0
        causal     = se.get(cid, {}).get("causal_score", 0) or 0
        ax.bar(i - bar_w / 2, actual_hrs, bar_w, color=PALETTE["critical_path"],
               edgecolor="white",
               label="Actual CP hours (RF-22)" if i == 0 else "")
        ax.bar(i + bar_w / 2, causal, bar_w, color=tc, edgecolor="white", alpha=0.8,
               label=f"{_TIER_DISPLAY.get(tier, '?')} score" if i == 0 else "")
        ax.text(i - bar_w / 2, actual_hrs + 0.3, f"{actual_hrs:.0f}h",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(tp_ids)))
    ax.set_xticklabels([COMP_SHORT.get(c, c) for c in tp_ids], fontsize=9)
    ax.set_ylabel("Hours / score")
    ax.set_title("Actual CP Impact vs Predicted Score")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, axes
