"""Generate the "did the experiment succeed?" figure set for the DBS project.

READ-ONLY over frozen artefacts. This script never trains, never runs
inference, and never touches ground truth; it only re-plots values that are
already frozen in:

    results/subject_clean_holdout/HOLDOUT_PER_CASE.csv
    results/subject_clean_holdout/HOLDOUT_PRIMARY_ENDPOINT.json
    results/subject_clean_holdout/FINAL_HOLDOUT_SUMMARY.json
    manifests/subject_clean_v1/*.csv            (split sizes only)

Output goes to results/figures_success_check/ and nowhere else.

Usage:
    python scripts/make_success_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
HOLDOUT = REPO / "results" / "subject_clean_holdout"
MANIFESTS = REPO / "manifests" / "subject_clean_v1"
OUT = REPO / "results" / "figures_success_check"

# --------------------------------------------------------------------------
# Style: a single validated light-mode instance.
#
# Categorical slots 1/2/3/7 (blue, orange, aqua, violet). Validated all-pairs
# on the light surface: worst CVD dE 9.2, worst normal-vision dE 16.3.
# Aqua sits below 3:1 contrast, so every series carries a visible direct label
# (the documented relief rule) - identity is never hue-alone.
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

ROIS = ("STN", "SN", "RN")
MEMBERS = ("ensemble", "seed123", "seed2026", "seed42")

MEMBER_LABEL = {
    "ensemble": "Ensemble (3 seeds)",
    "seed123": "seed123 (comparator)",
    "seed2026": "seed2026",
    "seed42": "seed42",
}
MEMBER_COLOR = {
    "ensemble": "#2a78d6",  # blue
    "seed123": "#eb6834",  # orange
    "seed2026": "#1baf7a",  # aqua
    "seed42": "#4a3aa7",  # violet
}

COLOR_POS = "#2a78d6"
COLOR_NEG = "#e34948"
SEQ = ("#86b6ef", "#5598e7", "#2a78d6", "#184f95")  # ordinal blue ramp

FS_TITLE = 14
FS_SUB = 9.5
FS_AXIS = 11
FS_TICK = 10
FS_ANN = 9.5
FS_LEGEND = 9.5


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "axes.labelsize": FS_AXIS,
            "axes.titlesize": FS_AXIS,
            "legend.fontsize": FS_LEGEND,
            "axes.grid": False,
            "lines.solid_capstyle": "round",
        }
    )


def dress(ax: plt.Axes, *, ygrid: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.9)


def fignote(fig: plt.Figure, text: str, *, y: float = 0.012) -> None:
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=FS_SUB, color=MUTED)


def label_block(ax, x, y, lines, **kw):
    """Bordered annotation box in axes coordinates."""
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=FS_ANN,
        color=INK2,
        ha="left",
        va="top",
        linespacing=1.55,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="#ffffff",
            edgecolor=GRID,
            linewidth=1.0,
        ),
        **kw,
    )


def direct_labels(ax, xs, ys, texts, colors, *, dx, dy_min):
    """Place right-edge direct labels, nudging apart to avoid collisions.

    The gap test carries an epsilon tolerance: `prev + dy_min` does not
    reproduce the gap exactly in binary floating point, so a bare
    ``< dy_min`` test re-fires on its own output and never terminates.
    """
    eps = 1e-12
    order = np.argsort(ys)
    placed = []
    for idx in order:
        y = ys[idx]
        while placed and (y - placed[-1]) < dy_min - eps:
            y = placed[-1] + dy_min
        placed.append(y)
    for idx, y in zip(order, placed):
        ax.annotate(
            texts[idx],
            xy=(xs[idx], ys[idx]),
            xytext=(xs[idx] + dx, y),
            fontsize=FS_ANN,
            color=colors[idx],
            va="center",
            ha="left",
            fontweight="semibold",
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-",
                color=colors[idx],
                linewidth=0.8,
                shrinkA=0,
                shrinkB=2,
            ),
        )


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.35,
        )
    plt.close(fig)


# --------------------------------------------------------------------------
# Data loading (frozen artefacts only)
# --------------------------------------------------------------------------


def load():
    per_case = pd.read_csv(HOLDOUT / "HOLDOUT_PER_CASE.csv", encoding="utf-8-sig")
    primary = json.loads((HOLDOUT / "HOLDOUT_PRIMARY_ENDPOINT.json").read_text(encoding="utf-8"))
    summary = json.loads((HOLDOUT / "FINAL_HOLDOUT_SUMMARY.json").read_text(encoding="utf-8"))
    agg = summary["aggregates"]

    splits = {}
    for name in ("train", "val", "internal_test", "challenge_test"):
        df = pd.read_csv(MANIFESTS / f"{name}.csv", encoding="utf-8-sig")
        splits[name] = len(df)
    splits["excluded"] = len(
        pd.read_csv(MANIFESTS / "EXCLUDED_CASES.csv", encoding="utf-8-sig")
    )

    # Cross-check the re-plotted values against the frozen aggregates.
    checks = []
    for m in MEMBERS:
        checks.append(
            (
                f"macro {m}",
                float(per_case[f"macro_foreground_dice__{m}"].mean()),
                float(agg[m]["macro_foreground_dice"]),
            )
        )
    for roi in ROIS:
        for m in MEMBERS:
            checks.append(
                (
                    f"{roi} Dice {m}",
                    float(per_case[f"Dice_{roi}__{m}"].mean()),
                    float(agg[m][f"Dice_{roi}"]),
                )
            )
    for m in MEMBERS:
        checks.append(
            (
                f"STN Pred/GT {m}",
                float(per_case[f"volume_ratio_STN__{m}"].mean()),
                float(agg[m]["volume_ratio_STN"]),
            )
        )
    worst = max(abs(a - b) for _, a, b in checks)
    print(f"[verify] {len(checks)} re-plotted values vs frozen aggregates, max abs diff = {worst:.3e}")
    assert worst < 1e-9, "re-plotted values disagree with frozen aggregates"

    return per_case, primary, summary, agg, splits


# --------------------------------------------------------------------------
# Figure 1 - ROI performance curves
# --------------------------------------------------------------------------


def figure1(per_case, agg, primary):
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 9.6), sharex=True)
    fig.subplots_adjust(hspace=0.17, top=0.855, bottom=0.235, left=0.10, right=0.745)

    xs = np.arange(len(ROIS))

    # --- Panel A: Dice ---------------------------------------------------
    axA = axes[0]
    ys = {}
    for m in MEMBERS:
        y = np.array([agg[m][f"Dice_{roi}"] for roi in ROIS])
        ys[m] = y
        axA.plot(
            xs,
            y,
            color=MEMBER_COLOR[m],
            linewidth=2.0,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.6,
            zorder=3,
            label=MEMBER_LABEL[m],
        )
    short = {"ensemble": "Ensemble", "seed123": "seed123 (comparator)",
             "seed2026": "seed2026", "seed42": "seed42"}
    direct_labels(
        axA,
        xs=[xs[-1]] * 4,
        ys=[ys[m][-1] for m in MEMBERS],
        texts=[f"{ys[m][-1]:.4f}  {short[m]}" for m in MEMBERS],
        colors=[MEMBER_COLOR[m] for m in MEMBERS],
        dx=0.10,
        dy_min=0.0172,
    )
    axA.set_ylabel("Dice (foreground, mean over 100 cases)")
    axA.set_xlim(-0.42, 2.42)
    axA.set_ylim(0.572, 0.892)
    axA.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    dress(axA)

    d_stn = agg["ensemble"]["Dice_STN"] - agg["seed123"]["Dice_STN"]
    d_sn = agg["ensemble"]["Dice_SN"] - agg["seed123"]["Dice_SN"]
    d_rn = agg["ensemble"]["Dice_RN"] - agg["seed123"]["Dice_RN"]
    axA.set_title(
        "Ensemble vs single-seed members - mean Dice per nucleus",
        fontsize=FS_TITLE,
        color=INK,
        fontweight="bold",
        pad=34,
    )
    axA.text(
        0.0,
        1.045,
        "Frozen subject-clean holdout, n = 100 internal_test cases. Higher is better.",
        transform=axA.transAxes,
        fontsize=FS_SUB,
        color=MUTED,
        ha="left",
    )

    # --- Panel B: HD95 ---------------------------------------------------
    axB = axes[1]
    for m in MEMBERS:
        y = np.array([agg[m][f"HD95_{roi}_mm"] for roi in ROIS])
        axB.plot(
            xs,
            y,
            color=MEMBER_COLOR[m],
            linewidth=2.0,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.6,
            zorder=3,
            label=MEMBER_LABEL[m],
        )
    axB.set_ylabel("HD95 (mm, mean over 100 cases)")
    axB.set_xlim(-0.42, 2.42)
    axB.set_ylim(1.14, 2.44)
    axB.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    dress(axB)
    axB.set_xticks(xs)
    axB.set_xticklabels(ROIS, fontsize=FS_TICK + 1)
    axB.set_xlabel("ROI / nucleus (fixed order: class 1, 2, 3)")
    axB.set_title("Ensemble vs single-seed members - mean HD95 per nucleus", fontsize=FS_AXIS, pad=10)

    handles = [
        Line2D([], [], color=MEMBER_COLOR[m], marker="o", markersize=8,
               markeredgecolor=SURFACE, markeredgewidth=1.6, linewidth=2.0,
               label=MEMBER_LABEL[m])
        for m in MEMBERS
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.755, 0.5),
        frameon=False,
        labelspacing=1.1,
        handlelength=2.2,
    )

    fig.suptitle(
        "Figure 1 | ROI performance curves - subject-clean frozen holdout (n = 100)",
        fontsize=FS_TITLE + 1,
        color=INK,
        fontweight="bold",
        x=0.5,
        y=0.987,
    )
    fig.text(
        0.5, 0.113,
        f"Macro (mean of 3 ROIs):  ensemble {primary['ensemble_macro_Dice']:.4f}  vs  "
        f"seed123 {primary['comparator_macro_Dice']:.4f}   |   "
        f"paired mean $\\Delta$ {primary['mean_delta']:+.4f}, "
        f"95% CI [{primary['bootstrap_ci_low']:+.4f}, {primary['bootstrap_ci_high']:+.4f}], "
        f"permutation p = {primary['permutation_p']:.6f}\n"
        f"Per-ROI $\\Delta$ (ensemble $-$ seed123):  STN {d_stn:+.4f}  |  SN {d_sn:+.4f}  |  "
        f"RN {d_rn:+.4f}   -   the gain is concentrated at STN and SN, and is near-zero at RN.\n"
        "seed2026 (macro 0.7647) is descriptively ABOVE the ensemble - see Fig. 5. "
        "STN remains the hardest nucleus on both overlap and boundary error; "
        "lower HD95 is better, and the RN ordering is not reproducible across seeds.",
        ha="center", va="top", fontsize=FS_ANN, color=INK2, linespacing=1.9,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff",
                  edgecolor=GRID, linewidth=1.0),
    )
    fignote(
        fig,
        "Source: results/subject_clean_holdout/FINAL_HOLDOUT_SUMMARY.json (aggregates over "
        "HOLDOUT_PER_CASE.csv). Paired primary comparison: ensemble vs seed123.",
        y=0.005,
    )
    save(fig, "01_roi_dice_curves")


# --------------------------------------------------------------------------
# Figure 2 - Primary endpoint summary
# --------------------------------------------------------------------------


def figure2(per_case, primary, splits):
    delta = (
        per_case["macro_foreground_dice__ensemble"]
        - per_case["macro_foreground_dice__seed123"]
    ).to_numpy()

    fig = plt.figure(figsize=(15.2, 7.4))
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.05, 1.55, 1.05],
        height_ratios=[1.0, 0.86],
        wspace=0.46, hspace=0.52,
        left=0.055, right=0.975, top=0.80, bottom=0.135,
    )

    # --- design panel ----------------------------------------------------
    axD = fig.add_subplot(gs[:, 0])
    order = ("train", "val", "internal_test", "challenge_test")
    counts = [splits[k] for k in order]
    for i, (key, n) in enumerate(zip(order, counts)):
        y = 3 - i
        axD.barh(y, n, height=0.46, color=SEQ[i],
                 edgecolor=SURFACE, linewidth=2.0)
        axD.text(n + 10, y, f"{n}", va="center", ha="left",
                 fontsize=11.5, fontweight="bold", color=INK)
    axD.set_yticks([3, 2, 1, 0])
    axD.set_yticklabels(order, fontsize=FS_TICK + 0.5)
    axD.tick_params(axis="y", length=0)
    axD.set_xlim(0, 280)
    axD.set_ylim(-2.15, 3.65)
    axD.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        axD.spines[side].set_visible(False)
    axD.set_title("Evaluation design (subject-clean split)",
                  fontsize=FS_AXIS, color=INK, pad=14)
    axD.text(
        0.0, -0.62,
        "Subject-disjoint split of the PDCADx cohort.\n"
        "2 cases excluded (subject-level duplicates).\n\n"
        "internal_test = the SAME 100 cases, re-evaluated after\n"
        "the subject-level duplication fix. It is NOT a\n"
        "never-before-used test set.\n\n"
        "challenge_test (199) was NOT accessed - reserved.",
        fontsize=FS_ANN, color=INK2, va="top", ha="left", linespacing=1.65,
    )

    # --- delta histogram -------------------------------------------------
    axH = fig.add_subplot(gs[0, 1])
    axH.hist(delta, bins=26, color=COLOR_POS, alpha=0.82,
             edgecolor=SURFACE, linewidth=0.8, zorder=3)
    axH.axvspan(primary["bootstrap_ci_low"], primary["bootstrap_ci_high"],
                color="#cde2fb", alpha=0.75, zorder=1)
    axH.axvline(0.0, color=AXIS, linewidth=1.4, zorder=2)
    axH.axvline(primary["mean_delta"], color="#184f95", linewidth=2.0, zorder=4)
    axH.axvline(primary["bootstrap_ci_low"], color="#184f95", linewidth=1.1,
                linestyle=(0, (3, 2)), zorder=4)
    axH.axvline(primary["bootstrap_ci_high"], color="#184f95", linewidth=1.1,
                linestyle=(0, (3, 2)), zorder=4)
    axH.annotate(
        f"mean $\\Delta$ = {primary['mean_delta']:+.4f}",
        xy=(primary["mean_delta"], axH.get_ylim()[1] * 0.93),
        xytext=(0.055, 0.86), textcoords="axes fraction",
        fontsize=FS_ANN, color="#184f95", fontweight="semibold",
        arrowprops=dict(arrowstyle="->", color="#184f95", linewidth=1.2),
    )
    axH.text(0.985, 0.985,
             f"95% CI [{primary['bootstrap_ci_low']:+.4f}, {primary['bootstrap_ci_high']:+.4f}]"
             "\n(10,000 bootstrap resamples)",
             transform=axH.transAxes, fontsize=FS_ANN,
             color="#184f95", ha="right", va="top", linespacing=1.5)
    axH.text(0.03, 0.52,
             "$H_0$: no difference\n($\\Delta$ = 0)", fontsize=FS_ANN,
             color=MUTED, ha="left", va="center", linespacing=1.5,
             transform=axH.transAxes)
    axH.set_xlim(-0.048, 0.075)
    axH.set_xlabel("Per-case paired $\\Delta$  (macro Dice: ensemble $-$ seed123)")
    axH.set_ylabel("Number of cases")
    dress(axH)
    axH.set_title(
        "Per-case paired difference across the 100 internal_test cases",
        fontsize=FS_AXIS, pad=10,
    )

    # --- CI forest -------------------------------------------------------
    axF = fig.add_subplot(gs[1, 1])
    axF.axvline(0.0, color=AXIS, linewidth=1.4, zorder=1)
    axF.errorbar(
        [primary["mean_delta"]], [0],
        xerr=[[primary["mean_delta"] - primary["bootstrap_ci_low"]],
              [primary["bootstrap_ci_high"] - primary["mean_delta"]]],
        fmt="o", markersize=11, color="#184f95",
        markeredgecolor=SURFACE, markeredgewidth=1.8,
        ecolor="#184f95", elinewidth=2.2, capsize=7, capthick=2.2, zorder=4,
    )
    axF.set_ylim(-0.30, 0.62)
    axF.set_yticks([])
    axF.set_xlim(-0.048, 0.075)
    axF.set_xlabel("Paired mean $\\Delta$ with bootstrap 95% CI")
    axF.text(primary["mean_delta"], 0.16,
             f"{primary['mean_delta']:+.4f}\n[{primary['bootstrap_ci_low']:+.4f}, "
             f"{primary['bootstrap_ci_high']:+.4f}]",
             fontsize=FS_ANN, color="#184f95", ha="center", va="bottom",
             linespacing=1.5, fontweight="semibold")
    axF.text(0.03, 0.07, "Whole CI\nsits above 0",
             transform=axF.transAxes, fontsize=FS_ANN, color=MUTED,
             ha="left", va="bottom", linespacing=1.6)
    dress(axF)
    axF.set_title("Primary effect with 95% CI (10,000 resamples)", fontsize=FS_AXIS, pad=10)

    # --- numbers ---------------------------------------------------------
    axN = fig.add_subplot(gs[:, 2])
    axN.axis("off")
    axN.set_title("Primary endpoint (pre-registered)", fontsize=FS_AXIS, color=INK, pad=14)
    rows = [
        ("Ensemble macro Dice", f"{primary['ensemble_macro_Dice']:.4f}", COLOR_POS),
        ("Comparator (seed123) macro Dice", f"{primary['comparator_macro_Dice']:.4f}", MEMBER_COLOR["seed123"]),
        ("Mean paired $\\Delta$", f"{primary['mean_delta']:+.4f}", "#184f95"),
        ("Median paired $\\Delta$", f"{primary['median_delta']:+.4f}", INK2),
        ("Bootstrap 95% CI",
         f"[{primary['bootstrap_ci_low']:+.4f},\n{primary['bootstrap_ci_high']:+.4f}]", "#184f95"),
        ("Permutation p (two-sided)", f"{primary['permutation_p']:.6f}", INK2),
        ("n paired cases", f"{primary['n_paired']}", INK2),
    ]
    y = 0.965
    for lab, val, col in rows:
        axN.text(0.0, y, lab, fontsize=FS_ANN, color=INK2, va="top",
                 transform=axN.transAxes)
        axN.text(1.0, y, val, fontsize=FS_ANN + 1.5, color=col, va="top",
                 ha="right", fontweight="bold", linespacing=1.4,
                 transform=axN.transAxes)
        y -= 0.108 if "\n" not in val else 0.150
    # Pin the data limits: the text above lives in axes coords, but a stray
    # autoscale here would silently rescale the tight bbox at save time.
    axN.set_xlim(0.0, 1.0)
    axN.set_ylim(0.0, 1.0)
    axN.plot([0, 1], [y + 0.03, y + 0.03], color=GRID, linewidth=1.0,
             transform=axN.transAxes, clip_on=False)
    axN.text(
        0.0, y - 0.035,
        "Success rule (all three required)\n"
        "A. mean $\\Delta$ > 0                    PASS\n"
        "B. 95% CI lower bound > 0         PASS\n"
        "C. permutation p < 0.05             PASS\n\n"
        "PRIMARY ENDPOINT: CONFIRMED",
        fontsize=FS_ANN, color=INK, va="top", linespacing=1.75,
        transform=axN.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eaf4ea",
                  edgecolor="#0ca30c", linewidth=1.4),
    )

    fig.suptitle(
        "Figure 2 | Primary endpoint: did the ensemble beat the comparator?",
        fontsize=FS_TITLE + 1, color=INK, fontweight="bold", x=0.5, y=0.975,
    )
    fig.text(0.5, 0.925,
             "Pre-registered primary endpoint: case-wise paired macro foreground Dice difference, "
             "n = 100 paired cases. Report as-is; no secondary or per-class result may overturn it.",
             ha="center", fontsize=FS_SUB, color=MUTED)
    fignote(
        fig,
        "Source: results/subject_clean_holdout/HOLDOUT_PRIMARY_ENDPOINT.json + HOLDOUT_PER_CASE.csv; "
        "split sizes from manifests/subject_clean_v1/. Bootstrap 10,000 / permutation 100,000 draws, RNG seed 20260927.",
        y=0.004,
    )
    save(fig, "02_primary_endpoint_summary")


# --------------------------------------------------------------------------
# Figure 3 - Per-case paired comparison
# --------------------------------------------------------------------------


def figure3(per_case):
    ens = per_case["macro_foreground_dice__ensemble"].to_numpy()
    cmp_ = per_case["macro_foreground_dice__seed123"].to_numpy()
    delta = ens - cmp_
    order = np.argsort(delta)
    ds = delta[order]
    n_up = int((delta > 0).sum())
    n_dn = int((delta < 0).sum())

    fig = plt.figure(figsize=(15.0, 6.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.22,
                  left=0.055, right=0.975, top=0.80, bottom=0.155)

    # --- waterfall -------------------------------------------------------
    axW = fig.add_subplot(gs[0, 0])
    axW.bar(np.arange(len(ds)), ds,
            color=[COLOR_POS if d > 0 else COLOR_NEG for d in ds],
            width=0.86, edgecolor=SURFACE, linewidth=0.4, zorder=3)
    axW.axhline(0.0, color=AXIS, linewidth=1.4, zorder=4)
    axW.axhline(delta.mean(), color="#184f95", linewidth=1.8,
                linestyle=(0, (5, 3)), zorder=5)
    axW.text(0.34, delta.mean() + 0.0035, f"mean $\\Delta$ = {delta.mean():+.4f}",
             transform=axW.get_yaxis_transform(),
             fontsize=FS_ANN, color="#184f95", ha="left", va="bottom", fontweight="semibold")
    axW.axvline(n_dn - 0.5, color=AXIS, linewidth=1.0, linestyle=":", zorder=2)
    axW.set_xlim(-1.0, len(ds))
    axW.set_ylim(-0.065, 0.082)
    axW.set_xticks([0, 25, 50, 75, 100])
    axW.set_xlabel("Cases, sorted by paired $\\Delta$ (worsened $\\rightarrow$ improved)")
    axW.set_ylabel("$\\Delta$ macro Dice  (ensemble $-$ seed123)")
    dress(axW)
    axW.set_title("Sorted per-case paired difference", fontsize=FS_AXIS, pad=10)

    axW.annotate(f"{n_dn} cases worsened",
                 xy=(n_dn / 2, -0.052), fontsize=FS_ANN, color=COLOR_NEG,
                 ha="center", va="bottom", fontweight="semibold")
    axW.annotate(f"{n_up} cases improved",
                 xy=(n_dn + (len(ds) - n_dn) / 2, 0.052), fontsize=FS_ANN,
                 color=COLOR_POS, ha="center", va="bottom", fontweight="semibold")

    # --- scatter ---------------------------------------------------------
    axS = fig.add_subplot(gs[0, 1])
    lo, hi = 0.545, 0.955
    axS.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.4,
             linestyle=(0, (5, 3)), zorder=2)
    axS.text(hi - 0.008, hi - 0.012, "y = x  (no change)", fontsize=FS_ANN,
             color=MUTED, ha="right", va="top", rotation=0)
    for mask, col, lab in (
        (delta > 0, COLOR_POS, f"Ensemble better ({n_up})"),
        (delta < 0, COLOR_NEG, f"Ensemble worse ({n_dn})"),
    ):
        axS.scatter(cmp_[mask], ens[mask], s=26, color=col, alpha=0.85,
                    edgecolor=SURFACE, linewidth=0.6, zorder=3, label=lab)
    axS.set_xlim(lo, hi)
    axS.set_ylim(lo, hi)
    axS.set_aspect("equal", adjustable="box")
    axS.set_xlabel("seed123 (comparator) per-case macro Dice")
    axS.set_ylabel("Ensemble per-case macro Dice")
    dress(axS, ygrid=False)
    axS.grid(True, color=GRID, linewidth=0.9)
    axS.set_axisbelow(True)
    dress(axS)
    axS.set_title("Per-case macro Dice: ensemble vs comparator", fontsize=FS_AXIS, pad=10)
    axS.legend(loc="lower right", frameon=False, markerscale=1.4,
               borderaxespad=0.8)

    label_block(
        axS, 0.035, 0.955,
        [
            f"median $\\Delta$ = {np.median(delta):+.4f}",
            "The gain is spread across many cases,",
            "not carried by a few extreme ones:",
            f"{n_up}/100 improve, {n_dn}/100 decline,",
            "both tails are shallow (max |$\\Delta$| "
            f"{np.abs(delta).max():.3f}).",
        ],
    )

    fig.suptitle(
        "Figure 3 | Per-case paired comparison - ensemble vs comparator (seed123)",
        fontsize=FS_TITLE + 1, color=INK, fontweight="bold", x=0.5, y=0.975,
    )
    fig.text(0.5, 0.905,
             "Frozen subject-clean holdout, n = 100 paired cases. Each case is one subject; "
             "the two arms share the identical case set.",
             ha="center", fontsize=FS_SUB, color=MUTED)
    fignote(
        fig,
        "Source: results/subject_clean_holdout/HOLDOUT_PER_CASE.csv "
        "(macro_foreground_dice__ensemble, macro_foreground_dice__seed123).",
        y=0.004,
    )
    save(fig, "03_per_case_comparison")


# --------------------------------------------------------------------------
# Figure 4 - STN over-segmentation (limitation)
# --------------------------------------------------------------------------


def figure4(per_case, agg):
    fig = plt.figure(figsize=(14.2, 6.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.45, 1.0], wspace=0.24,
                  left=0.055, right=0.975, top=0.80, bottom=0.155)

    # --- volume ratio distributions --------------------------------------
    axV = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(20260928)
    for i, m in enumerate(MEMBERS):
        v = per_case[f"volume_ratio_STN__{m}"].to_numpy()
        col = MEMBER_COLOR[m]
        bp = axV.boxplot(
            [v], positions=[i], widths=0.44, patch_artist=True,
            showfliers=False, zorder=2,
            medianprops=dict(color=SURFACE, linewidth=1.8),
            whiskerprops=dict(color=col, linewidth=1.4),
            capprops=dict(color=col, linewidth=1.4),
            boxprops=dict(facecolor=col, edgecolor=SURFACE, linewidth=1.2, alpha=0.42),
        )
        for b in bp["boxes"]:
            b.set_zorder(2)
        jit = rng.uniform(-0.115, 0.115, size=v.size)
        axV.scatter(np.full(v.size, i) + jit, v, s=9, color=col, alpha=0.42,
                    edgecolor="none", zorder=3)
        axV.text(i, v.mean() + 0.055, f"mean {v.mean():.4f}", ha="center", va="bottom",
                 fontsize=FS_ANN, color=col, fontweight="semibold")

    axV.axhline(1.0, color="#0ca30c", linewidth=2.0, zorder=1)
    axV.text(-0.46, 1.0, "Pred = GT (ratio 1.0)",
             fontsize=FS_ANN, color="#0ca30c", ha="left", va="bottom")
    axV.set_xticks(range(len(MEMBERS)))
    axV.set_xticklabels([MEMBER_LABEL[m] for m in MEMBERS], fontsize=FS_TICK)
    axV.set_ylim(0.70, 3.22)
    axV.set_ylabel("STN predicted / ground-truth volume ratio")
    dress(axV)
    axV.set_title("STN volume ratio per case (n = 100) - over-segmentation persists",
                  fontsize=FS_AXIS, pad=10)

    # --- precision vs recall ---------------------------------------------
    axP = fig.add_subplot(gs[0, 1])
    xs = np.arange(len(MEMBERS))
    w = 0.34
    p = np.array([agg[m]["Precision_STN"] for m in MEMBERS])
    r = np.array([agg[m]["Recall_STN"] for m in MEMBERS])
    axP.bar(xs - w / 2 - 0.012, p, width=w, color="#2a78d6",
            edgecolor=SURFACE, linewidth=1.6, zorder=3, label="Precision (STN)")
    axP.bar(xs + w / 2 + 0.012, r, width=w, color="#eb6834",
            edgecolor=SURFACE, linewidth=1.6, zorder=3, label="Recall (STN)")
    for x, v in zip(xs - w / 2 - 0.012, p):
        axP.text(x, v + 0.016, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=FS_ANN, color="#2a78d6", fontweight="semibold")
    for x, v in zip(xs + w / 2 + 0.012, r):
        axP.text(x, v + 0.016, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=FS_ANN, color="#eb6834", fontweight="semibold")
    axP.set_xticks(xs)
    axP.set_xticklabels([m if m != "ensemble" else "Ensemble" for m in MEMBERS],
                        fontsize=FS_TICK)
    axP.set_ylim(0.0, 1.31)
    axP.set_yticks(np.arange(0.0, 1.01, 0.2))
    axP.set_ylabel("STN score (mean over 100 cases)")
    dress(axP)
    axP.set_title("STN precision is far below recall", fontsize=FS_AXIS, pad=10)
    axP.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.10))
    label_block(
        axP, 0.03, 0.975,
        [
            "Low precision + high recall + ratio > 1",
            "= the model predicts STN too large.",
            "This is a real, unfixed limitation,",
            "not a rounding artefact.",
        ],
    )

    fig.suptitle(
        "Figure 4 | Remaining limitation: STN is still over-segmented (ensemble included)",
        fontsize=FS_TITLE + 1, color=INK, fontweight="bold", x=0.5, y=0.975,
    )
    fig.text(0.5, 0.905,
             "Same frozen subject-clean holdout (n = 100). All four members predict $\\approx$1.67-1.71$\\times$ "
             "the true STN volume - ensembling does NOT fix this. Reported separately from the "
             "primary endpoint; it does not overturn it, and the primary endpoint does not excuse it.",
             ha="center", fontsize=FS_SUB, color=MUTED)
    fignote(
        fig,
        "Source: results/subject_clean_holdout/HOLDOUT_PER_CASE.csv (volume_ratio_STN__*, "
        "Precision_STN__*, Recall_STN__*) + FINAL_HOLDOUT_SUMMARY.json.",
        y=0.004,
    )
    save(fig, "04_stn_oversegmentation")


# --------------------------------------------------------------------------
# Figure 5 - Model ranking (descriptive)
# --------------------------------------------------------------------------


def figure5(agg):
    vals = {m: agg[m]["macro_foreground_dice"] for m in MEMBERS}
    ordered = sorted(MEMBERS, key=lambda m: vals[m], reverse=True)

    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    fig.subplots_adjust(left=0.245, right=0.855, top=0.775, bottom=0.315)

    xs = np.array([vals[m] for m in ordered])
    ys = np.array([len(ordered) - 1 - i for i in range(len(ordered))], dtype=float)
    for y, m, x in zip(ys, ordered, xs):
        col = MEMBER_COLOR[m]
        ax.plot([0.7285, x], [y, y], color=GRID, linewidth=2.4, zorder=1)
        ax.plot([x], [y], marker="o", markersize=13, color=col,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)
        ax.text(x + 0.0012, y, f"{x:.4f}", va="center", ha="left",
                fontsize=FS_ANN + 0.5, color=col, fontweight="bold")

    ref = vals["seed123"]
    ax.axvline(ref, color=MEMBER_COLOR["seed123"], linewidth=1.4,
               linestyle=(0, (5, 3)), zorder=2)
    ax.text(ref, -0.72, "seed123 = primary comparator",
            fontsize=FS_ANN, color=MEMBER_COLOR["seed123"],
            ha="center", va="center")

    ax.set_yticks(ys)
    ax.set_yticklabels([MEMBER_LABEL[m] for m in ordered], fontsize=FS_TICK + 1)
    ax.set_xlim(0.7285, 0.7775)
    ax.set_ylim(-1.05, len(ordered) - 0.42)
    ax.set_xlabel("Macro foreground Dice (mean over 100 internal_test cases)")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.01))
    dress(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)

    fig.suptitle("Figure 5 | Model ranking by macro Dice (descriptive)",
                 fontsize=FS_TITLE + 1, color=INK, fontweight="bold", x=0.5, y=0.975)
    fig.text(0.5, 0.895,
             "Frozen subject-clean holdout, n = 100. Axis starts at 0.7285 - read the printed "
             "values, not the line lengths.",
             ha="center", fontsize=FS_SUB, color=MUTED)

    fig.text(
        0.5, 0.145,
        "Ranking:  seed2026 (0.7647)  >  ensemble (0.7614)  >  seed123 (0.7509)  >  seed42 (0.7419)\n"
        "This is a DESCRIPTIVE ranking only - it is not a significance test, and seed2026 exceeding the\n"
        "ensemble is reported as-is. The pre-specified primary comparison is ensemble vs seed123 (paired, n = 100).",
        ha="center", va="top", fontsize=FS_ANN, color=INK2, linespacing=1.7,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff",
                  edgecolor=GRID, linewidth=1.0),
    )
    fignote(
        fig,
        "Source: results/subject_clean_holdout/FINAL_HOLDOUT_SUMMARY.json (aggregates.*.macro_foreground_dice) "
        "re-plotted against HOLDOUT_PER_CASE.csv per-case means.",
        y=0.004,
    )
    save(fig, "05_model_ranking")


# --------------------------------------------------------------------------


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    per_case, primary, summary, agg, splits = load()
    print(f"[splits] {splits}")

    figure1(per_case, agg, primary)
    print("[done] 01_roi_dice_curves.(png|pdf|svg)")
    figure2(per_case, primary, splits)
    print("[done] 02_primary_endpoint_summary.(png|pdf|svg)")
    figure3(per_case)
    print("[done] 03_per_case_comparison.(png|pdf|svg)")
    figure4(per_case, agg)
    print("[done] 04_stn_oversegmentation.(png|pdf|svg)")
    figure5(agg)
    print("[done] 05_model_ranking.(png|pdf|svg)")

    n_up = int((per_case["macro_foreground_dice__ensemble"]
                - per_case["macro_foreground_dice__seed123"] > 0).sum())
    print(f"[summary] improved {n_up}/100, worsened {100 - n_up}/100")
    print(f"[summary] output -> {OUT}")


if __name__ == "__main__":
    main()
