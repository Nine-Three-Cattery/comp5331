"""Build the group figures from audited summaries. Requires Matplotlib."""

import csv
import math
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FOLDS = {"eth": "ETH", "hotel": "Hotel", "univ": "Univ", "zara1": "Zara1", "zara2": "Zara2"}
CONDITIONS = {"noise_005": "Noise 0.05 m", "noise_010": "Noise 0.10 m",
              "noise_020": "Noise 0.20 m", "missing_1": "Missing 1", "missing_2": "Missing 2"}
INK = "#172d3a"
MUTED = "#405864"
TEAL = "#007465"
ORANGE = "#ad482b"
GRID = "#d9e4e4"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "pdf.fonttype": 42,
})


def read(name):
    with (HERE / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def setup(ax, title):
    ax.set_title(title, loc="left", weight="bold", fontsize=14, pad=12)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=5)


def save(fig, name):
    fig.savefig(HERE / "assets" / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def sampling_chart(budgets, fold_min, fold_mean):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.1, 3.18))
    fig.subplots_adjust(left=.085, right=.975, bottom=.22, top=.83, wspace=.38)
    ks = [int(row["sample_budget"]) for row in budgets]
    minimum = [float(row["min_ade_mean_mean"]) for row in budgets]
    typical = [float(row["mean_ade_mean_mean"]) for row in budgets]
    setup(left, "A  |  Same forecast stream")
    left.plot(ks, typical, "s-", color=ORANGE, linewidth=2, markersize=6)
    left.plot(ks, minimum, "o-", color=TEAL, linewidth=2, markersize=6)
    left.set(xticks=ks, xlim=(0, 21), ylim=(0, 1.04),
             xlabel="Samples scored (K)", ylabel="Clean ADE (m)")
    left.text(20, .93, "Mean", color=ORANGE, ha="right", weight="bold")
    left.text(20, .20, "Minimum", color=TEAL, ha="right", weight="bold")
    left.grid(axis="y", color=GRID)
    setup(right, "B  |  Five held-out scenes")
    for index, fold in enumerate(FOLDS):
        best, avg = fold_min[fold], fold_mean[fold]
        right.hlines(index, best, avg, color=GRID, linewidth=3)
        right.scatter(best, index, color=TEAL, s=57, zorder=3)
        right.scatter(avg, index, color=ORANGE, marker="s", s=52, zorder=3)
    right.set(yticks=range(len(FOLDS)), yticklabels=list(FOLDS.values()), xlim=(0, 1.55),
              ylim=(4.55, -.62), xlabel="Clean ADE at K=20 (m)")
    right.grid(axis="x", color=GRID)
    save(fig, "sampling")



def sampling_slide(budgets):
    fig, ax = plt.subplots(figsize=(6.4, 3.45))
    fig.subplots_adjust(left=.16, right=.96, bottom=.22, top=.95)
    k = [int(row["sample_budget"]) for row in budgets]
    setup(ax, "")
    ax.plot(k, [float(row["mean_ade_mean_mean"]) for row in budgets],
            "s-", color=ORANGE, linewidth=2.8, markersize=8)
    ax.plot(k, [float(row["min_ade_mean_mean"]) for row in budgets],
            "o-", color=TEAL, linewidth=2.8, markersize=8)
    ax.set(xticks=k, xlim=(0, 21), ylim=(0, 1.0), xlabel="Samples scored (K)", ylabel="Clean ADE (m)")
    ax.tick_params(labelsize=13)
    ax.text(19.9, .91, "Mean", color=ORANGE, ha="right", weight="bold", fontsize=14)
    ax.text(19.9, .22, "Minimum", color=TEAL, ha="right", weight="bold", fontsize=14)
    ax.grid(axis="y", color=GRID)
    save(fig, "sampling_slide")


def robustness_chart(effects, deltas):
    fig, ax = plt.subplots(figsize=(8.8, 3.24))
    fig.subplots_adjust(left=.20, right=.84, top=.80, bottom=.18)
    setup(ax, "Paired change from clean")
    ax.axvline(0, linewidth=1, color=MUTED)
    ax.grid(axis="x", color=GRID)
    for index, tag in enumerate(CONDITIONS):
        color = ORANGE if tag.startswith("noise") else TEAL
        value, low, high = (float(effects[tag][key]) for key in ("effect_mean", "ci_low", "ci_high"))
        ax.scatter([deltas[tag, fold] for fold in FOLDS],
                   [index + offset for offset in (-.15, -.08, 0, .08, .15)],
                   facecolor="white", edgecolor=color, linewidth=1, s=30, zorder=3)
        ax.errorbar(value, index, xerr=[[value-low], [high-value]], fmt="D",
                    color=color, linewidth=2.5, markersize=7, capsize=4, zorder=4)
        ax.text(1.04, index, f"{value:+.3f} m", transform=ax.get_yaxis_transform(),
                color=color, weight="bold", va="center", fontsize=13)
    ax.set(yticks=range(len(CONDITIONS)), yticklabels=list(CONDITIONS.values()),
           ylim=(4.64, -.62), xlim=(-.16, 3.01), xlabel="Change in meanADE (m)")
    ax.text(1.04, -.72, "FOLD MACRO", transform=ax.get_yaxis_transform(), color=MUTED, fontsize=9)
    save(fig, "paired_robustness")


def main():
    rows = read("results.csv")

    def fold_metric(tag, key, fold):
        return mean(float(row[key]) for row in rows if row["condition"] == tag and row["fold"] == fold)

    fold_min = {fold: fold_metric("clean_0", "min_ade_mean", fold) for fold in FOLDS}
    fold_mean = {fold: fold_metric("clean_0", "mean_ade_mean", fold) for fold in FOLDS}
    deltas = {(tag, fold): fold_metric(tag, "mean_ade_mean", fold) - fold_mean[fold]
              for tag in CONDITIONS for fold in FOLDS}
    budgets = read("evidence/sample_budget.csv")
    effects = {row["condition"]: row for row in read("evidence/paired_mean_ade.csv")}
    for tag in CONDITIONS:
        assert math.isclose(mean(deltas[tag, fold] for fold in FOLDS),
                            float(effects[tag]["effect_mean"]), abs_tol=1e-12)
    assert math.isclose(mean(fold_min.values()), float(budgets[-1]["min_ade_mean_mean"]), abs_tol=1e-6)
    runtime = read("evidence/runtime.csv")
    univ = [row for row in runtime if row["fold"] == "univ"]
    values = {
        "OneSampleADE": f'{float(budgets[0]["min_ade_mean_mean"]):.3f}',
        "BudgetReduction": f'{100*(1-mean(fold_min.values())/float(budgets[0]["min_ade_mean_mean"])):.0f}',
        "ETHMin": f'{fold_min["eth"]:.3f}', "ZaraTwoMin": f'{fold_min["zara2"]:.3f}',
        "MaxSeedSD": f'{max(stdev(float(row["min_ade_mean"]) for row in rows if row["condition"] == "clean_0" and row["fold"] == fold) for fold in FOLDS):.3f}',
        "NoiseDominantFolds": str(sum(deltas["noise_005", fold] > deltas["missing_2", fold] for fold in FOLDS)),
        "UnivRuntime": f'{median(float(row["end_to_end_ms"]) for row in univ):.1f}',
        "UnivPreprocess": f'{median(float(row["preprocessing_ms"]) for row in univ):.1f}',
    }
    lines = ["% Generated by refresh_charts.py from audited result snapshots."]
    lines += [rf"\expandafter\def\csname tpp:{key}\endcsname{{{value}}}" for key, value in values.items()]
    (HERE / "analysis.tex").write_text("\n".join(lines) + "\n")
    sampling_chart(budgets, fold_min, fold_mean)
    sampling_slide(budgets)
    robustness_chart(effects, deltas)


if __name__ == "__main__":
    main()
