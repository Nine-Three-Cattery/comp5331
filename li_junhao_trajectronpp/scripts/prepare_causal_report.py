#!/usr/bin/env python3
"""Generate English report tables, shared results and speaker notes from audited data."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE / "outputs/causal_v2"
REPORT = PACKAGE / "report"
RESULTS = ROOT / "results"

FOLD_NAMES = {"eth": "ETH", "hotel": "Hotel", "univ": "Univ", "zara1": "Zara1", "zara2": "Zara2"}
CONDITION_NAMES = {
    "clean_0": "Clean",
    "noise_005": "Noise 0.05 m",
    "noise_010": "Noise 0.10 m",
    "noise_020": "Noise 0.20 m",
    "missing_1": "Missing one",
    "missing_2": "Missing two",
}
METRICS = ("min_ade_mean", "min_fde_mean", "mean_ade_mean", "mean_fde_mean",
           "energy_score_mean", "sample_diversity_mean")


def latex_row(values):
    return " & ".join(str(item) for item in values) + r" \\"


def plot_robustness(macro):
    fig, ax = plt.subplots(figsize=(6.1, 3.25), layout="constrained")
    x = np.arange(len(CONDITION_NAMES))
    for field, label, color, marker in (
        ("mean_ade_mean", "Mean ADE", "#923520", "o"),
        ("min_ade_mean", "Min ADE", "#235b9e", "s"),
        ("energy_score_mean", "Energy Score", "#28745c", "^"),
    ):
        values = [macro.loc[(condition, field), "mean"] for condition in CONDITION_NAMES]
        ax.scatter(x, values, c=color, label=label, marker=marker, s=38, zorder=3)
        ax.plot(x[:4], values[:4], color=color, alpha=0.8)
        ax.plot(x[4:], values[4:], color=color, alpha=0.8)
    ax.axvline(3.5, color="0.7", linewidth=0.8)
    ax.set_xticks(x, ["Clean", "0.05 m", "0.10 m", "0.20 m", "Missing 1", "Missing 2"])
    ax.set_ylabel("Equal-weight fold macro (m)")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(RESULTS / "robustness.pdf")
    plt.close(fig)


def main():
    audit = json.loads((ROOT / "audit.json").read_text())
    fixture = json.loads((ROOT / "corruption_fixture.json").read_text())
    assert audit["status"] == "passed" and audit["prediction_bundles"] == 90
    results = pd.read_csv(RESULTS / "results.csv")
    index = pd.read_csv(ROOT / "prediction_index.csv")
    main_table = index.merge(
        results.drop(columns=["window_count", "sample_count", "checkpoint"]),
        on=["fold", "seed", "condition"], validate="one_to_one",
    )
    assert len(main_table) == 90
    main_table.to_csv(RESULTS / "group7_trajectronpp.csv", index=False)
    macro = pd.read_csv(RESULTS / "formal_statistical_macro.csv").set_index(["condition", "metric"])
    effects = pd.read_csv(RESULTS / "formal_statistical_effects.csv").set_index(["condition", "metric"])
    runtime = pd.read_csv(RESULTS / "runtime.csv")
    budget = pd.read_csv(RESULTS / "sample_budget_overall.csv").set_index(["condition", "sample_budget"])
    clean_runs = results[results["condition"] == "clean_0"]
    clean = clean_runs.groupby("fold").mean(numeric_only=True)
    std = clean_runs.groupby("fold")["min_ade_mean"].std()
    spatial = pd.read_csv(RESULTS / "spatial.csv")
    close = spatial[spatial["condition"] == "clean_0"]["close_approach_rate_mean"].mean()
    failures = pd.read_csv(RESULTS / "formal_failure_cases.csv")
    failure = failures.loc[failures["mean_ade_clean"].idxmax()]

    def value(condition, metric):
        return float(macro.loc[(condition, metric), "mean"])

    def delta(condition, metric):
        return float(effects.loc[(condition, metric), "effect_mean"])

    def interval(condition):
        item = effects.loc[(condition, "mean_ade_mean")]
        return f"[{item.ci_low:.3f}, {item.ci_high:.3f}]"

    clean_values = [value("clean_0", metric) for metric in METRICS]
    values = {
        "TestWindows": f'{audit["test_windows"]:,}',
        "CachedWindows": f'{fixture["cached_windows_checked"]:,}',
        "CleanMinADE": f"{clean_values[0]:.3f}",
        "CleanMinFDE": f"{clean_values[1]:.3f}",
        "CleanMeanADE": f"{clean_values[2]:.3f}",
        "CleanEnergy": f"{clean_values[4]:.3f}",
        "NoiseMeanDelta": f"{delta('noise_020', 'mean_ade_mean'):+.3f}",
        "MissingMeanDelta": f"{delta('missing_2', 'mean_ade_mean'):+.3f}",
        "NoiseMeanInterval": interval("noise_020"),
        "MissingMeanInterval": interval("missing_2"),
        "OneSampleADE": f"{budget.loc[('clean_0', 1), 'min_ade_mean_mean']:.3f}",
        "CleanSpeed": f"{clean_runs['speed_violation_rate_mean'].mean():.1%}",
        "CleanAccel": f"{clean_runs['acceleration_violation_rate_mean'].mean():.1%}",
        "CleanJerk": f"{clean_runs['jerk_violation_rate_mean'].mean():.1%}",
        "CleanClose": f"{close:.1%}",
        "RuntimePre": f"{runtime['preprocessing_ms'].median():.1f}",
        "RuntimePred": f"{runtime['prediction_sampling_ms'].median():.1f}",
        "RuntimeTotal": f"{runtime['end_to_end_ms'].median():.1f}",
        "FailureADE": f"{failure.mean_ade_clean:.3f}",
        "FailureFold": FOLD_NAMES[failure.fold],
    }
    macros = ["% Generated from the audited causal_v2 result tables."]
    for key, val in values.items():
        macros.append("\\newcommand{\\" + key + "}{" + val.replace("%", r"\%") + "}")
    macros.extend([
        r"\newcommand{\FoldRows}{",
        *[latex_row((label, f"{clean.loc[fold, 'window_count']:,.0f}",
                     f"{clean.loc[fold, 'min_ade_mean']:.3f}", f"{std.loc[fold]:.3f}",
                     *[f"{clean.loc[fold, name]:.3f}" for name in
                       ("min_fde_mean", "mean_ade_mean", "energy_score_mean")]))
          for fold, label in FOLD_NAMES.items()],
        "}",
        r"\newcommand{\SlideFoldRows}{",
        *[latex_row((label, *[f"{clean.loc[fold, name]:.3f}" for name in
                             ("min_ade_mean", "mean_ade_mean", "energy_score_mean")]))
          for fold, label in FOLD_NAMES.items()],
        "}",
        r"\newcommand{\ConditionRows}{",
        *[latex_row((label, *[f"{value(condition, metric):.3f}" for metric in METRICS]))
          for condition, label in CONDITION_NAMES.items()],
        "}",
    ])
    (REPORT / "causal_results.tex").write_text("\n".join(macros) + "\n")
    plot_robustness(macro)
    (REPORT / "trajectronpp_section.md").write_text(group_section(values, clean, std, value, effects))
    (REPORT / "TrajectronPP_5min_Speaker_Notes.md").write_text(speaker_notes(values))
    print("CAUSAL_REPORT_DATA_OK report, slides and speaker notes share audited values")


def group_section(v, clean, std, value, effects):
    text = f"""# Trajectron++ pedestrian implementation — LI Junhao

## Method and protocol

Our independent pedestrian implementation encodes target and 3 m neighbor
histories, samples a 25-category latent prior, and integrates Gaussian GRU
velocity samples into metric positions. The CVAE posterior observes future
labels only during training. The model uses one pedestrian edge type and
trajectory inputs. Complete neighbor histories, a hard radius mask and the
velocity NLL objective are documented deviations from the broader paper
configuration. [Trajectron++ paper](https://arxiv.org/abs/2001.03093)

The benchmark uses eight observations, twelve future steps and 0.4 s
spacing. We trained 100 epochs on five scene holdouts with three seeds and
selected checkpoints by clean validation Energy Score at K=20. A causal
input audit replaced whole-track centered derivatives with within-window
backward differences for targets and neighbors. Changing either agent's
positions after the anchor leaves predictions identical with a fixed seed.
The target-only noise and missing-suffix conditions use the same ordered
windows and unchanged test futures. Neighbor observations remain available.

## Results

The audit covers 15 fold–seed runs, 90 prediction bundles and {v['TestWindows']}
unique test windows. The clean scene macro is {v['CleanMinADE']} m minADE,
{v['CleanMinFDE']} m minFDE, {v['CleanMeanADE']} m meanADE and
{v['CleanEnergy']} m Energy Score. Each row below averages the three seeds.
SD is the across-seed minADE standard deviation.

| Holdout | Windows | minADE (m) | SD (m) | minFDE (m) | meanADE (m) | ES (m) |
|---|---:|---:|---:|---:|---:|---:|
"""
    for fold, label in FOLD_NAMES.items():
        row = clean.loc[fold]
        text += (f"| {label} | {row.window_count:,.0f} | {row.min_ade_mean:.3f} | "
                 f"{std.loc[fold]:.3f} | {row.min_fde_mean:.3f} | {row.mean_ade_mean:.3f} | "
                 f"{row.energy_score_mean:.3f} |\n")
    text += """
Each scene contributes equally to the following condition-level macro.
All distances are in meters.

| Condition | minADE | minFDE | meanADE | meanFDE | Energy Score | Diversity |
|---|---:|---:|---:|---:|---:|---:|
"""
    for condition, label in CONDITION_NAMES.items():
        text += f"| {label} | " + " | ".join(f"{value(condition, metric):.3f}" for metric in METRICS) + " |\n"
    noise = effects.loc[("noise_020", "mean_ade_mean")]
    missing = effects.loc[("missing_2", "mean_ade_mean")]
    return text + f"""
0.20 m noise changes meanADE by {v['NoiseMeanDelta']} m relative to clean
(95% paired scene bootstrap interval [{noise.ci_low:.3f}, {noise.ci_high:.3f}]).
Missing two changes it by {v['MissingMeanDelta']} m
([{missing.ci_low:.3f}, {missing.ci_high:.3f}]). These intervals describe
variation across five folds. The clean motion-flag fractions are
{v['CleanSpeed']} for speed, {v['CleanAccel']} for acceleration and
{v['CleanJerk']} for jerk. The clean close-approach fraction is
{v['CleanClose']} under train-only distance thresholds. Equal-index matching
of separately sampled targets is a descriptive proximity diagnostic.

The seed-0 failure list contains a clean window with {v['FailureADE']} m
meanADE on {v['FailureFold']}. Fixed and rule-selected figures show observed
histories, samples, endpoints and ground truth under clean, strong noise
and missing-two conditions. The common GPU benchmark (128 targets, batch
32, K=20) reports {v['RuntimePre']} ms preprocessing, {v['RuntimePred']} ms
combined inference and sampling, and {v['RuntimeTotal']} ms end-to-end,
each a median of run medians. Preprocessing starts from in-memory neutral
rows and end-to-end ends with metric predictions on CPU.

## Group handoff

Use `group7_trajectronpp.csv`, the prediction index and corruption fixture
to match manifest hashes, sample budgets, masks, metric definitions and
runtime boundaries before comparing with the other five models. These
results characterize the project-specific pedestrian implementation and
five test scenes.
"""


def speaker_notes(v):
    return f"""# Trajectron++ 5-minute speaker notes

## Slide 1 · 0:00–0:30

I implemented the trajectory-only pedestrian version of Trajectron++ for
Group 7. Our task is to forecast twelve future positions from eight observed
positions on ETH/UCY. Instead of committing to a single path, this model
produces twenty plausible trajectories for each target. I'll explain the
forecasting mechanism, the causal-input audit and the results the group can
compare under the same protocol.

## Slide 2 · 0:30–1:15

The model encodes the target history and the histories of nearby pedestrians
with recurrent networks. It uses a three-meter graph and sums neighbors
within one pedestrian edge type. The history context then defines a
categorical prior over twenty-five latent classes. A recurrent decoder
samples correlated two-dimensional velocities, and a single integrator
turns them into positions in meters. During training, the variational
posterior can use the labeled future. At inference, the prior sees only the
available target and neighbor histories. The implementation uses trajectory
inputs throughout, as required by the common pedestrian benchmark.

## Slide 3 · 1:15–2:05

We hold out each scene in turn and train with three seeds. A checkpoint is
chosen on the validation split by Energy Score, with the same twenty-sample
budget. The causal audit found that centered differences over an entire track
could put the next observed-in-hindsight position into the last history
velocity. We rebuilt the target and neighbor states with backward differences
inside the window and regenerated the caches. For every input condition, the
regression changes future target and neighbor coordinates separately and
checks that features and same-seed forecasts stay identical. That check
matters because a correct-looking eight-position window alone cannot expose
leakage through its derived features.

## Slide 4 · 2:05–2:50

We average each held-out fold over its three seeds, then give the five folds
equal weight. On clean data, minADE is {v['CleanMinADE']} meters and minFDE is
{v['CleanMinFDE']} meters. Mean ADE is {v['CleanMeanADE']} meters, and Energy
Score is {v['CleanEnergy']} meters. The best-of-twenty result asks whether
the sampled distribution contains a trajectory near the truth. Mean error
asks how far away a typical sample is. Energy Score combines accuracy with
sample spread. The fold table lets us see scene variation without allowing
the largest test scene to dominate the headline number.

## Slide 5 · 2:50–3:35

We add three levels of Gaussian position noise or hide the last one or two
target observations. A constant-velocity fill uses the last two visible
points. Each perturbation keeps the same test windows, future targets and
neighbor history. Relative to clean, twenty centimeters of noise changes
mean ADE by {v['NoiseMeanDelta']} meters and missing two changes it by
{v['MissingMeanDelta']} meters. The chart shows how typical error, best
sample error and Energy Score respond under the same perturbation. These
results describe degradation of the target history while neighbor context is
still available.

## Slide 6 · 3:35–4:25

This is the same ETH window in three conditions, selected by the fixed
manifest rule. Black shows the observed prefix and dashed black shows the
causal fill. The orange path is the labeled future. Blue paths and endpoints
show the twenty predictions. The scale is shared across panels so the
spread is comparable. We also timed a fixed batch workload of one hundred
twenty-eight target windows. The median end-to-end result is
{v['RuntimeTotal']} milliseconds, starting from in-memory neutral rows and
ending with metric predictions on the CPU. Separate times are available for
preprocessing and combined inference with sampling.

## Slide 7 · 4:25–5:00

The deliverable includes fifteen selected checkpoints, ninety test prediction
bundles and an index that links each result to its manifest and configuration.
A group member can reproduce one selected prediction or regenerate the
headline table from stored samples. I would compare models only after matching
their windows, perturbations and evaluator. The proximity diagnostic
describes how often separately sampled paths pass close together under a
training-derived threshold. It does not estimate a calibrated probability of
collision.
"""


if __name__ == "__main__":
    main()
