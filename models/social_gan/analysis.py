"""Derive compact evidence from the shared scorer and audited Social-GAN bundles."""

from pathlib import Path
from statistics import mean, median, stdev
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/social_gan"
FORMAL = ARTIFACTS / "formal/runs"
CONTRIBUTION = ROOT / "submission/contributions/social_gan"
EVIDENCE = CONTRIBUTION / "evidence"
ASSETS = CONTRIBUTION / "assets"
QUALITATIVE = ARTIFACTS / "qualitative"
FOLDS = {"eth": "ETH", "hotel": "Hotel", "univ": "Univ", "zara1": "Zara1", "zara2": "Zara2"}
CONDITIONS = {
    "clean_0": "Clean",
    "noise_005": "Noise 0.05 m",
    "noise_010": "Noise 0.10 m",
    "noise_020": "Noise 0.20 m",
    "missing_1": "Missing 1",
    "missing_2": "Missing 2",
}
METRICS = ("min_ade", "min_fde", "mean_ade", "mean_fde", "energy_score", "sample_diversity")
RUNTIME = ("preprocessing_ms", "prediction_sampling_ms", "end_to_end_ms")
MOTION = ("speed", "acceleration", "jerk")
INK, MUTED, TEAL, RUST, RULE = "#172d3a", "#405864", "#007465", "#ad482b", "#cfdede"

sys.path.insert(0, str(ROOT / "li_junhao_trajectronpp/scripts"))
from analyze_sample_budget import metrics_for_budget_batch


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_table(rows, name):
    frame = pd.DataFrame(rows)
    frame.to_csv(EVIDENCE / f"{name}.csv", index=False, lineterminator="\n")
    return frame


def run_path(fold, seed):
    return FORMAL / fold / f"seed_{seed}"


def bundle_path(fold, seed, tag):
    return run_path(fold, seed) / "evaluation" / tag / f"predictions_test_{tag}.npz"


def metrics_path(fold, seed, tag):
    return ARTIFACTS / "metrics" / fold / f"seed_{seed}" / tag


def clean_and_robustness(results):
    metric_fields = [f"{metric}_mean" for metric in METRICS]
    clean = results.xs("clean_0", level="condition")
    clean_rows = []
    for fold in (*FOLDS, "macro"):
        per_seed = (
            clean[metric_fields].groupby("seed").mean()
            if fold == "macro" else clean.loc[fold, metric_fields]
        )
        row = {
            "fold": fold,
            "window_count": (int(clean.xs(0, level="seed")["window_count"].sum())
                             if fold == "macro" else int(clean.loc[(fold, 0), "window_count"])),
        }
        for metric, field in zip(METRICS, metric_fields):
            row[f"{metric}_mean"] = per_seed[field].mean()
            row[f"{metric}_seed_sd"] = per_seed[field].std(ddof=1)
        clean_rows.append(row)
    clean_table = save_table(clean_rows, "clean_by_fold")

    robustness_rows = []
    for tag in CONDITIONS:
        paired = results.xs(tag, level="condition")[metric_fields] - clean[metric_fields]
        per_fold = paired.groupby("fold").mean()
        for fold in (*FOLDS, "macro"):
            row = {"fold": fold, "condition": tag}
            values = per_fold.mean() if fold == "macro" else per_fold.loc[fold]
            row.update({f"delta_{metric}": values[field] for metric, field in zip(METRICS, metric_fields)})
            if fold == "macro":
                row["delta_mean_ade_fold_min"] = per_fold["mean_ade_mean"].min()
                row["delta_mean_ade_fold_max"] = per_fold["mean_ade_mean"].max()
            robustness_rows.append(row)
    return clean_table, save_table(robustness_rows, "paired_robustness")


def sample_budget(results):
    rows = []
    clean = results.xs("clean_0", level="condition")
    for fold, seed in clean.index:
        with np.load(bundle_path(fold, seed, "clean_0"), allow_pickle=False) as bundle:
            samples = bundle["samples"].astype(np.float64)
            truth = bundle["ground_truth"].astype(np.float64)
        for k in (1, 5, 10, 20):
            if k == 20:
                values = {metric: clean.loc[(fold, seed), f"{metric}_mean"] for metric in METRICS}
            else:
                values = {metric: value.mean() for metric, value in metrics_for_budget_batch(samples, truth, k).items()}
            rows.append({"fold": fold, "seed": seed, "sample_count": k, **values})
    runs = save_table(rows, "sample_budget_runs")
    macro = runs.groupby(["sample_count", "fold"])[list(METRICS)].mean().groupby("sample_count").mean()
    return save_table(macro.reset_index(), "sample_budget")


def runtime_table(results):
    rows = []
    environments = {}
    for fold, seed in results.xs("clean_0", level="condition").index:
        log = read_json(run_path(fold, seed) / "runtime.json")
        repetitions = log["repetitions"]
        if len(repetitions) != 10:
            raise ValueError(f"{fold}/{seed} should contain ten runtime measurements")
        timings = {key: [float(rep[key]) for rep in repetitions] for key in RUNTIME}
        if any(not math.isfinite(value) or value < 0 for series in timings.values() for value in series):
            raise ValueError(f"{fold}/{seed} has invalid runtime measurements")
        rows.append({"fold": fold, "seed": seed, **{key: median(series) for key, series in timings.items()}})
        environments[f"{fold}/seed_{seed}"] = log["environment"]
    return save_table(rows, "runtime_cpu"), environments


def motion_spatial(results):
    rows = []
    for fold, seed, tag in results.index:
        path = metrics_path(fold, seed, tag)
        report = read_json(path / "summary.json")
        spatial = pd.read_csv(path / "spatial_consistency.csv")
        row = {"fold": fold, "seed": seed, "condition": tag, "multi_target_origins": len(spatial)}
        for quantity in MOTION:
            row[f"{quantity}_threshold"] = report["thresholds"][quantity]
            row[f"{quantity}_violation_fraction"] = report["overall"][f"{quantity}_violation_rate_mean"]
        row["close_approach_threshold_m"] = report["thresholds"]["close_approach_distance"]
        row["close_approach_fraction"] = spatial["close_approach_rate"].mean()
        rows.append(row)
    runs = save_table(rows, "motion_spatial")
    fields = [f"{quantity}_violation_fraction" for quantity in MOTION] + ["close_approach_fraction"]
    macro = runs.groupby(["condition", "fold"])[fields].mean().groupby("condition").mean()
    return save_table(macro.reindex(CONDITIONS).reset_index(), "motion_spatial_macro")


def style(axis):
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(RULE)
    axis.tick_params(length=0, pad=5)


def save_plot(figure, path):
    figure.savefig(path, bbox_inches="tight", pad_inches=0.07,
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def sampling_plot(budgets):
    figure, axis = plt.subplots(figsize=(6.5, 3.15), layout="constrained")
    axis.plot(budgets["sample_count"], budgets["mean_ade"], "s-", label="Mean", color=RUST, lw=2.2)
    axis.plot(budgets["sample_count"], budgets["min_ade"], "o-", label="Minimum", color=TEAL, lw=2.2)
    axis.set(xlabel="Samples scored (K)", ylabel="Clean ADE (m)", xticks=budgets["sample_count"], ylim=(0, None))
    axis.grid(axis="y", color=RULE, alpha=.75)
    axis.legend(frameon=False, loc="best")
    style(axis)
    save_plot(figure, ASSETS / "sampling.pdf")


def robustness_plot(paired):
    figure, axis = plt.subplots(figsize=(6.5, 3.05), layout="constrained")
    tags = list(CONDITIONS)[1:]
    for index, tag in enumerate(tags):
        selected = paired[paired["condition"] == tag].set_index("fold")
        values = selected.loc[list(FOLDS), "delta_mean_ade"]
        color = RUST if tag.startswith("noise") else TEAL
        axis.scatter(values, index + np.linspace(-.17, .17, len(FOLDS)),
                     facecolor="white", edgecolor=color, s=35, zorder=3)
        axis.scatter(selected.loc["macro", "delta_mean_ade"], index,
                     color=color, marker="D", s=50, zorder=4)
    axis.axvline(0, lw=1, color=MUTED)
    axis.set(xlabel="Change in meanADE from clean (m)", yticks=range(len(tags)),
             yticklabels=[CONDITIONS[tag] for tag in tags], ylim=(len(tags) - .45, -.55))
    axis.grid(axis="x", color=RULE, alpha=.75)
    style(axis)
    save_plot(figure, ASSETS / "paired_robustness.pdf")


def example_indices(fold, records, window_metrics):
    manifest = pd.DataFrame(records)
    manifest["target_id"] = pd.to_numeric(manifest["target_id"])
    ordered = manifest.sort_values(["scene_name", "scene_t", "target_id"])
    selected = {}
    for _, scene in ordered.groupby("scene_name", sort=False):
        for rank, position in enumerate(sorted({0, len(scene) // 2, len(scene) - 1})):
            selected[f"fixed_{rank}_{scene.iloc[position]['scene_name']}"] = int(scene.index[position])
    failure_id = window_metrics.sort_values(["min_ade", "window_id"], ascending=[False, True]).iloc[0]["window_id"]
    selected[f"worst_clean_min_ade_{fold}"] = int(manifest.index[manifest["window_id"] == failure_id][0])
    return selected


def heading_change(observed, truth):
    before = observed[-1] - observed[0]
    after = truth[-1] - observed[-1]
    denom = np.linalg.norm(before) * np.linalg.norm(after)
    if denom <= 1e-6:
        return None
    cosine = np.clip((before @ after) / denom, -1., 1.)
    return float(np.rad2deg(np.arccos(cosine)))


def draw_trajectory(axis, data, title):
    observed, mask, truth, samples, best, clean_anchor = data
    axis.plot(*observed[mask].T, "o-", color=INK, lw=1.4, ms=2.7, label="History", zorder=4)
    if not mask.all():
        filled = observed[int(mask.sum()) - 1:]
        axis.plot(*filled.T, "o--", color=MUTED, markerfacecolor="white", lw=1.4, ms=3, zorder=5)
    anchor = observed[-1:]
    for index, sample in enumerate(samples):
        if index != best:
            axis.plot(*np.concatenate((anchor, sample)).T, color=TEAL, alpha=.22, lw=1.1, zorder=1)
    axis.plot(*np.concatenate((anchor, samples[best])).T, color=TEAL, lw=1.8,
              linestyle="--", label="Best of 20", zorder=2)
    axis.plot(*np.concatenate((clean_anchor, truth)).T, color=RUST, lw=1.9, label="Observed future", zorder=3)
    axis.set(title=title, xlabel="x (m)", ylabel="y (m)", aspect="equal")
    style(axis)


def match_axes(axes):
    axes = tuple(axes)
    xlim = [axis.get_xlim() for axis in axes]
    ylim = [axis.get_ylim() for axis in axes]
    for axis in axes:
        axis.set_xlim(min(left for left, _ in xlim), max(right for _, right in xlim))
        axis.set_ylim(min(bottom for bottom, _ in ylim), max(top for _, top in ylim))


def trajectory_examples():
    rows = []
    main = {}
    for fold in FOLDS:
        seed = 0
        with np.load(bundle_path(fold, seed, "clean_0"), allow_pickle=False) as bundle:
            records = [json.loads(record) for record in bundle["records"]]
            clean_observed = bundle["observed"]
        window_metrics = pd.read_csv(metrics_path(fold, seed, "clean_0") / "window_metrics.csv")
        chosen = example_indices(fold, records, window_metrics)
        indices = list(chosen.values())
        origins = pd.DataFrame(records).groupby(["scene_name", "scene_t"]).size()
        panels = {}
        clean_headings = {}
        for tag in CONDITIONS:
            with np.load(bundle_path(fold, seed, tag), allow_pickle=False) as bundle:
                observations, masks, truths, paths = (bundle[key][indices] for key in
                                                       ("observed", "observed_mask", "ground_truth", "samples"))
            for j, (selection, index) in enumerate(chosen.items()):
                record = records[index]
                history, mask, truth, samples = observations[j], masks[j], truths[j], paths[j]
                ades = np.linalg.norm(samples.astype(np.float64) - truth, axis=-1).mean(axis=-1)
                best = int(ades.argmin())
                panels[selection, tag] = history, mask, truth, samples, best, clean_observed[index, -1:]
                if tag == "clean_0":
                    clean_headings[selection] = heading_change(history, truth)
                rows.append({
                    "fold": fold, "seed": seed, "selection": selection, "condition": tag,
                    "window_id": record["window_id"], "scene_name": record["scene_name"],
                    "scene_t": record["scene_t"], "target_id": record["target_id"],
                    "eligible_targets_at_origin": origins.loc[(record["scene_name"], record["scene_t"])],
                    "clean_history_to_future_heading_deg": clean_headings[selection],
                    "min_ade": ades[best], "mean_ade": ades.mean(), "best_sample_zero_based": best,
                })
        # Keep the complete qualitative grid with the experiment, apart from the two small report figures.
        for selection, index in chosen.items():
            figure, axes = plt.subplots(2, 3, figsize=(11, 6.4), layout="constrained")
            figure.suptitle(f"{FOLDS[fold]}, seed 0, window {records[index]['window_id']}")
            for axis, tag in zip(axes.flat, CONDITIONS):
                draw_trajectory(axis, panels[selection, tag], CONDITIONS[tag])
            match_axes(axes.flat)
            axes.flat[0].legend(frameon=False, fontsize=8)
            name = f"{fold}_{list(chosen).index(selection):02d}.pdf"
            save_plot(figure, QUALITATIVE / name)
        if fold == "eth":
            key = next(selection for selection in chosen if selection.startswith("fixed_1_"))
            main["ETH fixed window, clean"] = panels[key, "clean_0"]
            main["ETH same window, 0.20 m noise"] = panels[key, "noise_020"]
        if fold == "univ":
            main["Univ worst clean minADE"] = panels["worst_clean_min_ade_univ", "clean_0"]
    table = save_table(rows, "trajectory_examples")
    figure, axes = plt.subplots(1, 3, figsize=(9.3, 3.35), layout="constrained")
    for axis, (title, panel) in zip(axes, main.items()):
        draw_trajectory(axis, panel, title)
    match_axes(axes[:2])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False, ncols=3, fontsize=9)
    save_plot(figure, ASSETS / "trajectory_examples.pdf")
    return table


def write_tex(clean, paired, budgets, runtime, diagnostics, examples):
    by_fold = clean.set_index("fold")
    macro_pairs = paired[paired["fold"] == "macro"].set_index("condition")
    budget = budgets.set_index("sample_count")
    motion = diagnostics.set_index("condition")
    failure = examples[(examples["selection"] == "worst_clean_min_ade_univ") & (examples["condition"] == "clean_0")].iloc[0]
    fixed = examples[(examples["fold"] == "eth") & examples["selection"].str.startswith("fixed_1_")].set_index("condition")
    values = {
        "OneSampleADE": f"{budget.loc[1, 'min_ade']:.3f}",
        "BudgetReduction": f"{100 * (1 - budget.loc[20, 'min_ade'] / budget.loc[1, 'min_ade']):.0f}",
        "MaxSeedSD": f"{by_fold.loc[list(FOLDS), 'min_ade_seed_sd'].max():.3f}",
        "NoiseMaxFoldDelta": f"{macro_pairs.loc['noise_020', 'delta_mean_ade_fold_max']:+.3f}",
        "NoiseMinFoldDelta": f"{macro_pairs.loc['noise_020', 'delta_mean_ade_fold_min']:+.3f}",
        "UnivRuntime": f"{runtime.loc[runtime['fold'] == 'univ', 'end_to_end_ms'].median():.1f}",
        "CleanSpeedViolationPct": f"{100 * motion.loc['clean_0', 'speed_violation_fraction']:.1f}",
        "CleanCloseApproachPct": f"{100 * motion.loc['clean_0', 'close_approach_fraction']:.1f}",
        "FixedCleanMinADE": f"{fixed.loc['clean_0', 'min_ade']:.3f}",
        "FixedNoiseMinADE": f"{fixed.loc['noise_020', 'min_ade']:.3f}",
        "FailureMinADE": f"{failure['min_ade']:.3f}",
        "FailureCoTargets": str(failure["eligible_targets_at_origin"]),
    }
    failure_heading = failure["clean_history_to_future_heading_deg"]
    values["FailureHeading"] = f"{failure_heading:.0f}" if pd.notna(failure_heading) else "unavailable"
    for fold, title in FOLDS.items():
        for metric, label in (("min_ade", "MinADE"), ("min_fde", "MinFDE"), ("mean_ade", "MeanADE"),
                              ("mean_fde", "MeanFDE"), ("energy_score", "Energy"), ("sample_diversity", "Diversity")):
            values[title + label] = f"{by_fold.loc[fold, f'{metric}_mean']:.3f}"
    lines = ["% Generated by models/social_gan/handoff.py from audited results.",
             r"\providecommand{\SocialGanInsight}[1]{\csname social-gan:#1\endcsname}"]
    lines.extend(rf"\expandafter\def\csname social-gan:{key}\endcsname{{{value}}}" for key, value in values.items())
    (CONTRIBUTION / "analysis.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(results):
    for directory in (EVIDENCE, ASSETS, QUALITATIVE):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    clean, paired = clean_and_robustness(results)
    budgets = sample_budget(results)
    runtime, environments = runtime_table(results)
    diagnostics = motion_spatial(results)
    examples = trajectory_examples()
    sampling_plot(budgets)
    robustness_plot(paired)
    write_tex(clean, paired, budgets, runtime, diagnostics, examples)
    return {key: median(runtime[key]) for key in RUNTIME}, environments
