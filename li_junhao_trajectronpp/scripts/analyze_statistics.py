#!/usr/bin/env python3
"""Fold-level statistical summaries for the formal Trajectron++ results.

The formal table has three training seeds per held-out scene.  This script
keeps the held-out scene/fold as the primary independent unit, averages the
three seeds within each fold, and then reports scene-macro means, seed
variation, bootstrap confidence intervals, and paired clean-vs-perturbed
effects.  It deliberately does not treat overlapping windows as independent
replicates.
"""

from __future__ import absolute_import, division, print_function

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "formal_results.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs")
CONDITIONS = ["clean_0", "noise_005", "noise_010", "noise_020", "missing_1", "missing_2"]
METRICS = [
    "min_ade_mean",
    "min_fde_mean",
    "mean_ade_mean",
    "mean_fde_mean",
    "energy_score_mean",
    "sample_diversity_mean",
]


def percentile_bootstrap(values, rng, iterations):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(int(iterations), len(values)), replace=True)
    means = np.mean(samples, axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_pvalue(values):
    """Exact two-sided sign-flip p-value for a paired mean effect."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    counts = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        counts += int(statistic >= observed - 1e-12)
        total += 1
    return float(counts) / float(total)


def flatten_columns(frame):
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if str(part) != "")
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def build_report(macro, effects, output_path):
    lines = [
        "# Trajectron++ fold-level statistical analysis",
        "",
        "This analysis treats the five held-out scenes as the primary independent",
        "units. The three training seeds are averaged within each fold; seed",
        "variation is retained separately. Window-level rows are not treated as",
        "independent replicates because adjacent windows overlap in time.",
        "",
        "## Scene-macro results",
        "",
        "| condition | metric | mean | std across folds | 95% bootstrap CI | n folds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in macro.iterrows():
        lines.append(
            "| {condition} | {metric} | {mean:.4f} | {std:.4f} | [{low:.4f}, {high:.4f}] | {n} |".format(
                condition=row["condition"],
                metric=row["metric"],
                mean=row["mean"],
                std=row["std"],
                low=row["ci_low"],
                high=row["ci_high"],
                n=int(row["fold_count"]),
            )
        )
    lines.extend([
        "",
        "## Paired perturbation effects",
        "",
        "The effect is condition minus clean within the same held-out fold;",
        "positive values mean larger error/energy score. The exact sign-flip",
        "p-value is descriptive because only five folds are available.",
        "",
        "| comparison | metric | mean delta | median delta | 95% bootstrap CI | exact sign-flip p | n folds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in effects.iterrows():
        lines.append(
            "| {comparison} | {metric} | {mean:.4f} | {median:.4f} | [{low:.4f}, {high:.4f}] | {p:.4f} | {n} |".format(
                comparison=row["comparison"],
                metric=row["metric"],
                mean=row["effect_mean"],
                median=row["effect_median"],
                low=row["ci_low"],
                high=row["ci_high"],
                p=row["sign_flip_p"],
                n=int(row["fold_count"]),
            )
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These intervals quantify variation across the five held-out scenes, not",
        "uncertainty over all possible pedestrian environments. They do not make",
        "a causal claim about any Trajectron++ component, and they do not replace",
        "cross-model paired tests after all six independent implementations use",
        "the same window IDs and evaluator.",
        "",
    ])
    with open(output_path, "w") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    frame = pd.read_csv(args.input)
    required = {"fold", "seed", "condition"}.union(METRICS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Missing columns: {}".format(", ".join(missing)))
    if frame.duplicated(["fold", "seed", "condition"]).any():
        raise ValueError("Duplicate fold/seed/condition rows")
    if sorted(frame["condition"].unique().tolist()) != sorted(CONDITIONS):
        raise ValueError("Unexpected condition set: {}".format(sorted(frame["condition"].unique())))

    # Seed average is computed first; this is the primary fold-level analysis.
    seed_summary = frame.groupby(["fold", "condition"], as_index=False)[METRICS].agg(["mean", "std", "count"])
    seed_summary = flatten_columns(seed_summary.reset_index())
    seed_summary_path = os.path.join(args.output_dir, "formal_statistical_by_fold.csv")
    seed_summary.to_csv(seed_summary_path, index=False)

    fold_values = frame.groupby(["fold", "condition"], as_index=False)[METRICS].mean()
    rng = np.random.RandomState(args.seed)
    macro_rows = []
    for condition in CONDITIONS:
        subset = fold_values[fold_values["condition"] == condition]
        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            low, high = percentile_bootstrap(values, rng, args.bootstrap)
            macro_rows.append({
                "condition": condition,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "ci_low": low,
                "ci_high": high,
                "fold_count": int(len(values)),
            })
    macro = pd.DataFrame(macro_rows)
    macro_path = os.path.join(args.output_dir, "formal_statistical_macro.csv")
    macro.to_csv(macro_path, index=False)

    clean = fold_values[fold_values["condition"] == "clean_0"].set_index("fold")
    effect_rows = []
    for condition in CONDITIONS:
        if condition == "clean_0":
            continue
        current = fold_values[fold_values["condition"] == condition].set_index("fold")
        common_folds = sorted(set(clean.index).intersection(current.index))
        for metric in METRICS:
            deltas = (current.loc[common_folds, metric] - clean.loc[common_folds, metric]).to_numpy(dtype=float)
            low, high = percentile_bootstrap(deltas, rng, args.bootstrap)
            effect_rows.append({
                "comparison": "{} - clean_0".format(condition),
                "condition": condition,
                "metric": metric,
                "effect_mean": float(np.mean(deltas)),
                "effect_median": float(np.median(deltas)),
                "ci_low": low,
                "ci_high": high,
                "sign_flip_p": sign_flip_pvalue(deltas),
                "fold_count": int(len(deltas)),
            })
    effects = pd.DataFrame(effect_rows)
    effects_path = os.path.join(args.output_dir, "formal_statistical_effects.csv")
    effects.to_csv(effects_path, index=False)

    report_path = os.path.join(args.output_dir, "formal_statistical_report.md")
    build_report(macro, effects, report_path)
    metadata = {
        "input": os.path.abspath(args.input),
        "fold_unit": "held_out_scene_fold",
        "seed_handling": "mean within fold; seed variation retained in formal_statistical_by_fold.csv",
        "bootstrap_iterations": int(args.bootstrap),
        "bootstrap_seed": int(args.seed),
        "conditions": CONDITIONS,
        "metrics": METRICS,
        "outputs": [
            os.path.abspath(seed_summary_path),
            os.path.abspath(macro_path),
            os.path.abspath(effects_path),
            os.path.abspath(report_path),
        ],
    }
    with open(os.path.join(args.output_dir, "formal_statistical_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print("STATISTICAL_ANALYSIS_OK folds={} conditions={} metrics={}".format(
        len(fold_values["fold"].unique()), len(CONDITIONS), len(METRICS)
    ))


if __name__ == "__main__":
    main()
