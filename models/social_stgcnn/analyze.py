"""Extract formal test evidence and render the Social-STGCNN contribution figures.

Run the numeric phase after `make score`. The render phase reads only the
small evidence tables, so it can run on a separate machine with Matplotlib.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "benchmark/fixtures/runtime_windows.json"
sys.path.insert(0, str(ROOT))

SCORES = ["min_ade", "min_fde", "mean_ade", "mean_fde", "energy_score", "sample_diversity"]
MOTION = ["speed_mean", "acceleration_mean", "jerk_mean",
          "speed_violation_rate", "acceleration_violation_rate", "jerk_violation_rate"]
BUDGETS = (1, 5, 10, 20)
NEIGHBOR_BINS = ("0", "1-3", "4-7", "8+")
STAGES = ["preprocessing_ms", "prediction_sampling_ms", "end_to_end_ms"]
CONTRIBUTION = ROOT / "submission/contributions/social_stgcnn"

# Fixed before inspecting formal forecast errors. Index 64 is zero-based.
CASE_SEED = 0
CASE_INDEX = 64
FIXED_CASES = {"eth": "eth:biwi_eth:171:801", "univ": "univ:students001:101:156"}
FAILURE_RULE = (
    "Highest clean minADE@20 among all five held-out scenes at seed 0. "
    "Break exact ties by lexicographically smallest window ID."
)


def scored_windows(folder, window_ids):
    """Align each condition by window ID before taking paired differences."""
    columns = ["window_id", *SCORES, *MOTION]
    frame = pd.read_csv(folder / "window_metrics.csv", usecols=columns, index_col="window_id")
    if not frame.index.equals(window_ids):
        if (len(frame) != len(window_ids) or not frame.index.is_unique
                or not window_ids.isin(frame.index).all()):
            raise ValueError(f"Scored target IDs differ from the test manifest in {folder}")
        frame = frame.loc[window_ids]
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Non-finite score or motion statistic in {folder}")
    return frame


def summarize_runs(rows, keys, values, counts=("unique_windows",), seed_spread=()):
    """Average windows within a run, then seeds and folds equally. Empty bins have no weight."""
    frame = pd.DataFrame(rows)
    runs_by_fold = frame.groupby(["fold", *keys], sort=False)
    fold = runs_by_fold[values].mean()
    fold[list(counts)] = runs_by_fold[list(counts)].first()
    fold["seeds"] = runs_by_fold["seed"].nunique()
    fold["contributing_folds"] = fold[values[0]].notna().astype(int)
    if seed_spread:
        fold = fold.join(runs_by_fold[list(seed_spread)].std().add_suffix("_seed_sd"))

    # Taking a fold macro within each seed also defines the seed SD of the macro.
    per_seed = frame.groupby([*keys, "seed"], sort=False)[values].mean().reset_index()
    equal_folds = per_seed.groupby(keys, sort=False)
    macro = equal_folds[values].mean()
    fold_totals = fold.reset_index().groupby(keys, sort=False)
    macro[list(counts)] = fold_totals[list(counts)].sum()
    macro["seeds"] = equal_folds["seed"].nunique()
    macro["contributing_folds"] = fold_totals["contributing_folds"].sum()
    if seed_spread:
        macro = macro.join(equal_folds[list(seed_spread)].std().add_suffix("_seed_sd"))

    return pd.concat((fold.reset_index().assign(scope="fold"),
                      macro.reset_index().assign(scope="macro", fold="all")), ignore_index=True)


def spatial_diagnostics(folder):
    # The common evaluator emits an empty CSV if no scene/anchor has two targets.
    try:
        spatial = pd.read_csv(folder / "spatial_consistency.csv",
                              usecols=["min_distance_mean", "close_approach_rate"])
    except pd.errors.EmptyDataError:
        spatial = pd.DataFrame(columns=["min_distance_mean", "close_approach_rate"], dtype=float)
    return {"spatial_groups": len(spatial), "min_pair_distance_m": spatial.min_distance_mean.mean(),
            "close_approach_fraction": spatial.close_approach_rate.mean()}


def case_rows(name, rule, window_id, index, fold, counts, clean, observed, future, samples):
    record = {"case": name, "selection_rule": rule, "fold": fold, "seed": CASE_SEED,
              "condition": "clean_0", "k": 20, "window_id": window_id,
              "runtime_fixture_index": CASE_INDEX if name.startswith("fixed_") else None,
              "available_and_used_neighbors": int(counts[index]),
              **clean.loc[window_id, SCORES].to_dict()}
    paths = []
    def append(path_type, draw, points, first_step):
        for step, xy in enumerate(points, start=first_step):
            paths.append({"case": name, "window_id": window_id, "path": path_type, "draw": draw,
                          "step": step, "x_m": float(xy[0]), "y_m": float(xy[1])})
    append("observed", None, observed[index], -7)
    append("truth", None, future[index], 1)
    for draw, points in enumerate(samples[index]):
        append("sample", draw, points, 1)
    return record, paths


def clean_evidence(run, fold, seed, clean, counts, reference):
    from li_junhao_trajectronpp.scripts.analyze_sample_budget import metrics_for_budget_batch

    path = run / "evaluation/clean_0/predictions_test_clean_0.npz"
    with np.load(path, allow_pickle=False) as bundle:
        meta = json.loads(str(bundle["metadata"]))
        if any(reference[name] != meta[name] for name in ("config_hash", "checkpoint_sha256", "manifest_hash")):
            raise ValueError(f"Clean bundle identity differs from the formal results row for {fold}/{seed}")
        samples = bundle["samples"]
        future = bundle["ground_truth"]
        cases = []
        if seed == CASE_SEED:
            observed = bundle["observed"]
            fixed = FIXED_CASES.get(fold)
            if fixed is not None:
                cases.append(case_rows("fixed_" + fold, "Runtime fixture, zero-based index 64",
                                       fixed, clean.index.get_loc(fixed), fold, counts, clean,
                                       observed, future, samples))
            worst = clean.index[clean.min_ade == clean.min_ade.max()].min()
            cases.append(case_rows("failure", FAILURE_RULE, worst, clean.index.get_loc(worst),
                                   fold, counts, clean, observed, future, samples))

    # K=20 has already been scored by the shared evaluator. Only shorter
    # prefixes require new distances; reuse its exact Energy Score definition.
    budgets = []
    for k in BUDGETS:
        if k == 20:
            means = clean[SCORES].mean().to_dict()
        else:
            scores = metrics_for_budget_batch(samples[:, :k].astype(np.float64),
                                              future.astype(np.float64), k, batch_size=256)
            means = {key: values.mean() for key, values in scores.items()}
        budgets.append({"fold": fold, "seed": seed, "k": k, "unique_windows": len(clean), **means})
    return budgets, cases


def runtime_row(run, fold, seed, host_labels):
    timing = json.loads((run / "runtime.json").read_text())
    settings = (timing["targets"], timing["batch_size"], timing["sample_count"], timing["warmups"],
                timing["measured_repetitions"], len(timing["repetitions"]))
    if settings != (128, 32, 20, 3, 10, 10):
        raise ValueError(f"Unexpected official timing population or repetition count in {run}")
    environment = timing["environment"]
    host = environment["host"]
    host_label = host_labels.setdefault(host, f"Host {len(host_labels) + 1}")
    public = ("os", "architecture", "cpu", "threads", "python", "numpy", "torch", "device", "gpu", "cuda", "driver")
    setup = {"fold": fold, "seed": seed, "host": host_label, "batch_size": 32, "k": 20, "condition": "clean_0",
             **{field: environment.get(field) for field in public}}
    primary = {**setup, "scope": "run", "targets_per_run": 128, "warmups": 3,
               "measured_repetitions_per_run": 10, "runs": 1,
               **{stage: timing["medians"][stage] for stage in STAGES}}
    density = []
    if seed == CASE_SEED:
        for cell in timing["density"]:
            n = cell["timed_windows"]
            repetitions = len(cell.get("repetitions", []))
            if (n, repetitions) not in ((0, 0), (32, 10)):
                raise ValueError(f"Unrecognized density timing population in {run}")
            density.append({**setup, "available_neighbors_bin": cell["available_neighbors_bin"],
                            "available_windows": cell["available_windows"], "timed_windows": n,
                            "warmups": 3 if n else 0, "measured_repetitions": repetitions,
                            **{stage: cell["medians"][stage] if n else np.nan for stage in STAGES}})
        if sorted(row["available_neighbors_bin"] for row in density) != sorted(NEIGHBOR_BINS):
            raise ValueError(f"Missing or duplicated density timing bin in {run}")
    return primary, density


def summarize_runtime(rows):
    frame = pd.DataFrame(rows)
    setup = [col for col in frame if col not in ["scope", "fold", "seed", "runs", *STAGES]]
    matched = frame.groupby(setup, sort=False, dropna=False)
    summary = matched[STAGES].median()
    summary["runs"] = matched.size()
    return pd.concat((frame, summary.reset_index().assign(scope="host_median_of_run_medians",
                                                          fold="all", seed=None)), ignore_index=True)


def numeric_tables(runs, metrics, data, results):
    from li_junhao_trajectronpp.independent_tpp.protocol import CONDITIONS, FOLDS, SEEDS
    from models.social_stgcnn.windows import GraphWindows

    runtime_ids = json.loads(FIXTURE.read_text())
    if any(runtime_ids[fold][CASE_INDEX] != window_id for fold, window_id in FIXED_CASES.items()):
        raise ValueError("The preselected examples no longer match the registered runtime fixture")
    reference = pd.read_csv(results).set_index(["fold", "seed", "condition"], verify_integrity=True)
    scores, diagnostics, budgets, neighbors, runtime, density, cases = [], [], [], [], [], [], []
    host_labels, thresholds = {}, {}
    for fold in FOLDS:
        windows = GraphWindows.from_split(fold, "test", data)
        ids = pd.Index([row.window_id for row in windows.records], name="window_id")
        counts = windows.available_neighbors
        bins = np.digitize(counts, [1, 4, 8])
        for seed in SEEDS:
            run = runs / fold / f"seed_{seed}"
            score_root = metrics / fold / f"seed_{seed}"
            clean = scored_windows(score_root / "clean_0", ids)
            if seed == CASE_SEED:
                thresholds[fold] = json.loads((score_root / "clean_0/summary.json").read_text())["thresholds"]
            for tag, _, _ in CONDITIONS:
                folder = score_root / tag
                frame = clean if tag == "clean_0" else scored_windows(folder, ids)
                base = {"fold": fold, "seed": seed, "condition": tag, "unique_windows": len(frame)}
                means = frame[SCORES].mean()
                paired = (frame[SCORES] - clean[SCORES]).mean()
                scores.append({**base, **means.to_dict(), **paired.add_prefix("delta_").to_dict()})
                diagnostics.append({**base, **frame[MOTION].mean().to_dict(), **spatial_diagnostics(folder)})
            for number, name in enumerate(NEIGHBOR_BINS):
                subset = clean.loc[bins == number, SCORES]
                neighbors.append({"fold": fold, "seed": seed, "available_neighbors_bin": name,
                                  "unique_windows": len(subset), **subset.mean().to_dict()})
            values, examples = clean_evidence(run, fold, seed, clean, counts, reference.loc[(fold, seed, "clean_0")])
            budgets.extend(values)
            cases.extend(examples)
            primary, strata = runtime_row(run, fold, seed, host_labels)
            runtime.append(primary)
            density.extend(strata)
        print(f"Analyzed {fold}: {len(ids)} unique test windows", flush=True)

    score_keys = [*SCORES, *("delta_" + key for key in SCORES)]
    scores = summarize_runs(scores, ["condition"], score_keys, seed_spread=SCORES)
    diagnostic_keys = [*MOTION, "min_pair_distance_m", "close_approach_fraction"]
    diag = summarize_runs(diagnostics, ["condition"], diagnostic_keys, counts=("unique_windows", "spatial_groups"))
    present = diag.loc[(diag.scope == "fold") & (diag.spatial_groups > 0)].groupby("condition").size()
    diag["spatial_contributing_folds"] = np.where(diag.scope == "fold", (diag.spatial_groups > 0).astype(int),
                                                  diag.condition.map(present).fillna(0).astype(int))
    for field in ("speed", "acceleration", "jerk", "close_approach_distance"):
        diag[field + "_training_threshold"] = diag["fold"].map({f: val[field] for f, val in thresholds.items()})
    diag["motion_scope"] = ("Forecast sample/frame speed (m/s), acceleration (m/s^2) and jerk (m/s^3). "
                            "Compare the within-window p99 with each fold's training threshold. "
                            "Rates are fractions of target windows.")
    diag["spatial_scope"] = ("Scene and anchor groups with at least two forecast targets. For each draw "
                             "index use the minimum distance over all target pairs and future frames. "
                             "Targets are sampled separately. Rate is the fraction of draws below the "
                             "fold's training distance, averaged over groups.")
    selected = [case for case in cases if case[0]["case"] != "failure"]
    selected.append(min((case for case in cases if case[0]["case"] == "failure"),
                        key=lambda case: (-case[0]["min_ade"], case[0]["window_id"])))
    return {
        "scene_scores": scores.loc[scores.scope == "fold"], "macro_scores": scores.loc[scores.scope == "macro"],
        "sample_budget": summarize_runs(budgets, ["k"], SCORES).assign(condition="clean_0"),
        "neighbors": summarize_runs(neighbors, ["available_neighbors_bin"], SCORES).assign(
            condition="clean_0", neighbor_scope="Other pedestrians visible in all eight frames. All enter the graph."),
        "diagnostics": diag, "runtime": summarize_runtime(runtime), "runtime_density": pd.DataFrame(density),
        "case_selection": pd.DataFrame([record for record, _ in selected]),
        "case_paths": pd.DataFrame([row for _, paths in selected for row in paths]),
    }


FOLD_NAMES = {"eth": "ETH", "hotel": "Hotel", "univ": "Univ", "zara1": "Zara1", "zara2": "Zara2"}
CONDITION_NAMES = {"noise_005": "Noise 0.05 m", "noise_010": "Noise 0.10 m",
                   "noise_020": "Noise 0.20 m", "missing_1": "Missing 1 step", "missing_2": "Missing 2 steps"}
INK, TEAL, RUST, GRID = "#172D3A", "#007465", "#AD482B", "#CFDEDE"


def apply_figure_style(plt):
    """Use the shared document palette, embedded fonts and an open axis frame."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "out", "ytick.direction": "out",
        "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def save_figure(plt, fig, path):
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def render_sampling(plt, evidence, assets):
    scenes = pd.read_csv(evidence / "scene_scores.csv")
    scenes = scenes.loc[scenes.condition == "clean_0"].set_index("fold").loc[list(FOLD_NAMES)]
    prefixes = pd.read_csv(evidence / "sample_budget.csv")
    prefixes = prefixes.loc[prefixes.scope == "macro"].sort_values("k")
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.05, 2.75))
    fig.subplots_adjust(left=.092, right=.978, top=.80, bottom=.19, wspace=.47)
    y = np.arange(len(scenes))
    left.hlines(y, scenes.min_ade, scenes.mean_ade, color=GRID, linewidth=2.5)
    left.scatter(scenes.min_ade, y, color=TEAL, s=25, zorder=3)
    left.scatter(scenes.mean_ade, y, color=RUST, marker="s", s=23, zorder=3)
    left.set(yticks=y, yticklabels=[FOLD_NAMES[name] for name in scenes.index],
             xlabel="Clean ADE at 20 draws (m)", ylim=(len(scenes) - .6, -.4))
    left.set_xlim(left=0)
    left.grid(axis="x", color=GRID, linewidth=.55)
    left.set_title("Five held-out scenes", loc="left")
    mean_line, = right.plot(prefixes.k, prefixes.mean_ade, color=RUST, marker="s", markersize=4)
    min_line, = right.plot(prefixes.k, prefixes.min_ade, color=TEAL, marker="o", markersize=4)
    right.set(xticks=list(BUDGETS), xlabel="Number of draws scored", ylabel="Clean ADE (m)")
    right.set_ylim(bottom=0)
    right.grid(axis="y", color=GRID, linewidth=.55)
    right.set_title("Equal-scene average", loc="left")
    fig.legend([min_line, mean_line], ["Best draw", "Mean over draws"],
               loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=2)
    save_figure(plt, fig, assets / "clean_sampling.pdf")


def render_robustness(plt, evidence, assets):
    scenes = pd.read_csv(evidence / "scene_scores.csv").set_index(["condition", "fold"])
    macro = pd.read_csv(evidence / "macro_scores.csv").set_index("condition")
    fig, ax = plt.subplots(figsize=(6.35, 2.65))
    fig.subplots_adjust(left=.22, right=.965, bottom=.22, top=.88)
    ax.axvline(0, color=INK, linewidth=.8)
    jitter = np.linspace(-.18, .18, len(FOLD_NAMES))
    for pos, tag in enumerate(CONDITION_NAMES):
        values = scenes.loc[tag].loc[list(FOLD_NAMES), "delta_mean_ade"]
        points = ax.scatter(values, pos + jitter, facecolor="white", edgecolor=TEAL,
                            s=25, linewidth=.9, zorder=3)
        average = ax.scatter(macro.loc[tag, "delta_mean_ade"], pos, marker="D",
                             color=RUST, s=28, zorder=4)
    ax.set(yticks=np.arange(len(CONDITION_NAMES)), yticklabels=list(CONDITION_NAMES.values()),
           xlabel="Paired change in mean ADE from clean (m)", ylim=(len(CONDITION_NAMES) - .5, -.5))
    ax.margins(x=.075)
    ax.grid(axis="x", color=GRID, linewidth=.55)
    fig.legend([points, average], ["Held-out scene", "Equal-scene average"],
               loc="upper center", bbox_to_anchor=(.56, 1.02), ncol=2)
    save_figure(plt, fig, assets / "paired_robustness.pdf")


def render_cases(plt, evidence, assets):
    selection = pd.read_csv(evidence / "case_selection.csv").set_index("case")
    paths = pd.read_csv(evidence / "case_paths.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.85))
    fig.subplots_adjust(left=.092, right=.98, bottom=.22, top=.77, wspace=.32)
    for ax, case in zip(axes, ("fixed_eth", "fixed_univ", "failure")):
        example = paths.loc[paths["case"] == case]
        observed = example.loc[example.path == "observed"].sort_values("step")[["x_m", "y_m"]].to_numpy()
        origin = observed[-1].copy()
        observed = observed - origin
        for _, group in example.loc[example.path == "sample"].groupby("draw", sort=True):
            positions = np.vstack((origin, group.sort_values("step")[["x_m", "y_m"]])) - origin
            draw_line, = ax.plot(*positions.T, color=TEAL, alpha=.37, linewidth=.8)
        truth = example.loc[example.path == "truth"].sort_values("step")[["x_m", "y_m"]].to_numpy()
        truth = np.vstack((origin, truth)) - origin
        obs_line, = ax.plot(*observed.T, color=INK, linewidth=1.8, zorder=4)
        truth_line, = ax.plot(*truth.T, color=RUST, linestyle="--", linewidth=1.8, zorder=4)
        ax.scatter(0, 0, color=INK, s=12, zorder=5)
        all_points = example[["x_m", "y_m"]].to_numpy() - origin
        low, high = all_points.min(axis=0), all_points.max(axis=0)
        center = (high + low) / 2
        radius = .56 * max((high - low).max(), .5)
        ax.set(xlim=(center[0] - radius, center[0] + radius),
               ylim=(center[1] - radius, center[1] + radius), aspect="equal")
        ax.locator_params(nbins=4)
        kind = "highest minADE" if case == "failure" else "fixed window"
        ax.set_title(f"{FOLD_NAMES[selection.loc[case, 'fold']]} · {kind}", loc="left")
    axes[0].set_ylabel("World y offset (m)")
    fig.supxlabel("World x offset from final observation (m)", fontsize=9, y=.045)
    fig.legend([obs_line, truth_line, draw_line], ["Observed target", "True future", "20 sampled futures"],
               loc="upper center", bbox_to_anchor=(.5, 1.015), ncol=3)
    save_figure(plt, fig, assets / "trajectories.pdf")


def render(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_figure_style(plt)
    evidence, assets = out / "evidence", out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    render_sampling(plt, evidence, assets)
    render_robustness(plt, evidence, assets)
    render_cases(plt, evidence, assets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    artifact = ROOT / "artifacts/social_stgcnn"
    parser.add_argument("--phase", choices=("extract", "render", "all"), default="all")
    parser.add_argument("--runs", type=Path, default=artifact / "formal/runs")
    parser.add_argument("--metrics", type=Path, default=artifact / "metrics")
    parser.add_argument("--data", type=Path, default=ROOT / "data/causal_v2")
    parser.add_argument("--results", type=Path, default=CONTRIBUTION / "results.csv",
                        help="Formal result identities written by make score")
    parser.add_argument("--out", type=Path, default=CONTRIBUTION,
                        help="Contribution directory containing evidence/ and assets/")
    args = parser.parse_args()
    if args.phase in ("extract", "all"):
        tables = numeric_tables(args.runs, args.metrics, args.data, args.results)
        destination = args.out / "evidence"
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(destination / f"{name}.csv", index=False)
    if args.phase in ("render", "all"):
        render(args.out)


if __name__ == "__main__":
    main()
