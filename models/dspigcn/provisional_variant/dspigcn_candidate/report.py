"""Generate the DSPIGCN report section using only saved experimental evidence."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np

from .aggregate import aggregate
from .runner import save_json


def generate(root, destination, report_path):
    root, destination, report_path = Path(root), Path(destination), Path(report_path)
    destination.mkdir(parents=True, exist_ok=True)
    with (root / "results.csv").open() as f:
        results = list(csv.DictReader(f))
    summary = aggregate([root / "results.csv"], destination / "seed_summary.csv")
    shutil.copy2(root / "results.csv", destination / "results.csv")
    clean = [r for r in summary if r["condition"] == "clean"]
    relative_results = Path(os.path.relpath(destination, report_path.parent)).as_posix()
    text = ["# Zhengyang Yan — DSPIGCN report section", "",
            "**Status: candidate-protocol experiments; not yet a verified group comparison.**",
            "The package implements the assigned data preparation, dual-stream model, training,",
            "clean/perturbed evaluation, runtime measurement and visualization workflow. The",
            "group contract and full paper have now been inspected. This candidate differs",
            "from causal_v2 and the paper-based registered implementation. Its scores must",
            "remain separate. See TEAM_INTEGRATION.md and PAPER_CHECK.md for details.", "",
            "## Method", "",
            "The two temporal-attention/graph-convolution streams process observed displacement",
            "and velocity. Gaussian heads forecast 12 steps from 8 observations, at dt=0.4 s.",
            "A fourth-order polynomial head couples integrated displacement and analytic",
            "velocity; reconstruction, two Gaussian likelihood terms and physics residuals",
            "are optimized jointly. Five train-only Gaussian-mixture anchors generate 20",
            "sampled futures. This is a documented independent DSPIGCN variant; the GMM",
            "sampler is a project variant, distinct from the paper\'s Algorithm 2.", "",
            "All eligible continuous 20-frame trajectories, including nonlinear and singleton",
            "cases, are retained. Training/validation are chronological, disjoint before",
            "window generation, with a boundary gap. Checkpoint selection uses validation",
            "mean-trajectory ADE. No test examples select hyperparameters or anchors.", "",
            "## Clean accuracy", "",
            "Errors below are meters; ± is sample standard deviation over training seeds.",
            "MinADE and minFDE select whole samples independently per pedestrian. Mean",
            "errors use every sample. A blank deviation means only one seed was available.", "",
            "| Scene | Seeds | minADE | minFDE | meanADE | meanFDE | Energy score | Diversity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def number(row, metric):
        value, spread = row[metric + "_mean"], row[metric + "_std"]
        return f"{value:.3f}" + (f" ± {spread:.3f}" if spread != "" else "")
    for row in clean:
        text.append("| " + " | ".join([row["scene"], str(row["seed_count"])] + [number(row, key) for key in
                    ("minADE", "minFDE", "meanADE", "meanFDE", "energy_score", "diversity")]) + " |")
    text += ["", "The gap between minimum and mean errors quantifies selection benefit under",
             "the 20-sample budget. It does not by itself establish uncertainty calibration.", "",
             "## Robustness", "",
             f"![Robustness across conditions]({relative_results}/robustness.png)", "",
             "| Scene | Clean meanADE | Noise 0.20 m meanADE | Missing 2 meanADE |",
             "|---|---:|---:|---:|"]
    for row in clean:
        selected = {r["condition"]: r for r in summary if r["scene"] == row["scene"]}
        text.append(f"| {row['scene']} | {number(row, 'meanADE')} | {number(selected['noise_0.20'], 'meanADE')} | {number(selected['missing_2'], 'meanADE')} |")
    text += ["", "Noise and missing suffixes are evaluated separately against unchanged future",
             "targets. Corruptions use sample-addressed PCG64 draws; missing points use only",
             "the last two visible observations. The constant-velocity residual uses the last",
             "observed increment, so sensitivity to that increment is an architectural",
             "hypothesis; these experiments do not isolate its causal contribution.", "",
             "## Motion consistency", "",
             "| Scene | Mean speed (m/s) | Mean acceleration (m/s²) | Mean jerk (m/s³) | Close-pair rate |",
             "|---|---:|---:|---:|---:|"]
    for row in clean:
        text.append("| " + " | ".join([row["scene"]] + [number(row, key) for key in
                    ("speed_mean", "acceleration_mean", "jerk_mean", "close_approach_rate")]) + " |")
    text += ["", "Exceedance thresholds are estimated from training data and saved per fold.",
             "Close-pair rate is a simultaneous-pair diagnostic with an explicit denominator,",
             "not a collision-avoidance guarantee. The GMM samples are not a learned joint",
             "distribution over all pedestrians. Physical penalties constrain mean predictions;",
             "they do not guarantee that every sampled trajectory is physically plausible.", "",
             "## Selected failures", ""]
    failures, runtime, training = [], [], []
    for row in clean:
        run = root / row["scene"] / "seed_0"
        failure_rows = json.loads((run / "evaluation/clean/failures.json").read_text())
        failure = failure_rows[0]
        idx = failure["window_index"]
        with zipfile.ZipFile(run / "evaluation/clean/predictions.zip") as archive:
            with np.load(io.BytesIO(archive.read(f"{idx:06d}.npz")), allow_pickle=False) as data:
                trajectory = np.concatenate((data["observations"][-2:], data["target"]))
                velocity = np.diff(trajectory, axis=0)
                norm = np.linalg.norm(velocity, axis=-1)
                cosine = (velocity[1:] * velocity[:-1]).sum(-1) / np.maximum(norm[1:] * norm[:-1], 1e-8)
                valid = (norm[1:] > .02) & (norm[:-1] > .02)
                angle = np.degrees(np.arccos(np.clip(cosine[valid], -1, 1))).max(initial=0)
                failures.append({"scene": row["scene"], **failure, "pedestrians": data["target"].shape[1],
                                 "largest_target_heading_change_degrees": float(angle)})
        figure = run / f"evaluation/clean/failure_{idx}.png"
        shutil.copy2(figure, destination / f"{row['scene']}_failure.png")
        text.append(f"- {row['scene']}: `{failure['scene_id']}`, meanADE {failure['meanADE']:.3f} m; "
                    f"{failures[-1]['pedestrians']} simultaneous eligible people, largest target heading change "
                    f"{angle:.1f}°. [Trajectory figure]({relative_results}/{row['scene']}_failure.png).")
        for seed in range(row["seed_count"]):
            path = root / row["scene"] / f"seed_{seed}" / "runtime.json"
            timing = json.loads(path.read_text())
            runtime.append({"scene": row["scene"], "seed": seed, **timing})
            run_path = root / row["scene"] / f"seed_{seed}"
            metadata = json.loads((run_path / "run.json").read_text())
            with (run_path / "history.csv").open() as f:
                history = list(csv.DictReader(f))
            best = min(history, key=lambda entry: float(entry["val_mean_trajectory_ADE"]))
            training.append({"scene": row["scene"], "seed": seed,
                             "completed_epochs": metadata["completed_epochs"], "selected_epoch": int(best["epoch"]),
                             "training_seconds": metadata["training_seconds"],
                             "preprocessing_seconds": metadata["preprocessing_seconds"],
                             "parameters": metadata["parameters"]})
    text += ["", "These are worst windows by meanADE, not examples chosen to favor the model.",
             "Heading change and eligible crowd size describe observed failure context; they",
             "do not prove that turning or interaction density caused the error.", "",
             "## Runtime and reproducibility", "",
             "Runtime records use 32 pedestrians, 20 samples, ten warmups and 100 repeats.",
             "Graph construction is inside model prediction time; preprocessing and sampling",
             "are separate. See `runtime.json` for hardware, device and measurement boundaries.",
             "Entries below average each seed's median measurement; they are not speedups",
             "relative to another model. Training seconds exclude GMM fitting and data preparation.", "",
             "| Scene | Preprocess (ms) | Predict (ms) | Sample (ms) | Mean train (s) | Selected epochs |",
             "|---|---:|---:|---:|---:|---|"]
    for scene in sorted({r["scene"] for r in runtime}):
        time_rows = [r for r in runtime if r["scene"] == scene]
        train_rows = [r for r in training if r["scene"] == scene]
        measurements = [np.mean([r[key]["median"] for r in time_rows]) for key in
                        ("preprocessing_ms", "prediction_ms", "sampling_ms")]
        seconds = np.mean([r["training_seconds"] for r in train_rows])
        epochs = ", ".join(str(r["selected_epoch"]) for r in train_rows)
        text.append(f"| {scene} | " + " | ".join(f"{v:.3f}" for v in measurements) + f" | {seconds:.2f} | {epochs} |")
    text += ["",
             "Dependencies, seeds, parameter settings, validation history, population manifests,",
             "checkpoints and exported predictions are retained in each original run directory.", "",
             f"Saved results contain {len(results)} rows across {len(clean)} scenes. The audit must",
             "pass for all expected 15 runs and 90 conditions before treating this candidate",
             "matrix as complete. Matching the team's data and protocol is a separate gate.", "",
             "## Limitations and outstanding integration", "",
             "1. Source-code-guided implementation differs from the release in velocity input,",
             "   physics coupling, reconstruction loss, tensor layouts and GMM sampling.",
             "   Full-text review confirms these differences; see PAPER_CHECK.md.",
             "2. The exact GMM sampling distribution is not the Gaussian training likelihood.",
             "3. The provisional split, context selection and perturbations differ from causal_v2;",
             "   no cross-model numerical superiority claim is supported yet.",
             "4. Existing model code is preserved. The group-batch inference adapter provides",
             "   tensor interoperability, not retrospective benchmark comparability.", "",
             "References and AI coding assistance are acknowledged in THIRD_PARTY.md.", ""]
    save_json(destination / "failure_analysis.json", failures)
    save_json(destination / "runtime.json", runtime)
    save_json(destination / "training.json", training)
    audit_path = destination / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        if audit.get("passed") and audit.get("runs") == 15 and audit.get("conditions") == 90:
            text += ["## Completed local audit", "",
                     "All 15 runs / 90 condition rows passed the saved-artifact audit, including",
                     "raw-source trajectory matching, corruptions, per-trajectory and aggregate",
                     "accuracy/motion scores, checkpoint selection, runtime settings and figures.",
                     f"See [audit evidence]({relative_results}/audit.json). This does not verify the",
                     "candidate's equivalence to the group protocol or the paper equations.", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(text))
    plot_robustness(summary, destination / "robustness.png")


def plot_robustness(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions = ["clean", "noise_0.05", "noise_0.10", "noise_0.20", "missing_1", "missing_2"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for scene in sorted({r["scene"] for r in summary}):
        rows = {r["condition"]: r for r in summary if r["scene"] == scene}
        values = [rows[c]["meanADE_mean"] for c in conditions]
        std = [rows[c]["meanADE_std"] or 0 for c in conditions]
        ax.errorbar(range(6), values, yerr=std, marker="o", capsize=3, label=scene)
    ax.set_xticks(range(6), conditions, rotation=15)
    ax.set(ylabel="Mean ADE (m)", title="DSPIGCN candidate protocol — mean ± seed SD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    generate(args.runs, args.destination, args.report)


if __name__ == "__main__":
    main()
