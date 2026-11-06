#!/usr/bin/env python3
"""Validate ready model contributions and assemble the English group PDFs."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
sys.path.insert(0, str(HERE.parent / "benchmark"))
from contract import fingerprint, sha256

CONDITIONS = ("clean_0", "noise_005", "noise_010", "noise_020", "missing_1", "missing_2")
METRICS = ("min_ade_mean", "min_fde_mean", "mean_ade_mean", "mean_fde_mean",
           "energy_score_mean", "sample_diversity_mean")
RUNTIME = ("preprocessing_ms", "prediction_sampling_ms", "end_to_end_ms")
HASH_FIELDS = ("config_hash", "checkpoint_sha256", "prediction_sha256")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contribution(model, benchmark):
    model_id = model["id"]
    folder = HERE / "contributions" / model_id
    for name in ("report.tex", "slides.tex", "speaker_notes.md", "results.csv", "provenance.json"):
        require((folder / name).is_file(), f"{model_id}: missing {name}")
    with (folder / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    folds = benchmark["folds"]
    expected = {(fold, str(seed), tag) for fold in folds for seed in range(3) for tag in CONDITIONS}
    actual = {(row["fold"], row["seed"], row["condition"]) for row in rows}
    require(len(rows) == len(expected) and actual == expected,
            f"{model_id}: expected one row per fold, seed and condition")
    budget = model["sample_count"]
    require(budget in (1, 20), f"{model_id}: register its K=1 or K=20 sample budget")
    for row in rows:
        expected_test = folds[row["fold"]]["test"]
        require(row["protocol"] == benchmark["protocol"]
                and row["manifest_hash"] == expected_test["sha256"]
                and int(row["window_count"]) == expected_test["window_count"],
                f"{model_id}: protocol or test manifest mismatch in {row['fold']}")
        require(int(row["sample_count"]) == budget, f"{model_id}: inconsistent sampling budget")
        require(all(re.fullmatch(r"[0-9a-f]{64}", row[field]) for field in HASH_FIELDS),
                f"{model_id}: missing prediction, checkpoint or configuration hash")
        require(all(math.isfinite(float(row[key])) for key in METRICS), f"{model_id}: non-finite metric")
        if budget == 1:
            require(all(math.isclose(float(row[f"min_{kind}_mean"]), float(row[f"mean_{kind}_mean"]))
                        for kind in ("ade", "fde")), f"{model_id}: inconsistent single-sample errors")
    for fold in folds:
        for seed in range(3):
            run_rows = [row for row in rows if row["fold"] == fold and int(row["seed"]) == seed]
            for field in ("config_hash", "checkpoint_sha256"):
                require(len({row[field] for row in run_rows}) == 1,
                        f"{model_id}: {field} changed across conditions in {fold}/{seed}")

    provenance = read_json(folder / "provenance.json")
    require(provenance["protocol"] == benchmark["protocol"] and provenance["runs"] == 15
            and provenance["prediction_bundles"] == 90, f"{model_id}: invalid provenance summary")
    require(re.fullmatch(r"[0-9a-f]{64}", provenance["artifact"]["sha256"]),
            f"{model_id}: missing release artifact hash")
    runtime = {key: float(provenance["runtime_ms"][key]) for key in RUNTIME}
    require(all(math.isfinite(value) and value >= 0 for value in runtime.values()),
            f"{model_id}: invalid runtime summary")
    macro = {
        (tag, key): statistics.mean(
            statistics.mean(float(row[key]) for row in rows if row["condition"] == tag and row["fold"] == fold)
            for fold in folds
        )
        for tag in CONDITIONS for key in METRICS
    }
    return macro, runtime


def tex_value(kind, key, value):
    return rf"\expandafter\def\csname {kind}:{key}\endcsname{{{value}}}"


def render(models, ready, final, preview, scratch):
    generated = scratch / "generated"
    generated.mkdir()
    report_parts, slide_parts, metrics, table = [], [], [], []
    for model in models:
        model_id = model["id"]
        if model_id not in ready:
            continue
        macro, runtime = ready[model_id]
        report_parts.append(rf"\input{{contributions/{model_id}/report.tex}}")
        slide_parts.append(rf"\input{{contributions/{model_id}/slides.tex}}")
        metrics.extend(tex_value("metric", f"{model_id}:{tag}:{key}", f"{value:.3f}")
                       for (tag, key), value in macro.items())
        metrics.extend(tex_value("runtime", f"{model_id}:{key}", f"{value:.1f}")
                       for key, value in runtime.items())
        metrics.extend(tex_value("delta", f"{model_id}:{tag}",
                                 f"{macro[tag, 'mean_ade_mean'] - macro['clean_0', 'mean_ade_mean']:+.3f}")
                       for tag in CONDITIONS)
        fields = ("min_ade_mean", "min_fde_mean", "mean_ade_mean", "mean_fde_mean", "energy_score_mean")
        row = [model["name"], str(model["sample_count"]), *(f"{macro['clean_0', key]:.3f}" for key in fields)]
        table.append(" & ".join(row) + r" \\")
    if not table:
        table.append(r"\multicolumn{7}{l}{Awaiting audited contributions} \\")
    status = "Final assembly" if final else f"Integration draft: {len(ready)} of {len(models)} model contributions ready"
    if preview:
        status = "Contributor preview: " + preview.replace("_", "-")
    for name, lines in (
        ("model-report", report_parts),
        ("model-slides", slide_parts),
        ("model-metrics", metrics),
        ("comparison-rows", [*table, r"\bottomrule"]),
        ("submission-status", [rf"\newcommand{{\SubmissionStatus}}{{{status}}}"]),
    ):
        (generated / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_tex(job, scratch):
    latex = ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", f"{job}.tex"]
    commands = [latex, ["bibtex", "report"], latex, latex] if job == "report" else [latex, latex]
    env = os.environ.copy()
    for name in ("TEXINPUTS", "BIBINPUTS"):
        env[name] = f".{os.pathsep}{HERE}{os.pathsep}{env.get(name, '')}"
    for command in commands:
        result = subprocess.run(command, cwd=scratch, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(result.stdout[-6000:])
    log = (scratch / f"{job}.log").read_text(errors="replace")
    warnings = [line for line in log.splitlines()
                if re.search(r"Overfull \\[hv]box|undefined references|undefined citations", line)]
    if warnings:
        raise SystemExit(f"{job}.tex has layout or citation warnings:\n" + "\n".join(warnings))



def require_final_review(models, benchmark):
    checks = ("independence", "causality_and_validation", "artifact_reproduction", "runtime")
    review_path = HERE / "editor/review.json"
    require(review_path.is_file(), "Final assembly requires editor/review.json")
    review = read_json(review_path)
    require(review["runtime_policy"] in ("shared_hardware", "separate_hardware"),
            "Choose the group runtime policy before final assembly")
    contract = fingerprint()
    hardware = set()
    for model in models:
        model_id = model["id"]
        folder = HERE / "contributions" / model_id
        receipt = folder / "benchmark_audit.json"
        require(receipt.is_file(), f"{model_id}: missing benchmark audit receipt")
        audit = read_json(receipt)
        require(audit["status"] == "passed" and audit["model"] == model_id
                and audit["protocol"] == benchmark["protocol"] and audit["prediction_bundles"] == 90
                and audit["contract_sha256"] == contract
                and audit["results_sha256"] == sha256(folder / "results.csv"),
                f"{model_id}: benchmark audit is stale or incomplete")
        entry = review["models"][model_id]
        require(entry["reviewer"].strip() and entry["reviewer"] != model["owner"]
                and entry["audit_sha256"] == sha256(receipt)
                and entry["provenance_sha256"] == sha256(folder / "provenance.json")
                and all(entry[key] is True for key in checks),
                f"{model_id}: editor review is incomplete or stale")
        require(entry["hardware_id"].strip(), f"{model_id}: missing reviewed timing hardware")
        hardware.add(entry["hardware_id"])
        code = HERE.parent / model["code"]
        for name in ("paper_reading_notes.md", "source_inventory.md", "independent_design.md",
                     "implementation_provenance.md", "reused_fragments.csv", "own_unit_tests"):
            require((code / name).exists(), f"{model_id}: missing {name}")
    if review["runtime_policy"] == "shared_hardware":
        require(len(hardware) == 1, "Shared timing comparison requires the same reviewed host and device")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="require all six models and editor synthesis")
    parser.add_argument("--check", action="store_true", help="validate contributions without running LaTeX")
    parser.add_argument("--preview", help="include a pending model in a contributor draft")
    args = parser.parse_args()
    require(not (args.final and args.preview), "Use --preview or --final")
    models = read_json(HERE / "models.json")
    benchmark = read_json(HERE.parent / "benchmark/fixtures/manifests.json")
    require(len(models) == len({model["id"] for model in models}) == 6, "Expected six distinct model IDs")
    require(not args.preview or any(model["id"] == args.preview for model in models), "Unknown preview model")
    ready = {}
    pending = []
    for model in models:
        require(re.fullmatch(r"[a-z][a-z0-9_]*", model["id"]), "Invalid model ID")
        require((HERE.parent / model["code"]).is_dir(), f"Missing code directory for {model['id']}")
        if model["status"] == "ready" or model["id"] == args.preview:
            ready[model["id"]] = contribution(model, benchmark)
        else:
            require(model["status"] == "pending", f"Unknown status for {model['id']}")
            pending.append(model["id"])
    if args.final:
        require(not pending, f"Final assembly blocked. Pending models: {', '.join(pending)}")
        for name in ("report_synthesis.tex", "slides_synthesis.tex"):
            require((HERE / "editor" / name).is_file(), f"Final assembly requires editor/{name}")
        require_final_review(models, benchmark)
    print(f"Validated {len(ready)} of {len(models)} model contributions.", flush=True)
    if not args.check:
        with tempfile.TemporaryDirectory(prefix="comp5331-docs-") as directory:
            scratch = Path(directory)
            render(models, ready, args.final, args.preview, scratch)
            for job in ("report", "slides"):
                compile_tex(job, scratch)
            BUILD.mkdir(exist_ok=True)
            for job in ("report", "slides"):
                shutil.copyfile(scratch / f"{job}.pdf", BUILD / f"{job}.pdf")
                print(f"Built submission/build/{job}.pdf")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
