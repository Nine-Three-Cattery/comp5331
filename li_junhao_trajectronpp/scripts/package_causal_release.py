#!/usr/bin/env python3
"""Build a relocation-safe archive of audited independent source and results."""

import argparse
import json
from pathlib import Path
import sys
import tarfile
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent
ROOT = PACKAGE / "outputs/causal_v2"
sys.path.insert(0, str(PACKAGE))
from independent_tpp.protocol import FOLDS, SEEDS, sha256

SCRIPTS = (
    "prepare_neutral_data", "validate_neutral_data", "cache_independent_data", "verify_causal_data",
    "fetch_causal_raw", "tiny_metric_check", "verify_independent_core", "train_independent",
    "select_independent_checkpoint", "predict_independent", "evaluate_independent_run", "evaluate",
    "metrics", "compute_independent_thresholds", "run_independent_formal", "audit_independent_formal",
    "aggregate_independent_results", "aggregate_spatial", "analyze_statistics", "analyze_sample_budget",
    "analyze_independent_diagnostics", "generate_failure_analysis", "benchmark_independent_runtime",
    "prepare_causal_report", "verify_causal_release", "package_causal_release",
)
DOCS = (
    "README.md", "requirements-causal-v2.lock", "source_inventory.md", "paper_reading_notes.md",
    "implementation_provenance.md", "independent_design.md", "reused_fragments.csv",
)
PRESENTATION = (
    "TrajectronPP_LI_Junhao_Technical_Report.pdf", "TrajectronPP_LI_Junhao_Technical_Report.tex",
    "TrajectronPP_5min_Slides.pdf", "TrajectronPP_5min_Slides.tex",
    "TrajectronPP_5min_Speaker_Notes.md", "TrajectronPP_Defense_Notes.md", "trajectronpp_section.md",
    "causal_results.tex", "build_latex.sh",
)


def files_to_archive():
    yield from (PACKAGE / name for name in DOCS)
    yield from (PACKAGE / "scripts" / f"{name}.py" for name in SCRIPTS)
    yield PACKAGE / "scripts/finalize_causal_results.sh"
    yield from (PACKAGE / "report" / name for name in PRESENTATION)
    yield PACKAGE / "configs/causal_v2_raw_sha256.json"
    yield from sorted((PACKAGE / "independent_tpp").glob("*.py"))
    yield from sorted((PACKAGE / "own_unit_tests").glob("*.py"))
    for relative in ("data/causal_v2", "data/causal_v2_cache"):
        yield from sorted(path for path in (PROJECT / relative).rglob("*") if path.is_file())
    for name in ("corruption_fixture.json", "formal_progress.log", "audit.json", "prediction_index.csv"):
        yield ROOT / name
    for directory in ("thresholds", "results", "runs/figures"):
        yield from sorted(path for path in (ROOT / directory).rglob("*") if path.is_file())
    for fold in FOLDS:
        for seed in SEEDS:
            run = ROOT / "runs" / fold / f"seed_{seed}"
            yield from sorted(path for path in run.iterdir() if path.is_file())
            yield from sorted(path for path in (run / "evaluation").rglob("*") if path.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    audit = json.loads((ROOT / "audit.json").read_text())
    assert audit["status"] == "passed" and audit["prediction_bundles"] == 90
    files = sorted(set(files_to_archive()))
    assert sum(path.suffix == ".npz" for path in files) == 90
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    pending = args.archive.with_suffix(args.archive.suffix + ".tmp")
    archive_root = "comp5331_trajectronpp_causal_v2"
    with tempfile.TemporaryDirectory() as directory:
        index = Path(directory) / "SHA256SUMS"
        start = Path(directory) / "README.md"
        start.write_text(
            "# COMP5331 Trajectron++ causal_v2 release\n\n"
            "Start with [the reproducibility guide](li_junhao_trajectronpp/README.md).\n"
            "The English report, slides, speaker notes and defense notes are in "
            "`li_junhao_trajectronpp/report`.\n"
        )
        index.write_text(
            f"{sha256(start)}  README.md\n"
            + "".join(f"{sha256(path)}  {path.relative_to(PROJECT)}\n" for path in files)
        )
        with tarfile.open(pending, "w:gz", compresslevel=1) as archive:
            archive.add(index, arcname=f"{archive_root}/SHA256SUMS")
            archive.add(start, arcname=f"{archive_root}/README.md")
            for path in files:
                archive.add(path, arcname=f"{archive_root}/{path.relative_to(PROJECT)}", recursive=False)
    pending.replace(args.archive)
    args.archive.with_suffix(args.archive.suffix + ".sha256").write_text(
        f"{sha256(args.archive)}  {args.archive.name}\n"
    )
    print(f"CAUSAL_RELEASE_PACKED files={len(files)} archive={args.archive}")


if __name__ == "__main__":
    main()
