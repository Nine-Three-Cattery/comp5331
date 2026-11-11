"""Build and verify the audited owner review package from the repository root."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from analysis import (
    ARTIFACTS, ASSETS, CONDITIONS, CONTRIBUTION, EVIDENCE, FOLDS, FORMAL, QUALITATIVE,
    ROOT, RUNTIME, read_json, runtime_table, summarize,
)

sys.path[:0] = [str(ROOT / "benchmark"), str(ROOT / "li_junhao_trajectronpp")]
from contract import fingerprint, sha256
from independent_tpp.protocol import digest_json

BASE = "df016def5890898b70d372fb0e8e2bc142121330"
ARCHIVE_NAME = "COMP5331_SocialGAN_causal_v2.tar.gz"
ARTIFACT = ARTIFACTS / ARCHIVE_NAME
OWNER = ROOT / "models/social_gan"
RAW_LOCK = ROOT / "li_junhao_trajectronpp/configs/causal_v2_raw_sha256.json"
REFERENCES = ROOT / "submission/references.bib"
RESULTS = CONTRIBUTION / "results.csv"
AUDIT = CONTRIBUTION / "benchmark_audit.json"
PROVENANCE = CONTRIBUTION / "provenance.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def valid_audit(audit, results_hash):
    expected = {"model": "social_gan", "protocol": "causal_v2", "status": "passed",
                "prediction_bundles": 90, "results_sha256": results_hash,
                "contract_sha256": fingerprint()}
    require(all(audit.get(key) == value for key, value in expected.items()),
            "The formal audit receipt does not match this CSV and benchmark contract")


def audited_results():
    import pandas as pd

    receipt = ARTIFACTS / AUDIT.name
    valid_audit(read_json(receipt), sha256(RESULTS))
    shutil.copyfile(receipt, AUDIT)
    results = pd.read_csv(RESULTS).set_index(["fold", "seed", "condition"])
    require(results.index.is_unique, "Duplicate fold, seed and condition in results")
    return results


def archive_filter(member):
    path = Path(member.name)
    if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
        return None
    if member.isfile() and path.suffix != ".tmp" and path.name not in ("provenance.json", "reproduction_check.json"):
        member.uid = member.gid = 0
        member.uname = member.gname = ""
        return member
    return None


def archive_files():
    mandatory = (OWNER, CONTRIBUTION, REFERENCES, RAW_LOCK, FORMAL, ARTIFACTS / "pilot",
                 ARTIFACTS / "metrics", QUALITATIVE, ARTIFACTS / "pilot_audit.json")
    optional = (ARTIFACTS / "benchmark.log", ARTIFACTS / "environment-lock.txt",
                ARTIFACTS / "model_smoke.log",
                *(ARTIFACTS / "formal").glob("*.log"))
    for path in (*mandatory, *(path for path in optional if path.is_file())):
        require(path.exists(), f"Missing release input {path.relative_to(ROOT)}")
        if path.is_dir():
            yield from (item for item in sorted(path.rglob("*")) if item.is_file())
        else:
            yield path


def relative(path):
    return path.relative_to(ROOT).as_posix()


def add_file(archive, path):
    if path.name != "summary.json" or not path.is_relative_to(ARTIFACTS / "metrics"):
        archive.add(path, arcname=relative(path), recursive=False, filter=archive_filter)
        return
    # The common scorer writes machine-local paths in summary files.
    summary = read_json(path)
    for key in ("predictions", "window_metrics", "scene_summary", "spatial_consistency"):
        value = Path(summary[key])
        if value.is_absolute():
            summary[key] = relative(value)
    content = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    member = tarfile.TarInfo(relative(path))
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def package(results):
    require(all((CONTRIBUTION / name).is_file() for name in
                ("report.tex", "slides.tex", "speaker_notes.md", "analysis.tex")),
            "Complete the report, slides, notes and analysis before packaging")
    runtime, environments = runtime_table(results)
    pilot_receipt = ARTIFACTS / "pilot_audit.json"
    pilot = read_json(pilot_receipt)
    require(pilot == {"model": "social_gan", "protocol": "causal_v2", "status": "passed",
                      "prediction_bundles": 6, "results_sha256": None, "contract_sha256": fingerprint()},
            "The pilot receipt does not match this benchmark contract")
    pilot_execution = ARTIFACTS / "pilot/runs/eth/seed_0/execution.json"
    logs = [ARTIFACTS / name for name in ("benchmark.log", "model_smoke.log", "pilot/gate.log",
                                        "formal/audit.log", "formal/replay.log")]
    artifact_meta = {
        "protocol": "causal_v2",
        "runs": 15,
        "prediction_bundles": 90,
        "base_commit": BASE,
        "runtime_ms": {key: float(runtime[key].median()) for key in RUNTIME},
        "runtime_environments": environments,
        "runtime_comparability": "Compare timing only with measurements taken on the same host and device.",
        "pilot": {"audit_sha256": sha256(pilot_receipt), "execution_sha256": sha256(pilot_execution),
                  "execution": read_json(pilot_execution)},
        "validation_logs_sha256": {relative(path): sha256(path) for path in logs if path.is_file()},
        "raw_release_sha256": read_json(RAW_LOCK),
        "references_sha256": sha256(REFERENCES),
        "owner_source_sha256": {
            relative(path): sha256(path) for path in sorted(OWNER.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith(".")
        },
        "evidence": {str(path.relative_to(CONTRIBUTION)): sha256(path)
                     for path in (AUDIT, RESULTS, CONTRIBUTION / "analysis.tex", CONTRIBUTION / "report.tex",
                                  CONTRIBUTION / "slides.tex", CONTRIBUTION / "speaker_notes.md",
                                  CONTRIBUTION / "proposal_checklist.md",
                                  *sorted(EVIDENCE.glob("*.csv")), *sorted(EVIDENCE.glob("*.json")),
                                  *sorted(ASSETS.glob("*.pdf")))},
        "assistance": ("Generative AI assisted with paper synthesis, model and experiment code, experiment execution, "
                       "analysis, and drafting the report, slides and speaker notes."),
    }
    with tempfile.TemporaryDirectory(prefix="package-", dir=ARTIFACTS) as scratch:
        candidate = Path(scratch) / ARCHIVE_NAME
        with tarfile.open(candidate, "w:gz", compresslevel=1) as archive:
            for path in archive_files():
                add_file(archive, path)
        artifact_meta["artifact"] = {"name": ARCHIVE_NAME, "sha256": sha256(candidate)}
        verify(candidate, artifact_meta)
        candidate.replace(ARTIFACT)
    write_json(PROVENANCE, artifact_meta)
    print(f"OWNER_HANDOFF_OK {relative(ARTIFACT)} {artifact_meta['artifact']['sha256']}")


def member_hash(archive, member):
    with archive.extractfile(member) as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(artifact, provenance):
    require(provenance["base_commit"] == BASE and provenance["raw_release_sha256"] == read_json(RAW_LOCK),
            "The provenance does not identify the pinned group benchmark")
    require(sha256(artifact) == provenance["artifact"]["sha256"], "Archive SHA-256 mismatch")
    csv_name, audit_name = relative(RESULTS), relative(AUDIT)
    run_prefix = relative(FORMAL) + "/"
    evidence_prefix = relative(CONTRIBUTION) + "/"
    all_hashes, configs, selections = {}, {}, {}
    found_results, found_audit = None, None
    with tarfile.open(artifact, "r|gz") as archive:
        for member in archive:
            name = member.name
            require(member.isfile() and not Path(name).is_absolute() and ".." not in Path(name).parts,
                    f"Unexpected archive member {name}")
            require(name not in all_hashes, f"Duplicate archive member {name}")
            if name in (csv_name, audit_name) or (name.startswith(run_prefix) and
                                                Path(name).name in ("config.json", "checkpoint_selection.json")):
                with archive.extractfile(member) as handle:
                    content = handle.read()
                all_hashes[name] = hashlib.sha256(content).hexdigest()
                if name == csv_name:
                    found_results = list(csv.DictReader(content.decode("utf-8").splitlines()))
                elif name == audit_name:
                    found_audit = json.loads(content)
                elif Path(name).name == "config.json":
                    configs[Path(name).parent.as_posix()] = digest_json(json.loads(content))
                else:
                    selections[Path(name).parent.as_posix()] = json.loads(content)
            else:
                all_hashes[name] = member_hash(archive, member)
    require(found_results is not None and found_audit is not None, "Missing archived results or audit receipt")
    valid_audit(found_audit, all_hashes[csv_name])
    require(all_hashes.get(relative(REFERENCES)) == provenance["references_sha256"], "Bibliography hash mismatch")
    for name, digest in provenance["owner_source_sha256"].items():
        require(all_hashes.get(name) == digest, f"Archived owner source mismatch for {name}")
        require(sha256(ROOT / name) == digest, f"Working source differs from the archive for {name}")
    for name, key in (("pilot_audit.json", "audit_sha256"), ("pilot/runs/eth/seed_0/execution.json", "execution_sha256")):
        require(all_hashes.get(relative(ARTIFACTS / name)) == provenance["pilot"][key], f"Pilot hash mismatch for {name}")
    for name, digest in provenance["validation_logs_sha256"].items():
        require(all_hashes.get(name) == digest, f"Validation log hash mismatch for {name}")
    for name, digest in provenance["evidence"].items():
        require(all_hashes.get(evidence_prefix + name) == digest, f"Evidence hash mismatch for {name}")
    expected = {(fold, str(seed), tag) for fold in FOLDS for seed in range(3) for tag in CONDITIONS}
    actual = {(row["fold"], row["seed"], row["condition"]) for row in found_results}
    require(actual == expected and len(found_results) == len(expected), "Archive result matrix is incomplete")
    criterion = ["energy_score", "mean_ade", "epoch"]
    require(len(configs) == len(selections) == 15, "Expected 15 formal training records")
    for directory, selection in selections.items():
        require(selection["selection_split"] == "val" and selection["test_used_for_selection"] is False
                and selection["criterion"] == criterion, f"Invalid selection rule in {directory}")
        candidates = selection["candidates"]
        require(candidates, f"Missing validation candidates in {directory}")
        winner = min(candidates, key=lambda row: tuple(row[key] for key in criterion))
        require(all(selection["selected"][key] == value for key, value in winner.items()),
                f"Selected checkpoint does not match the validation winner in {directory}")
    required_bundles = set()
    for row in found_results:
        tag = row["condition"]
        directory = f"{run_prefix}{row['fold']}/seed_{row['seed']}"
        require(configs[directory] == row["config_hash"], f"Config hash mismatch in {directory}")
        checkpoint = selections[directory]["selected"]["checkpoint"]
        require(checkpoint == Path(checkpoint).name and
                all_hashes.get(f"{directory}/{checkpoint}") == row["checkpoint_sha256"],
                f"Checkpoint hash mismatch in {directory}")
        bundle = f"{directory}/evaluation/{tag}/predictions_test_{tag}.npz"
        require(all_hashes.get(bundle) == row["prediction_sha256"], f"Bundle hash mismatch for {bundle}")
        required_bundles.add(bundle)
    archived_bundles = {name for name in all_hashes if name.startswith(run_prefix) and "/predictions_test_" in name}
    require(archived_bundles == required_bundles, "Archived formal prediction bundle set differs from the audit")
    print("OWNER_ARCHIVE_VERIFIED 15 selected checkpoints and 90 prediction bundles")


def rescore(artifact, provenance):
    """Audit the extracted bundles and CSV against an independent copy of the benchmark."""
    verify(artifact, provenance)
    with tempfile.TemporaryDirectory(prefix="social-gan-reproduction-") as temporary:
        checkout = Path(temporary)
        with subprocess.Popen(["git", "archive", provenance["base_commit"]], cwd=ROOT, stdout=subprocess.PIPE) as git:
            with tarfile.open(fileobj=git.stdout, mode="r|") as baseline:
                baseline.extractall(checkout, filter="data")
            require(git.wait() == 0, "Could not materialize the pinned group benchmark")
        raw_relative = Path("data/eth_ucy_raw")
        raw_destination = checkout / raw_relative
        raw_destination.mkdir(parents=True)
        for filename in provenance["raw_release_sha256"]:
            source = ROOT / raw_relative / filename
            if source.is_file():
                shutil.copyfile(source, raw_destination / filename)
        subprocess.run([sys.executable, "benchmark/setup.py"], cwd=checkout, check=True)
        run_prefixes = (relative(FORMAL) + "/", relative(ARTIFACTS / "pilot/runs") + "/")
        with tarfile.open(artifact, "r|gz") as archive:
            for member in archive:
                if member.name == relative(RESULTS) or member.name.startswith((*run_prefixes, relative(OWNER) + "/")):
                    archive.extract(member, checkout, filter="data")
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                        "models/social_gan/own_unit_tests"], cwd=checkout, check=True)
        subprocess.run([sys.executable, "models/social_gan/run.py", "formal", "--replay", "eth", "0"],
                       cwd=checkout, check=True)
        subprocess.run([sys.executable, "models/social_gan/run.py", "pilot", "--runs",
                        "artifacts/social_gan/retrain_check/runs"], cwd=checkout, check=True)
        subprocess.run([sys.executable, "models/social_gan/check_training.py"], cwd=checkout, check=True)
        subprocess.run([sys.executable, "models/social_gan/check_metrics.py"], cwd=checkout, check=True)
        for pilot in (True, False):
            runs = relative(ARTIFACTS / "pilot/runs") if pilot else relative(FORMAL)
            command = [sys.executable, "benchmark/validate.py", "social_gan", "--runs", runs]
            if pilot:
                command.append("--pilot")
            subprocess.run(command, cwd=checkout, check=True)
        verified = read_json(checkout / relative(ARTIFACTS / AUDIT.name))
        require(verified == read_json(AUDIT), "Extracted formal audit differs from the review copy")
        pilot = checkout / relative(ARTIFACTS / "pilot_audit.json")
        require(sha256(pilot) == provenance["pilot"]["audit_sha256"], "Extracted pilot audit differs")
        receipt = {
            "status": "passed",
            "base_commit": provenance["base_commit"],
            "artifact_sha256": provenance["artifact"]["sha256"],
            "contract_sha256": verified["contract_sha256"],
            "results_sha256": verified["results_sha256"],
            "formal_bundles": verified["prediction_bundles"],
            "pilot_bundles": read_json(pilot)["prediction_bundles"],
            "archived_source_tests": "passed",
            "archived_source_replay": "eth/seed_0, all six conditions, exact arrays and metadata",
            "archived_source_training": "Fresh two-epoch pilot, exact selected tensors and all six exports",
            "independent_metrics": "90 bundles, six headline scores",
        }
        write_json(ARTIFACTS / "reproduction_check.json", receipt)
        write_json(CONTRIBUTION / "reproduction_check.json", receipt)
    print("OWNER_REPRODUCTION_OK extracted pilot and formal bundles scored with the pinned group benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("analyze", "package", "verify", "rescore"))
    parser.add_argument("--artifact", type=Path, default=ARTIFACT, help="Archive to check in verify or rescore mode")
    args = parser.parse_args()
    if args.mode == "verify":
        verify(args.artifact, read_json(PROVENANCE))
    elif args.mode == "rescore":
        rescore(args.artifact, read_json(PROVENANCE))
    elif args.mode == "analyze":
        runtime, _ = summarize(audited_results())
        require(all(math.isfinite(value) for value in runtime.values()), "Runtime summary is non-finite")
        print(f"OWNER_EVIDENCE_OK {relative(EVIDENCE)}")
    else:
        package(audited_results())


if __name__ == "__main__":
    main()
