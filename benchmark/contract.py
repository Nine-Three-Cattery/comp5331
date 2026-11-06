"""Files that identify the shared comparison protocol."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = ROOT / "li_junhao_trajectronpp"
FIXTURES = ROOT / "benchmark/fixtures"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint():
    paths = [
        *sorted(FIXTURES.glob("*.json")),
        Path(__file__),
        ROOT / "benchmark/setup.py",
        ROOT / "benchmark/validate.py",
        OWNER / "configs/causal_v2_raw_sha256.json",
        OWNER / "independent_tpp/protocol.py",
        OWNER / "independent_tpp/data.py",
        *(OWNER / "scripts" / name for name in (
            "fetch_causal_raw.py", "prepare_neutral_data.py", "compute_independent_thresholds.py",
            "tiny_metric_check.py",
            "evaluate.py", "metrics.py", "analyze_sample_budget.py",
        )),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(f"{path.relative_to(ROOT).as_posix()} {sha256(path)}\n".encode())
    return digest.hexdigest()
