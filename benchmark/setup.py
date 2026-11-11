"""Prepare and verify the common ETH/UCY benchmark."""

import json
import subprocess
import sys

from contract import FIXTURES, OWNER, ROOT, fingerprint, sha256

DATA = ROOT / "data/causal_v2"
THRESHOLDS = ROOT / "data/benchmark_thresholds"


def run(script, *args):
    subprocess.run([sys.executable, str(OWNER / "scripts" / script), *map(str, args)], check=True)


def check_manifests():
    expected = json.loads((FIXTURES / "manifests.json").read_text())["folds"]
    for fold, splits in expected.items():
        for split, row in splits.items():
            path = DATA / fold / f"{split}_manifest.csv"
            if sha256(path) != row["sha256"]:
                raise SystemExit(f"Manifest mismatch: {path}")


def check_corruption_fixture():
    import numpy as np

    sys.path.insert(0, str(OWNER))
    from independent_tpp.protocol import CONDITIONS, corruption_seed, perturb_history

    conditions = {tag: (kind, level) for tag, kind, level in CONDITIONS}
    for row in json.loads((FIXTURES / "corruption.json").read_text())["examples"]:
        kind, level = conditions[row["condition"]]
        seed = corruption_seed(row["training_seed"], row["manifest_index"])
        if seed != row["corruption_seed"]:
            raise ValueError(f"Corruption seed mismatch for {row['window_id']}")
        observed, mask = perturb_history(row["clean_observed_m"], kind, level, seed)
        np.testing.assert_array_equal(observed.astype("float32"), row["model_observed_m"])
        np.testing.assert_array_equal(mask, row["observed_mask"])


def main():
    raw = ROOT / "data/eth_ucy_raw"
    run("fetch_causal_raw.py", raw)
    run("prepare_neutral_data.py", "--raw-root", raw, "--output-root", DATA)
    check_manifests()
    check_corruption_fixture()
    run("tiny_metric_check.py")
    run("compute_independent_thresholds.py", "--data-root", DATA, "--output-dir", THRESHOLDS)
    print(f"BENCHMARK_OK {fingerprint()}")


if __name__ == "__main__":
    main()
