"""Reject incompatible populations before merging independent model results."""
import argparse
import json
from pathlib import Path


def compare_manifests(ours, theirs):
    """Requires exported explicit sample IDs, never accepts matching counts alone."""
    differences = []
    for key in ("held_out", "split", "files_sha256"):
        if key not in theirs or ours.get(key) != theirs[key]:
            differences.append(key)
    for key in ("obs_len", "pred_len", "dt", "stride", "train_fraction", "gap"):
        if ours.get("config", {}).get(key) != theirs.get("config", {}).get(key):
            differences.append("config." + key)
    ids = theirs.get("sample_ids")
    if not isinstance(ids, list) or ours["sample_ids"] != ids:
        differences.append("sample_ids/order")
    if differences:
        raise ValueError("Incompatible protocol: " + ", ".join(differences))
    return {"compatible": True, "trajectories": len(ids), "sample_ids_sha256": ours["sample_ids_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ours")
    parser.add_argument("theirs")
    args = parser.parse_args()
    print(json.dumps(compare_manifests(json.loads(Path(args.ours).read_text()), json.loads(Path(args.theirs).read_text()))))


if __name__ == "__main__":
    main()
