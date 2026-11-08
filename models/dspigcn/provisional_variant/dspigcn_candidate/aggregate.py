"""Aggregate per-scene/per-condition means and variation across training seeds."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .runner import save_csv


def aggregate(paths, output):
    groups = defaultdict(list)
    seen = set()
    for path in paths:
        with Path(path).open() as file:
            for row in csv.DictReader(file):
                key = (row["model"], row["scene"], row["condition"], row["sample_count"], row["test_set_id"])
                identity = (*key, row["seed"])
                if identity in seen:
                    raise ValueError(f"Duplicate result: {identity}")
                seen.add(identity)
                groups[key].append(row)
    rows = []
    for key, group in sorted(groups.items()):
        summary = dict(zip(("model", "scene", "condition", "sample_count", "test_set_id"), key))
        summary.update(seed_count=len(group), seeds=",".join(sorted(row["seed"] for row in group)))
        for metric in ("minADE", "minFDE", "meanADE", "meanFDE", "energy_score", "diversity", "speed_mean",
                       "acceleration_mean", "jerk_mean", "close_approach_rate"):
            values = np.array([float(row[metric]) for row in group])
            summary[metric + "_mean"] = float(values.mean())
            summary[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else ""
        rows.append(summary)
    save_csv(output, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aggregate(args.inputs, args.output)


if __name__ == "__main__":
    main()
