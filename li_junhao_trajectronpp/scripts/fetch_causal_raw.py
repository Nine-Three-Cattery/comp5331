#!/usr/bin/env python3
"""Retrieve only the six locked ETH/UCY text files, with SHA-256 verification."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

PACKAGE = Path(__file__).resolve().parents[1]
COMMIT = "ef0165a93ee5ba8cdc14f9b999b3e00070cd8588"
BASE = (f"https://raw.githubusercontent.com/StanfordASL/Trajectron-plus-plus/{COMMIT}/"
        "experiments/pedestrians/raw/raw/all_data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    expected = json.loads((PACKAGE / "configs/causal_v2_raw_sha256.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, digest in expected.items():
        path = args.output / filename
        content = path.read_bytes() if path.is_file() else urlopen(f"{BASE}/{filename}", timeout=30).read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Raw file hash mismatch: {filename}")
        if not path.is_file():
            path.write_bytes(content)
        print(f"RAW_OK {filename} {digest}")


if __name__ == "__main__":
    main()
