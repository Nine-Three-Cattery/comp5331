"""Assemble the group handoff from a clean, committed final source tree."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from build import HERE, sha256

ROOT = HERE.parent
DESTINATION = ROOT / "artifacts/group7"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    subprocess.run([sys.executable, str(HERE / "build.py"), "--final", "--check"], check=True)
    if git("status", "--porcelain"):
        raise SystemExit("Commit the reviewed final sources before exporting the release.")
    files = git("ls-files", "-z").decode().split("\0")
    for name in filter(None, files):
        path = Path(name)
        if path.parts[0] in {"data", "local", "third_party", "artifacts"} or "outputs" in path.parts \
                or path.suffix in {".pt", ".pth", ".npz", ".zip", ".ckpt", ".pkl"} \
                or path.is_relative_to("submission/build"):
            raise SystemExit(f"Release blocked by tracked local artifact: {name}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="group7-release-", dir=DESTINATION.parent) as directory:
        stage = Path(directory)
        git("archive", "--format=zip", "--prefix=group7/", f"--output={stage / 'code.zip'}", "HEAD")
        with zipfile.ZipFile(stage / "code.zip") as archive:
            archive.extractall(stage / "source")
        packaged = stage / "source/group7/submission"
        subprocess.run([sys.executable, str(packaged / "build.py"), "--final"], check=True)
        for name in ("report.pdf", "slides.pdf"):
            shutil.copyfile(packaged / "build" / name, stage / name)
        models = json.loads((packaged / "models.json").read_text())
        combined = []
        notes = ["# Group 7 speaker notes\n"]
        for model in models:
            folder = packaged / "contributions" / model["id"]
            with (folder / "results.csv").open(newline="") as handle:
                combined.extend({"model": model["id"], **row} for row in csv.DictReader(handle))
            notes.append((folder / "speaker_notes.md").read_text())
        with (stage / "results.csv").open("w", newline="") as handle:
            fields = list(dict.fromkeys(key for row in combined for key in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(combined)
        (stage / "speaker_notes.md").write_text("\n".join(notes))
        shutil.rmtree(stage / "source")
        (stage / "SOURCE_COMMIT").write_bytes(git("rev-parse", "HEAD"))
        (stage / "SHA256SUMS").write_text("".join(
            f"{sha256(path)}  {path.name}\n" for path in sorted(stage.iterdir())
        ))
        for path in stage.iterdir():
            path.replace(DESTINATION / path.name)
    print(f"GROUP7_RELEASE_OK {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
