#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -s causal_results.tex ]; then
  echo "Generate causal_results.tex from the audited matrix before building." >&2
  exit 2
fi
exec python3 - <<'PY'
from pathlib import Path
import shutil
import subprocess
import tempfile

documents = ("TrajectronPP_LI_Junhao_Technical_Report", "TrajectronPP_5min_Slides")
with tempfile.TemporaryDirectory(prefix="comp5331-personal-") as scratch:
    for document in documents:
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-halt-on-error", "-interaction=nonstopmode",
                 f"-output-directory={scratch}", f"{document}.tex"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            if result.returncode:
                raise SystemExit(result.stdout[-6000:])
    for document in documents:
        shutil.copyfile(Path(scratch) / f"{document}.pdf", f"{document}.pdf")
PY
