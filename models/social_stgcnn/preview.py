"""Compile the owner's review PDFs with the group's preambles and metric macros."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission"))
import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/social_stgcnn/previews")
    args = parser.parse_args()
    model_id = "social_stgcnn"
    registry = build.read_json(build.HERE / "models.json")
    model = next(row for row in registry if row["id"] == model_id)
    lock = build.read_json(ROOT / "benchmark/fixtures/manifests.json")
    ready = {model_id: build.contribution(model, lock)}
    bodies = {
        "report": r"""\color{Ink}\setcounter{secnumdepth}{0}
{\small\sffamily\color{Teal}COMP5331 GROUP 7 \quad / \quad CONTRIBUTOR REVIEW\par}
\input{contributions/social_stgcnn/report.tex}
\section*{Contribution}
\input{contributions/social_stgcnn/contribution_statement.tex}
\small\bibliographystyle{plain}\bibliography{references}""",
        "slides": r"""\input{contributions/social_stgcnn/slides.tex}""",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="social-stgcnn-preview-") as directory:
        scratch = Path(directory)
        build.render([model], ready, False, model_id, scratch)
        for job, body in bodies.items():
            master = (build.HERE / f"{job}.tex").read_text()
            preamble = master.split(r"\begin{document}", 1)[0]
            (scratch / f"{job}.tex").write_text(preamble + "\n\\begin{document}\n" + body + "\n\\end{document}\n")
            build.compile_tex(job, scratch)
            dest = args.out / f"social_stgcnn_{job}.pdf"
            shutil.copy2(scratch / f"{job}.pdf", dest)
            print(dest)


if __name__ == "__main__":
    main()
