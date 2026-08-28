"""
Assemble the Round-1 deliverable ZIP.

The brief asks for "a ZIP file containing their trained model(s) and the
necessary files/code required to run or evaluate the models". This bundles the
persisted artifacts, the code that produced and consumes them, the submission,
and the write-ups -- and deliberately excludes the two directories that would
make the archive unshippable:

  data/       625 MB of competition CSVs the organisers already have
  ag_models/  274 MB of AutoGluon predictors, a benchmark rather than the
              shipped model (its edge under the official metric is ~0.008 skill,
              and loading it would require a matching AutoGluon install)
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "playhack_ml_submission.zip"

FILES = ["readme.txt", "predict.py", "requirements.txt", "submission.csv",
         "README.md", "RESULTS.md", "RUNME.md"]
DIRS = ["src", "models", "docs"]
SKIP_SUFFIX = {".pyc", ".log"}
SKIP_DIR = {"__pycache__", "catboost_info"}


def should_include(p: Path) -> bool:
    if p.suffix in SKIP_SUFFIX:
        return False
    return not any(part in SKIP_DIR for part in p.parts)


def main():
    if OUT.exists():
        OUT.unlink()
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in FILES:
            p = ROOT / name
            if p.exists():
                z.write(p, p.name)
                n += 1
            else:
                print("  (missing, skipped) " + name)
        for d in DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and should_include(p):
                    z.write(p, str(p.relative_to(ROOT)))
                    n += 1
        # a couple of figures make the deck/README readable offline
        for p in sorted((ROOT / "reports").glob("*.png")):
            z.write(p, "reports/" + p.name)
            n += 1
        for name in ["metrics.json", "threshold_sweep.csv",
                     "official_ceiling.json", "generator_ceiling.json"]:
            p = ROOT / "reports" / name
            if p.exists():
                z.write(p, "reports/" + name)
                n += 1

    size = OUT.stat().st_size / 1e6
    print("wrote " + OUT.name + "  " + str(n) + " files  " + format(size, ".1f") + " MB")
    if size > 50:
        print("  WARNING: larger than a typical portal limit (50 MB)")


if __name__ == "__main__":
    main()
