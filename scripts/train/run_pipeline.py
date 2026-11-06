"""
run_pipeline.py - from prepared Slakh backings to an installed, verified,
scored learned layer, in one command.

Usage (training env):
    .venv-sep/Scripts/python.exe scripts/train/run_pipeline.py [--data data/slakh] [--features data/features]
        [--sizes 32,48] [--epochs 40] [--skip-features]

Steps:
  1. make_features.py on every split (parallel, resumable)
  2. train_beatnet.py once per model size; keep the best validation beat F
  3. parity_check.py: the C++ layer must match PyTorch (< 1e-3)
  4. install as weights/beatnet.json and point weights/drummer.json at it
     (the previous files are kept as *.bak)
  5. eval_dataset.py on the validation split: hand-written vs learned
"""

import json
import shutil
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable


def dump_compact(doc) -> str:
    """Pretty JSON with every array of plain numbers on one line."""
    import re
    text = json.dumps(doc, indent=2)
    return re.sub(r"\[\s+([-0-9.e,\s]+?)\s+\]",
                  lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text) + "\n"


def run(cmd, check=True):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    res = subprocess.run([str(c) for c in cmd])
    if check and res.returncode != 0:
        raise SystemExit(f"step failed ({res.returncode}): {cmd[1]}")
    return res.returncode


def main() -> int:
    args = sys.argv[1:]
    opt = {"data": "data/slakh", "features": "data/features", "sizes": "32,48", "epochs": "40",
           "steps": "300", "weights-dir": "weights"}
    skip_features = "--skip-features" in args
    for i, a in enumerate(args):
        k = a.lstrip("-")
        if k in opt and i + 1 < len(args):
            opt[k] = args[i + 1]
    data, feats = ROOT / opt["data"], ROOT / opt["features"]

    # ---- 1. features ------------------------------------------------------------
    if not skip_features:
        run([PY, HERE / "make_features.py", data, feats, "--workers", "28"])

    # ---- 2. train each size, keep the best -------------------------------------------
    best = None
    for c in [int(s) for s in opt["sizes"].split(",") if s]:
        out = ROOT / "output" / f"beatnet_c{c}.json"
        run([PY, HERE / "train_beatnet.py", feats, out, "--channels", c, "--epochs", opt["epochs"],
             "--steps", opt["steps"]])
        f = json.loads(out.read_text(encoding="utf-8"))["training"]["best_val_beat_f"]
        print(f"size {c}: best validation beat F {f:.3f}")
        if best is None or f > best[1]:
            best = (out, f, c)
    out, f, c = best
    print(f"\nchosen: {c} channels (validation beat F {f:.3f})")

    # ---- 3. parity ------------------------------------------------------------------------
    val_tracks = sorted(p for p in (data / "validation").iterdir() if (p / "backing_full.flac").exists())
    run([PY, HERE / "parity_check.py", out, out.with_suffix(".pt"), val_tracks[0] / "backing_full.flac"])

    # ---- 4. install ------------------------------------------------------------------------
    wdir = ROOT / opt["weights-dir"]
    for name in ("beatnet.json", "drummer.json"):
        if (wdir / name).exists():
            shutil.copyfile(wdir / name, wdir / f"{name}.bak")
    shutil.copyfile(out, wdir / "beatnet.json")
    drummer = json.loads((wdir / "drummer.json").read_text(encoding="utf-8"))
    hand_only = json.loads(json.dumps(drummer))          # deep copy
    drummer["beatNet"]["file"] = "beatnet.json"
    hand_only["beatNet"]["file"] = ""
    (wdir / "drummer.json").write_text(dump_compact(drummer), encoding="utf-8")
    (ROOT / "output" / "handwritten_only.json").write_text(dump_compact(hand_only), encoding="utf-8")
    print(f"installed {wdir / 'beatnet.json'} ({c} channels)")

    # ---- 5. score ---------------------------------------------------------------------------
    val = data / "validation"
    run([PY, ROOT / "scripts" / "eval_dataset.py", val, "--weights", ROOT / "output" / "handwritten_only.json",
         "--tag", "hand-written"], check=False)
    run([PY, ROOT / "scripts" / "eval_dataset.py", val, "--weights", wdir / "drummer.json",
         "--tag", f"learned-c{c}"], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
