"""
parity_check.py - prove the C++ learned layer computes what PyTorch trained.

Usage (training env):
    .venv-sep/Scripts/python.exe scripts/train/parity_check.py <beatnet.json> <beatnet.pt> <backing.flac>

Runs the C++ harness with the network on one backing (exporting its
features), runs the PyTorch model on those same features, and compares the
beat and downbeat activations frame by frame. Exit code 0 when the largest
difference is below 1e-3.
"""

import csv
import json
import shutil
import subprocess
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from beatnet import BeatNet          # noqa: E402
from make_features import read_daf   # noqa: E402

WAV2MID = ROOT / "build" / "Release" / "wav2mid.exe"


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    net_json, net_pt, backing = map(Path, sys.argv[1:])
    meta = json.loads(net_json.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="parity_") as tmp:
        tmp = Path(tmp)
        # A weights file next to a copy of the network, so the relative name resolves.
        shutil.copyfile(net_json, tmp / "beatnet.json")
        w = json.loads((ROOT / "weights" / "drummer.json").read_text(encoding="utf-8"))
        w["beatNet"]["file"] = "beatnet.json"
        (tmp / "drummer.json").write_text(json.dumps(w), encoding="utf-8")

        y, sr = sf.read(str(backing), dtype="int16", always_2d=True)
        sf.write(str(tmp / "in.wav"), y, sr, subtype="PCM_16")
        res = subprocess.run([str(WAV2MID), str(tmp / "in.wav"), str(tmp / "run"),
                              "--weights", str(tmp / "drummer.json"), "--features", str(tmp / "x.daf")],
                             capture_output=True, text=True)
        if res.returncode != 0:
            print("wav2mid failed:", res.stderr.strip())
            return 1
        x, *_ = read_daf(tmp / "x.daf")
        rows = list(csv.DictReader(open(tmp / "run_frames.csv", encoding="utf-8")))

    model = BeatNet(channels=meta["channels"], dilations=meta["dilations"])
    model.load_state_dict(torch.load(net_pt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        py = torch.sigmoid(model(torch.from_numpy(x.copy())[None]))[0].numpy()

    cpp = np.array([[float(r["beat_act"]), float(r["downbeat_act"])] for r in rows])
    audible = np.array([r["silent"] == "0" for r in rows])
    n = min(len(cpp), len(py))
    d = np.abs(cpp[:n][audible[:n]] - py[:n, :2][audible[:n]])
    worst = float(d.max()) if d.size else float("nan")
    print(f"parity: {int(audible[:n].sum())} frames, max |C++ - PyTorch| = {worst:.2e}")
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
