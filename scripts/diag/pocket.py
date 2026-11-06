"""
pocket.py - is the drummer in the pocket? Checks a render directly against the
real drum stem, with no beat tracker in between.

A beat tracker on a drum stem can sit half a beat off and still look right,
because hi-hats fill the gaps. The snare cannot be fooled that way: in a groove
it marks 2 and 4. So each stretch of the song is folded onto OUR bar, and this
reports where the real snare and kick landed. Personal test songs only.

Usage: .venv/Scripts/python.exe scripts/diag/pocket.py <drums.wav> <prefix_steps.csv> [--window 10]

Snare column:  2+4 in the pocket   | 1+3 bar is one beat off   | & half a beat off
"""

import csv
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

HOP_S = 0.005             # requested resolution; attacks() reports the exact one
BINS = 16                 # sixteenths across two beats would be 8; a whole bar folds to 16


def attacks(y, sr, lo, hi):
    """
    Rectified energy rise in one band, and the true seconds per value. The hop
    is a whole number of samples, so at 44.1 kHz it is 220 samples, or 4.989 ms.
    Treating that as 5 ms stretches the time axis by 0.23%, which reads as the
    drummer sliding a full beat away from the band over three minutes.
    """
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    e = np.abs(sosfiltfilt(sos, y))
    hop = int(sr * HOP_S)
    n = len(e) // hop
    e = e[: n * hop].reshape(n, hop).mean(axis=1)
    return np.maximum(np.diff(e, prepend=0.0), 0.0), hop / sr


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 2
    window = float(args[args.index("--window") + 1]) if "--window" in args else 10.0

    y, sr = sf.read(args[0], dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    mid, dt = attacks(y, sr, 180, 900)
    snare = mid + 0.5 * attacks(y, sr, 2000, 6000)[0]
    kick = attacks(y, sr, 40, 110)[0]

    with open(args[1], encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["playing"] == "1"]
    if not rows:
        print("nothing played")
        return 1
    t = np.array([float(r["time_s"]) for r in rows])
    s = np.array([int(r["step_in_bar"]) for r in rows])

    print(f"{'time':>9}  snare at  verdict")
    verdicts = []
    for t0 in np.arange(0.0, t[-1], window):
        m = (t >= t0) & (t < t0 + window)
        if m.sum() < 16:
            continue
        energy, ke = np.zeros(BINS), np.zeros(BINS)
        for tt, st in zip(t[m], s[m]):
            i = int(tt / dt)
            a, b = max(i - 6, 0), min(i + 7, len(snare))
            if b > a:
                energy[st] += snare[a:b].max()
                ke[st] += kick[a:b].max()
        if energy.sum() < 1e-6:
            continue
        energy /= energy.sum()
        ke /= ke.sum() + 1e-9
        # Fold to 2 beats: backbeat lands on position 4 of 8 sixteenth slots.
        fold = energy[:8] + energy[8:]
        pos = int(np.argmax(fold))
        share = fold[pos]
        name = {0: "1+3", 2: "1e+3e", 4: "2+4", 6: "2&+4&"}.get(pos, f"16th{pos}")
        odd = fold[2] + fold[6]
        v = "in the pocket" if pos == 4 else ("bar off by a beat" if pos == 0 else ("half a beat off" if pos in (2, 6) else "off grid"))
        if share < 0.3:
            v = "unclear"
        verdicts.append(v)
        print(f"{t0:5.0f}-{t0 + window:<3.0f}  {name:>6} {share:4.0%}  {v}")
    print("\nsummary: " + ", ".join(f"{k} {verdicts.count(k)}" for k in sorted(set(verdicts), key=lambda k: -verdicts.count(k))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
