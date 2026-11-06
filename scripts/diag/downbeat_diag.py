"""
downbeat_diag.py - why is the "1" wrong? Separates three possible causes on the
validation split, with no C++ tracker in the loop:

  grid     are the MIDI bar lines themselves trustworthy? Measured from the
           drum part: does the snare sit on beats 2 and 4 of the MIDI bar?
  head     given perfect beats, does the learned downbeat output pick the right
           slot? Scored after 8, 16, 32 and 64 beats, causally, so the answer
           says how long the drummer needs to listen.
  backbeat the same, for the coarser question a listener hears first: is the
           snare on the right pair of beats (1 and 3 versus 2 and 4)?

Usage: .venv-sep/Scripts/python.exe scripts/diag/downbeat_diag.py <model.pt> [--channels 48]
"""

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from beatnet import BeatNet  # noqa: E402


def snare_grid(track: str) -> dict | None:
    """Share of snare hits per slot of the MIDI's own bar, for one 4/4 song."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(track)
    ts = pm.time_signature_changes
    if ts and not all(t.numerator == 4 and t.denominator == 4 for t in ts):
        return None
    beats, downs = pm.get_beats(), pm.get_downbeats()
    if len(beats) < 32 or len(downs) < 4:
        return None
    # Slot of each beat inside its bar.
    idx = np.searchsorted(downs, beats, side="right") - 1
    slot = np.array([np.searchsorted(beats, b) - np.searchsorted(beats, downs[max(i, 0)]) for b, i in zip(beats, idx)]) % 4
    counts = np.zeros(4)
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            if n.pitch not in (37, 38, 39, 40):
                continue
            k = int(np.clip(np.searchsorted(beats, n.start), 1, len(beats) - 1))
            k = k - 1 if abs(n.start - beats[k - 1]) < abs(n.start - beats[k]) else k
            if abs(n.start - beats[k]) < 0.07:
                counts[slot[k]] += 1
    total = counts.sum()
    if total < 16:
        return None
    return {"backbeat_share": float((counts[1] + counts[3]) / total), "offbeat_share": float((counts[0] + counts[2]) / total)}


@torch.no_grad()
def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    pt = Path(args[0])
    channels = int(args[args.index("--channels") + 1]) if "--channels" in args else 48
    dil = [int(d) for d in args[args.index("--dilations") + 1].split(",")] if "--dilations" in args else [1, 2, 4, 8, 16, 32, 64, 128]

    model = BeatNet(channels=channels, dilations=dil).cuda().eval()
    model.load_state_dict(torch.load(pt, map_location="cuda"))

    feat = ROOT / "data" / "features" / "validation"
    midi = ROOT / "data" / "slakh" / "validation"
    files = sorted(feat.glob("*_full.npz"))
    names = [f.stem.split("_")[0] for f in files]

    with ProcessPoolExecutor(12) as pool:
        grids = list(pool.map(snare_grid, [str(midi / n / "all_src.mid") for n in names], chunksize=4))

    horizons = (8, 16, 32, 64)
    hit = {h: [] for h in horizons}
    pair = {h: [] for h in horizons}
    per_song = []
    good_grid = 0
    for f, name, g in zip(files, names, grids):
        z = np.load(f)
        if not bool(z["down_ok"]) or g is None or not (50 <= float(z["bpm"]) <= 220):
            continue
        if g["backbeat_share"] < 0.7:
            continue                    # bar lines disagree with the drummer: label unusable
        good_grid += 1
        x = torch.from_numpy(np.load(f.with_suffix(".x.npy")).astype(np.float32))[None].cuda()
        act = torch.sigmoid(model(x)[0, :, 1]).cpu().numpy()
        beat_f = np.nonzero(z["beat"])[0]
        down_f = set(np.nonzero(z["down"])[0].tolist())
        if len(beat_f) < 80:
            continue
        # True slot of each beat, from the first labelled downbeat.
        first = next((i for i, b in enumerate(beat_f) if int(b) in down_f), None)
        if first is None:
            continue
        # Causal evidence per slot, starting 5 s in as the drummer would.
        begin = int(np.searchsorted(beat_f, 500))
        score = np.zeros(4)
        for k in range(begin, len(beat_f)):
            b = beat_f[k]
            score[(k - first) % 4] += act[max(b - 2, 0):b + 5].max()
            n = k - begin + 1
            if n in horizons:
                best = int(np.argmax(score))
                hit[n].append(best == 0)
                pair[n].append(best % 2 == 0)
        per_song.append(name)

    usable = sum(g is not None for g in grids)
    print(f"songs with a 4/4 drum part to check: {usable}; snare on the MIDI's 2 and 4: {good_grid} ({good_grid / max(usable, 1):.0%})")
    print(f"scored {len(per_song)} songs with perfect beats given to the head:")
    print("beats heard   right slot   right pair (1+3 vs 2+4)")
    for h in horizons:
        if hit[h]:
            print(f"{h:>10d}   {np.mean(hit[h]):>9.0%}   {np.mean(pair[h]):>9.0%}   (n={len(hit[h])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
