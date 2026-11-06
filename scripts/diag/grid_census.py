"""
grid_census.py - compare each song's MIDI bar lines with what its drum part
actually does, and write the verdict per song to data/features/grid.json.

MIDI files record what the sequencer's grid said, not what the music does, so
a bar line can sit a beat away from the real "1". The drum part settles it:
in a groove the snare marks 2 and 4. Songs are sorted by where the snare sits:

  backbeat   on the MIDI's 2 and 4: the bar lines can be trusted
  shifted    on the MIDI's 1 and 3: bars are one beat off
  half       on the MIDI's 3 only: half-time feel
  evens      on the "&" of the MIDI's beats: the MIDI pulse is half the felt pulse
  every      on all four beats, or no clear pattern
  none       too few snare hits (or not 4/4) to judge

Usage: .venv-sep/Scripts/python.exe scripts/diag/grid_census.py [splits...]
"""

import json
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNARES = (37, 38, 39, 40)   # side stick, acoustic snare, clap, electric snare


def census(track: str) -> dict:
    """Verdict for one song. Never raises: a broken file is simply 'none'."""
    import pretty_midi
    warnings.simplefilter("ignore")

    out = {"kind": "none", "slots": [0, 0, 0, 0], "and_share": 0.0, "first_hit": None}
    try:
        pm = pretty_midi.PrettyMIDI(str(Path(track) / "all_src.mid"))
        ts = pm.time_signature_changes
        out["four_four"] = bool(not ts or all(t.numerator == 4 and t.denominator == 4 for t in ts))
        beats, downs = pm.get_beats(), pm.get_downbeats()
        if not out["four_four"] or len(beats) < 32 or len(downs) < 4:
            return out

        # Slot in the bar for every beat. Bar length is constant in 4/4, so the
        # first downbeat anchors the whole song.
        anchor = int(np.argmin(np.abs(beats - downs[0])))
        hits = np.array(sorted(n.start for i in pm.instruments if i.is_drum for n in i.notes if n.pitch in SNARES))
        if hits.size < 16:
            return out
        out["first_hit"] = float(hits[0])

        # Position of each hit in beats: whole part = beat index, fraction = where inside it.
        pos = np.interp(hits, beats, np.arange(len(beats), dtype=np.float64))
        frac = pos - np.floor(pos)
        on_beat = (frac < 0.12) | (frac > 0.88)
        on_and = np.abs(frac - 0.5) < 0.12
        slots = np.zeros(4)
        for p in pos[on_beat]:
            slots[(int(np.rint(p)) - anchor) % 4] += 1
        grid_hits = on_beat.sum() + on_and.sum()
        out["slots"] = [int(v) for v in slots]
        out["and_share"] = float(on_and.sum() / max(grid_hits, 1))

        total = slots.sum()
        if total < 16 and on_and.sum() < 16:
            return out
        if out["and_share"] > 0.6:
            out["kind"] = "evens"
        elif (slots[1] + slots[3]) / total >= 0.7:
            out["kind"] = "backbeat"
        elif slots[2] / total >= 0.7:
            out["kind"] = "half"
        elif (slots[0] + slots[2]) / total >= 0.7:
            out["kind"] = "shifted"
        else:
            out["kind"] = "every"
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


def main() -> int:
    splits = sys.argv[1:] or ["train", "validation"]
    grid = {}
    with ProcessPoolExecutor(12) as pool:
        for split in splits:
            tracks = sorted(p for p in (ROOT / "data" / "slakh" / split).iterdir() if p.is_dir())
            res = list(pool.map(census, [str(t) for t in tracks], chunksize=8))
            for t, r in zip(tracks, res):
                grid[t.name] = {**r, "split": split}
            c = Counter(r["kind"] for r in res)
            print(f"{split:<11} {len(res)} songs  " + "  ".join(f"{k} {c[k]} ({c[k] / len(res):.0%})" for k in
                  ("backbeat", "shifted", "half", "evens", "every", "none")))
    (ROOT / "data" / "features" / "grid.json").write_text(json.dumps(grid, indent=0), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
