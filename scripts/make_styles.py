"""
make_styles.py - write the genre style files in weights/styles/.

Usage:
    python scripts/make_styles.py

Each style file says "extends": "../drummer.json", so it inherits every
tempo / phase / intensity weight and only replaces the groove: all ten drums
are written in full (a drum a style does not use is set to "never"), plus
the feel (swing) and fill policy.

Step grid (16 sixteenths per bar):
    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15
    1 e & a | 2 e & a | 3 e &  a  | 4  e  &  a
A cell plays while  on <= intensity < off.  on = 2 means never.
"""

import json
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRUMS = ["kick", "snare", "closedHat", "openHat", "crash", "ride", "tomHi", "tomMid", "tomLo", "pedalHat"]
NOTES = [36, 38, 42, 46, 49, 51, 50, 47, 45, 44]
NEVER, TOP = 2.0, 2.0


def groove(cells):
    """cells: list of (drum, steps, on, off, velocity or [velocities])."""
    g = {d: {"note": n, "on": [NEVER] * 16, "off": [TOP] * 16, "velocity": [0.0] * 16} for d, n in zip(DRUMS, NOTES)}
    for drum, steps, on, off, vel in cells:
        vels = vel if isinstance(vel, list) else [vel] * len(steps)
        for s, v in zip(steps, vels):
            g[drum]["on"][s], g[drum]["off"][s], g[drum]["velocity"][s] = on, off, float(v)
    return g


BEATS = [0, 4, 8, 12]
EIGHTHS = [0, 2, 4, 6, 8, 10, 12, 14]
OFFS = [2, 6, 10, 14]
SIXTEENTHS = list(range(16))

STYLES = {
    # ------------------------------------------------------------------
    "rock": {
        "about": "Straight rock: the stock groove (kick 1 & 3, snare 2 & 4, hats to ride as it builds).",
        "groove": None,   # inherit the hand-written defaults unchanged
    },
    # ------------------------------------------------------------------
    "alt_rock": {
        "about": "Alt rock / grunge: hard backbeat, driving kick, eighths that wash from closed hat "
                 "to half-open hat to ride, hi-hat foot under the wash, tom fills come earlier.",
        "groove": groove([
            ("kick", [0, 8], 0.0, TOP, [118, 110]),
            ("kick", [10], 0.30, TOP, 100),
            ("kick", [2], 0.50, TOP, 96),
            ("kick", [14], 0.75, TOP, 98),
            ("snare", [4, 12], 0.0, TOP, [116, 120]),
            ("snare", [7], 0.60, TOP, 30),
            ("closedHat", EIGHTHS, 0.0, 0.60, [92, 66, 86, 66, 90, 66, 86, 66]),
            ("openHat", EIGHTHS, 0.60, 0.85, [86, 70, 82, 70, 86, 70, 82, 70]),
            ("ride", EIGHTHS, 0.85, TOP, [104, 84, 100, 84, 104, 84, 100, 84]),
            ("pedalHat", [4, 12], 0.85, TOP, 70),
        ]),
        "fills": {"tomLevel": 0.50, "longLevel": 0.65, "snareLevel": 0.30, "crashLevel": 0.30},
    },
    # ------------------------------------------------------------------
    "pop": {
        "about": "Pop: tight and busy hats (eighths, then sixteenths), syncopated kick, clean "
                 "backbeat, open hats on the '&' in the chorus, toms only at the very top.",
        "groove": groove([
            ("kick", [0, 8], 0.0, TOP, [108, 100]),
            ("kick", [6], 0.30, TOP, 92),
            ("kick", [10], 0.60, TOP, 88),
            ("snare", [4, 12], 0.0, TOP, [104, 106]),
            ("snare", [15], 0.70, TOP, 34),
            ("closedHat", EIGHTHS, 0.0, 0.40, [84, 64, 80, 64, 84, 64, 80, 64]),
            ("closedHat", SIXTEENTHS, 0.40, 0.85, [86, 44, 68, 44] * 4),
            ("closedHat", BEATS, 0.85, TOP, 88),
            ("openHat", OFFS, 0.85, TOP, 82),
        ]),
        "fills": {"tomLevel": 0.85, "longLevel": 0.92, "snareLevel": 0.45, "crashLevel": 0.40},
    },
    # ------------------------------------------------------------------
    "jazz": {
        "about": "Medium swing: full triplet swing, spang-a-lang ride, hi-hat foot on 2 and 4, "
                 "feathered kick on every beat, snare comping on swung upbeats, no toms.",
        "swing8": 1.0,
        "velLevelGain": 25.0,
        "groove": groove([
            ("ride", [0, 4, 6, 8, 12, 14], 0.0, TOP, [88, 100, 68, 88, 100, 68]),
            ("pedalHat", [4, 12], 0.0, TOP, 72),
            ("kick", BEATS, 0.0, 0.70, 26),
            ("kick", [14], 0.70, TOP, 92),
            ("snare", [6], 0.30, TOP, 44),
            ("snare", [14], 0.55, TOP, 50),
            ("snare", [10], 0.75, TOP, 40),
        ]),
        "fills": {"tomLevel": NEVER, "longLevel": NEVER, "snareLevel": 0.50, "crashLevel": 0.60},
    },
}


def main() -> int:
    out_dir = ROOT / "weights" / "styles"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, st in STYLES.items():
        doc = {"format": "drum_along.weights", "version": 1, "extends": "../drummer.json",
               "about": st["about"], "groove": {}}
        if "swing8" in st:
            doc["groove"]["swing8"] = st["swing8"]
        if "velLevelGain" in st:
            doc["groove"]["velLevelGain"] = st["velLevelGain"]
        if st["groove"] is not None:
            doc["groove"]["drums"] = st["groove"]
        if "fills" in st:
            doc["fills"] = st["fills"]
        if not doc["groove"]:
            del doc["groove"]

        # One drum pattern per line, like a sequencer row: fold every array of
        # plain numbers onto a single line.
        text = json.dumps(doc, indent=2)
        text = re.sub(r"\[\s+([-0-9.,\s]+?)\s+\]",
                      lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text)
        (out_dir / f"{name}.json").write_text(text + "\n", encoding="utf-8")
        print(f"wrote weights/styles/{name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
