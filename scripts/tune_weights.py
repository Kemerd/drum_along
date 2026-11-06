"""
tune_weights.py - the hand-tuning loop, one weight at a time.

Usage (training env):
    .venv-sep/Scripts/python.exe scripts/tune_weights.py <split_dir> <base_weights.json> <out_weights.json>
        --try phase.leadership=0.3,0.5,0.7 --try beatNet.onsetMix=0.5,0.75,1
        [--limit 120] [--workers 14] [--rounds 1]

For each --try line, every candidate value is scored on the same songs with
the whole drummer, the best one is kept, and the next line starts from it
(coordinate descent). Every run is appended to output/tune_log.jsonl.

The score is  J = F + 0.15 * down + 0.15 * pair - 0.02 * pause - 0.003 * lurch :
beat F already pays for wrong tempo and wrong phase, and the next terms pay for
the bar: the "1" landing on the band's "1", and above all the snare landing on
2 and 4, since a backbeat on 1 and 3 sounds wrong even with every beat in place.
The last two pay for steadiness. Beat F barely notices the clock jumping half a
beat and back, yet that is the first thing a listener hears.

Tuning stays on the validation split. The test split and the personal test
songs are never passed here, so they stay a fair final exam.
"""

import copy
import json
import sys
import tempfile
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dataset import ROOT, score_song  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "train"))
from run_pipeline import dump_compact  # noqa: E402

LOG = ROOT / "output" / "tune_log.jsonl"


def get_path(doc: dict, path: str):
    """Value at a dotted path, or raise a readable error for a mistyped weight."""
    node = doc
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(f"error: no weight named {path!r} (stopped at {part!r})")
        node = node[part]
    return node


def set_path(doc: dict, path: str, value) -> None:
    """Write a value at a dotted path that get_path has already validated."""
    parts = path.split(".")
    node = doc
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def parse_value(text: str, like):
    """Candidate value with the same type as the existing weight."""
    return int(float(text)) if isinstance(like, int) and not isinstance(like, bool) else float(text)


def summarize(results: list[dict]) -> dict:
    """Averages over the scored songs, including J."""
    results = [r for r in results if r]                 # None = song out of scope (not 4/4)
    ok = [r for r in results if "error" not in r]
    if not ok:
        reason = next((r["error"] for r in results if "error" in r), "no songs in scope")
        raise SystemExit(f"error: no song could be scored: {reason}")

    def mean(key):
        vals = [r[key] for r in ok if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else 0.0

    tempo_ok = sum(r["tempo"] == "correct" for r in ok) / len(ok)
    s = {"songs": len(ok), "errors": len(results) - len(ok), "F": mean("F"), "on": mean("on"), "off": mean("off"),
         "down": mean("down"), "pair": mean("pair"), "pause": mean("pause"), "lurch": mean("lurch"), "tempo_ok": tempo_ok, "start": mean("start"),
         "offset": float(np.median([r["offset"] for r in ok]))}
    s["J"] = s["F"] + 0.15 * s["down"] + 0.15 * s["pair"] - 0.02 * s["pause"] - 0.003 * s["lurch"]
    return s


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 3:
        print(__doc__)
        return 2
    split_dir, base_path, out_path = Path(args[0]), Path(args[1]).resolve(), Path(args[2]).resolve()
    trials, opt = [], {"limit": 120, "workers": 14, "rounds": 1, "variant": "full"}
    rest = args[3:]
    while rest:
        a = rest.pop(0)
        if a == "--try" and rest:
            name, _, values = rest.pop(0).partition("=")
            trials.append((name, values.split(",")))
        elif a.lstrip("-") in opt and rest:
            key = a.lstrip("-")
            opt[key] = type(opt[key])(rest.pop(0))
    if not trials:
        raise SystemExit("error: nothing to try; pass at least one --try section.name=a,b,c")

    best = json.loads(base_path.read_text(encoding="utf-8"))
    # A relative beatNet.file must keep resolving from the temporary copy.
    net = best.get("beatNet", {}).get("file", "")
    if net:
        best["beatNet"]["file"] = str((base_path.parent / net).resolve())

    # Catch a mistyped weight now, not an hour into the sweep.
    for name, _ in trials:
        get_path(best, name)

    tracks = sorted(p for p in split_dir.iterdir() if p.is_dir() and (p / "metadata.yaml").exists())
    if opt["limit"]:
        # Evenly spread through the split rather than its first N songs.
        step = max(1, len(tracks) // opt["limit"])
        tracks = tracks[::step][: opt["limit"]]
    print(f"{len(tracks)} songs, {len(trials)} weights, {opt['rounds']} round(s)", flush=True)

    def save(doc: dict) -> None:
        """Write the running best next to the base, with a portable network path."""
        doc = copy.deepcopy(doc)
        if net:
            doc["beatNet"]["file"] = net
        out_path.write_text(dump_compact(doc), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="tune_") as tmp, ProcessPoolExecutor(opt["workers"]) as pool:
        tmp = Path(tmp)

        def run(doc: dict) -> dict:
            path = tmp / "w.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            n = len(tracks)
            return summarize(list(pool.map(score_song, [str(t) for t in tracks], [opt["variant"]] * n, [str(path)] * n)))

        t0 = time.time()
        top = run(best)
        print(f"base  J {top['J']:.4f}  {top}", flush=True)

        for round_no in range(1, opt["rounds"] + 1):
            for name, values in trials:
                current = get_path(best, name)
                chosen = current
                for text in values:
                    value = parse_value(text, current)
                    if value == current:
                        continue            # the running best already scored it
                    doc = copy.deepcopy(best)
                    set_path(doc, name, value)
                    # An out-of-range candidate is a typo in the sweep, not a reason to lose it.
                    try:
                        s = run(doc)
                    except SystemExit as exc:
                        print(f"r{round_no} {name}={value:<10g} rejected: {exc}", flush=True)
                        continue
                    won = s["J"] > top["J"] + 1e-4      # ignore noise-level wins
                    print(f"r{round_no} {name}={value:<10g} J {s['J']:.4f}  F {s['F']:.3f}  on {s['on']:.0%}  "
                          f"down {s['down']:.0%}  pair {s['pair']:.0%}  pause {s['pause']:.1f}  lurch {s['lurch']:.1f}  tempo {s['tempo_ok']:.0%}  start {s['start']:.1f}"
                          f"{'  *' if won else ''}", flush=True)
                    with open(LOG, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"weight": name, "value": value, **s}) + "\n")
                    if won:
                        top, chosen = s, value
                set_path(best, name, chosen)
                save(best)                  # a stopped sweep keeps every win so far

        print(f"\nfinal J {top['J']:.4f}  {top}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    save(best)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
