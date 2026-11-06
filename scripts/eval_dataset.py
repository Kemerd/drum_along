"""
eval_dataset.py - score the WHOLE drummer (not just the learned layer) on many
songs with exact MIDI ground truth. This is the tuning loop for hand-edited
weights: change weights/*.json, rerun, compare the summary line.

Usage (training env: pretty_midi + mir_eval):
    .venv-sep/Scripts/python.exe scripts/eval_dataset.py <backings_split_dir> [--weights W.json]
        [--limit N] [--workers 24] [--variant full] [--tag name]

For each <Track>/backing_<variant>.flac + all_src.mid it runs wav2mid on the
drumless backing and compares the quarter notes the drummer actually PLAYED
against the MIDI beats:

  F        beat F-measure, +-70 ms
  on/&     share of played beats on the real beat / on the off-beat "&"
  tempo    correct / double / half / other (median played vs MIDI tempo, 4%)
  down     share of played "1"s that land on a real downbeat (+-70 ms)
  pair     share of played "1"s and "3"s on a real 1 or 3: the snare is on the
           right two beats, even if the bar is two beats off
  offset   median signed timing error of matched beats (ms, + = late)
  pause    clock jumps of a quarter beat or more, per minute. A listener hears
           each one as the drummer stumbling, so one matters more than a
           percent of F, which barely notices it
  lurch    smaller clock moves (8% of a beat or more), per minute
  ghost    share of the time with no drummer in the song that we played anyway
  missed   share of the time with a drummer in the song that we sat out
  entry    for songs whose drums come in 10 s or more after the music starts:
           seconds from the real drum entry to ours (negative = we came in early)
  start    seconds until the drummer starts
"""

import json
import subprocess
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WAV2MID = ROOT / "build" / "Release" / "wav2mid.exe"


def score_song(track_dir: str, variant: str, weights: str | None) -> dict | None:
    import csv
    import mir_eval
    import pretty_midi
    import soundfile as sf

    track_dir = Path(track_dir)
    flac = track_dir / f"backing_{variant}.flac"
    mid = track_dir / "all_src.mid"
    if not flac.exists() or not mid.exists():
        return None

    pm = pretty_midi.PrettyMIDI(str(mid))
    ts = pm.time_signature_changes
    if ts and not all(t.numerator == 4 and t.denominator == 4 for t in ts):
        return None                                     # the drummer assumes 4/4
    ref = np.asarray(pm.get_beats())
    ref_down = np.asarray(pm.get_downbeats())
    if ref.size < 16:
        return None
    ref_bpm = 60.0 / float(np.median(np.diff(ref)))
    if not (50.0 <= ref_bpm <= 220.0):
        return None

    with tempfile.TemporaryDirectory(prefix="eval_") as tmp:
        tmp = Path(tmp)
        audio, sr = sf.read(str(flac), dtype="int16", always_2d=True)
        sf.write(str(tmp / "in.wav"), audio, sr, subtype="PCM_16")
        cmd = [str(WAV2MID), str(tmp / "in.wav"), str(tmp / "run")]
        if weights:
            cmd += ["--weights", weights]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return {"track": track_dir.name, "error": res.stderr.strip()[:200]}
        rows = list(csv.DictReader(open(tmp / "run_steps.csv", encoding="utf-8")))

    played = [r for r in rows if r["playing"] == "1" and int(r["step_in_bar"]) % 4 == 0]
    est = np.array([float(r["time_s"]) for r in played])
    est_down = np.array([float(r["time_s"]) for r in played if int(r["step_in_bar"]) == 0])
    out = {"track": track_dir.name, "ref_bpm": ref_bpm, "played": int(est.size)}
    if est.size < 8:
        out.update(F=0.0, on=0.0, off=0.0, tempo="none", down=0.0, pair=0.0, offset=0.0, start=float("nan"),
                   pause=0.0, lurch=0.0)
        return out

    start = float(est[0])
    r = ref[ref >= start - 0.1]
    out["F"] = float(mir_eval.beat.f_measure(r, est))
    out["start"] = start

    # Placement inside the real beat: 0 = on it, 0.5 = on the "&".
    k = np.clip(np.searchsorted(ref, est) - 1, 0, ref.size - 2)
    ph = (est - ref[k]) / (ref[k + 1] - ref[k])
    ph = np.where(ph > 0.75, ph - 1.0, ph)
    out["on"] = float(np.mean(np.abs(ph) < 0.12))
    out["off"] = float(np.mean(np.abs(ph - 0.5) < 0.12))

    # Tempo class from the median played interval.
    our_bpm = 60.0 / float(np.median(np.diff(est)))
    ratio = our_bpm / ref_bpm
    cls = "other"
    for name, target in (("correct", 1.0), ("double", 2.0), ("half", 0.5)):
        if abs(ratio / target - 1.0) < 0.04:
            cls = name
    out["tempo"], out["our_bpm"] = cls, our_bpm

    # Steadiness: a beat interval far from its neighbours means the clock was moved.
    iv = np.diff(est)
    local = np.array([np.median(iv[max(i - 4, 0):i + 5]) for i in range(iv.size)])
    dev = np.abs(iv - local) / np.maximum(local, 1e-6)
    dev = dev[dev < 0.75]                               # longer gaps are the drummer laying out
    minutes = max((est[-1] - est[0]) / 60.0, 1e-6)
    out["pause"] = float(np.sum(dev >= 0.25)) / minutes
    out["lurch"] = float(np.sum(dev >= 0.08)) / minutes

    # Sitting out: compare with the song's own drum part, 10 times a second.
    hits = np.array(sorted(n.start for inst in pm.instruments if inst.is_drum for n in inst.notes))
    if hits.size >= 2:
        first_note = min((n.start for inst in pm.instruments if not inst.is_drum for n in inst.notes), default=0.0)
        if hits[0] - first_note >= 10.0:
            out["entry"] = float(start - hits[0])
        grid = np.arange(start, est[-1], 0.1)
        j = np.clip(np.searchsorted(hits, grid), 1, hits.size - 1)
        drums_on = np.minimum(np.abs(grid - hits[j - 1]), np.abs(grid - hits[j])) <= 1.5
        k = np.clip(np.searchsorted(est, grid), 1, est.size - 1)
        we_play = np.minimum(np.abs(grid - est[k - 1]), np.abs(grid - est[k])) <= 1.0
        out["ghost"] = float(np.mean(we_play[~drums_on])) if (~drums_on).any() else float("nan")
        out["missed"] = float(np.mean(~we_play[drums_on])) if drums_on.any() else float("nan")

    # Backbeat pair: every real beat 0 or 2 of its bar, from the song's own bar lines.
    est_pair = np.array([float(r["time_s"]) for r in played if int(r["step_in_bar"]) % 8 == 0])
    slot = np.array([np.searchsorted(ref, b) - np.searchsorted(ref, ref_down[max(np.searchsorted(ref_down, b, side="right") - 1, 0)])
                     for b in ref]) % 4 if ref_down.size else np.ones(ref.size, dtype=int)
    ref_pair = ref[slot % 2 == 0]
    if est_pair.size and ref_pair.size >= 2:
        j = np.clip(np.searchsorted(ref_pair, est_pair), 1, ref_pair.size - 1)
        near = np.minimum(np.abs(est_pair - ref_pair[j - 1]), np.abs(est_pair - ref_pair[j]))
        out["pair"] = float(np.mean(near <= 0.07))
    else:
        out["pair"] = 0.0

    # Downbeats: share of played "1"s within 70 ms of a real downbeat.
    if est_down.size and ref_down.size:
        j = np.clip(np.searchsorted(ref_down, est_down), 1, ref_down.size - 1)
        near = np.minimum(np.abs(est_down - ref_down[j - 1]), np.abs(est_down - ref_down[j]))
        out["down"] = float(np.mean(near <= 0.07))
    else:
        out["down"] = 0.0

    # Signed timing error of beats that matched (+ = we are late).
    j = np.clip(np.searchsorted(ref, est), 1, ref.size - 1)
    nearest = np.where(np.abs(est - ref[j - 1]) < np.abs(est - ref[j]), ref[j - 1], ref[j])
    d = est - nearest
    d = d[np.abs(d) <= 0.07]
    out["offset"] = 1000.0 * float(np.median(d)) if d.size else 0.0
    return out


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    split_dir = Path(args.pop(0))
    opt = {"weights": None, "limit": 0, "workers": 24, "variant": "full", "tag": ""}
    while args:
        a = args.pop(0).lstrip("-")
        if a in opt and args:
            v = args.pop(0)
            opt[a] = int(v) if a in ("limit", "workers") else v

    tracks = sorted(p for p in split_dir.iterdir() if p.is_dir() and (p / "metadata.yaml").exists())
    if opt["limit"]:
        tracks = tracks[: opt["limit"]]

    with ProcessPoolExecutor(max_workers=opt["workers"]) as pool:
        results = [r for r in pool.map(score_song, [str(t) for t in tracks],
                                       [opt["variant"]] * len(tracks), [opt["weights"]] * len(tracks)) if r]
    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("no songs scored", errors[:3])
        return 1

    def mean(key):
        vals = [r[key] for r in ok if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    tempo = {c: sum(r["tempo"] == c for r in ok) / len(ok) for c in ("correct", "double", "half", "other", "none")}
    summary = {
        "tag": opt["tag"], "weights": opt["weights"], "songs": len(ok), "errors": len(errors),
        "F": mean("F"), "on": mean("on"), "off": mean("off"), "down": mean("down"), "pair": mean("pair"),
        "pause": mean("pause"), "lurch": mean("lurch"), "ghost": mean("ghost"), "missed": mean("missed"),
        "entry": float(np.median([r["entry"] for r in ok if "entry" in r])) if any("entry" in r for r in ok) else float("nan"),
        "entry_songs": sum("entry" in r for r in ok),
        "offset_ms": float(np.median([r["offset"] for r in ok])), "start_s": mean("start"), "tempo": tempo,
    }
    print(f"{opt['tag'] or 'run':<14} songs {len(ok):3d}  F {summary['F']:.3f}  on {summary['on']:.0%}  "
          f"& {summary['off']:.0%}  down {summary['down']:.0%}  pair {summary['pair']:.0%}  pause {summary['pause']:.1f}/min  lurch {summary['lurch']:.1f}/min  ghost {summary['ghost']:.0%}  missed {summary['missed']:.0%}  entry {summary['entry']:+.1f} s ({summary['entry_songs']} songs)  tempo ok {tempo['correct']:.0%} "
          f"(x2 {tempo['double']:.0%}, x0.5 {tempo['half']:.0%}, other {tempo['other']:.0%})  "
          f"offset {summary['offset_ms']:+.1f} ms  start {summary['start_s']:.1f} s")
    log = ROOT / "output" / "eval_log.jsonl"
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({**summary, "per_song": ok}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
