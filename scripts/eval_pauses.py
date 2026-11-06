"""
eval_pauses.py - does the drummer survive a pause in the band?

Usage (training env):
    .venv-sep/Scripts/python.exe scripts/eval_pauses.py <weights.json> [--songs 24] [--workers 14] [--tag name]

Beat tracking scores miss this, because most songs never stop. Two kinds of
test song are cut from the validation split, with exact beats from the MIDI:

  hold    One song with a full stop of 2 to 4 beats every 30 s. The band comes
          back exactly on the grid, as a band does after a break. The drummer
          must stop with them and come back on the same beat: its clock must
          not move at all.

  shift   60 s of song A, 1.5 s of silence, then song B at a different tempo
          (at least 10% away). The drummer must let go of the old tempo and
          find the new one.

Reported:
  hold  stumbles   clock jumps per pause (a quarter beat or more; goal 0)
        on-beat    share of played beats on the real beat, 8 s after each pause
        back in    seconds from the band returning to the drummer playing again
  shift found      share of songs where it reaches the new tempo at all
        in         median seconds from B starting to 8 of 10 beats landing right

Songs are picked from the ones the drummer already follows well, so a failure
here is a failure of pause handling and not of beat tracking in general.
"""

import csv
import json
import subprocess
import sys
import tempfile
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WAV2MID = ROOT / "build" / "Release" / "wav2mid.exe"
VAL = ROOT / "data" / "slakh" / "validation"
WINDOW_S = 0.07


def load_song(track: str):
    """Audio plus MIDI beats and downbeats for one validation song."""
    import pretty_midi
    import soundfile as sf

    warnings.simplefilter("ignore")
    y, sr = sf.read(str(VAL / track / "backing_full.flac"), dtype="float32", always_2d=True)
    pm = pretty_midi.PrettyMIDI(str(VAL / track / "all_src.mid"))
    return y.mean(axis=1), sr, np.asarray(pm.get_beats()), np.asarray(pm.get_downbeats())


def run_drummer(y: np.ndarray, sr: int, weights: str) -> tuple[np.ndarray, np.ndarray]:
    """Play audio through the drummer. Returns played quarter-note times and every played step."""
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="pause_") as tmp:
        tmp = Path(tmp)
        sf.write(str(tmp / "in.wav"), y, sr, subtype="PCM_16")
        res = subprocess.run([str(WAV2MID), str(tmp / "in.wav"), str(tmp / "run"), "--weights", weights],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip()[:200])
        with open(tmp / "run_steps.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    played = [r for r in rows if r["playing"] == "1"]
    quarters = np.array([float(r["time_s"]) for r in played if int(r["step_in_bar"]) % 4 == 0])
    steps = np.array([float(r["time_s"]) for r in played])
    return quarters, steps


def clock_jumps(est: np.ndarray) -> np.ndarray:
    """Times where a beat interval sits a quarter beat or more from its neighbours."""
    if est.size < 8:
        return np.zeros(0)
    iv = np.diff(est)
    local = np.array([np.median(iv[max(i - 4, 0):i + 5]) for i in range(iv.size)])
    dev = np.abs(iv - local) / np.maximum(local, 1e-6)
    return est[1:][(dev >= 0.25) & (dev < 0.75)]


def on_beat(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """True for each played time within the tolerance of a real beat."""
    if est.size == 0 or ref.size < 2:
        return np.zeros(est.size, dtype=bool)
    j = np.clip(np.searchsorted(ref, est), 1, ref.size - 1)
    return np.minimum(np.abs(est - ref[j - 1]), np.abs(est - ref[j])) <= WINDOW_S


def hold_case(track: str, seed: int, weights: str) -> dict:
    """One song with stops every 30 s; the grid carries on underneath."""
    y, sr, beats, downs = load_song(track)
    rng = np.random.default_rng(seed)
    fade = int(0.01 * sr)
    y = y.copy()
    returns = []
    k = int(np.searchsorted(beats, 30.0))
    while k + 8 < beats.size and beats[k] < 150.0:
        # Stops start on a beat and end on a beat, like a written band break.
        n = int(rng.integers(2, 5))
        a, b = int(beats[k] * sr), int(beats[k + n] * sr)
        y[a:a + fade] *= np.linspace(1, 0, fade, dtype=np.float32)[: len(y[a:a + fade])]
        y[a + fade:b] = 0.0
        returns.append(float(beats[k + n]))
        k += int(np.searchsorted(beats[k:], beats[k] + 30.0))
    est, _ = run_drummer(y, sr, weights)
    jumps = clock_jumps(est)
    after, back = [], []
    for r in returns:
        m = (est >= r) & (est < r + 8.0)
        if m.any():
            after.append(float(on_beat(est[m], beats).mean()))
            back.append(float(est[m][0] - r))
        else:
            after.append(0.0)
            back.append(8.0)
    return {"kind": "hold", "track": track, "pauses": len(returns), "jumps": int(jumps.size),
            "after": float(np.mean(after)) if after else 0.0, "back": float(np.mean(back)) if back else 0.0}


def shift_case(track_a: str, track_b: str, weights: str) -> dict:
    """60 s of A, a rest, then B from one of its bar lines. B's beats are the truth."""
    ya, sr, beats_a, downs_a = load_song(track_a)
    yb, sr_b, beats_b, downs_b = load_song(track_b)
    if sr_b != sr:
        return {"kind": "shift", "skip": True}
    cut_a = float(downs_a[int(np.searchsorted(downs_a, 60.0))])
    start_b = float(downs_b[int(np.searchsorted(downs_b, 30.0))])
    gap = int(1.5 * sr)
    head = ya[: int(cut_a * sr)]
    tail = yb[int(start_b * sr):]
    y = np.concatenate([head, np.zeros(gap, np.float32), tail])
    t_join = (len(head) + gap) / sr
    ref = beats_b[beats_b >= start_b] - start_b + t_join

    est, _ = run_drummer(y, sr, weights)
    est = est[est >= t_join]
    ok = on_beat(est, ref)
    found_at = None
    for i in range(0, max(ok.size - 9, 0)):
        if ok[i:i + 10].sum() >= 8:
            found_at = float(est[i] - t_join)
            break
    return {"kind": "shift", "track": f"{track_a}+{track_b}", "found": found_at is not None,
            "seconds": found_at if found_at is not None else float("nan")}


def good_songs(n: int) -> tuple[list[str], dict]:
    """Validation songs the drummer already follows well, from the latest full evaluation."""
    log = [json.loads(line) for line in open(ROOT / "output" / "eval_log.jsonl", encoding="utf-8")]
    rows = [r for r in log if r.get("songs", 0) >= 150][-1]["per_song"]
    good = [r for r in rows if r.get("F", 0) >= 0.85 and r.get("tempo") == "correct"]
    good = sorted(good, key=lambda r: r["track"])
    step = max(1, len(good) // max(n, 1))
    picked = good[::step][:n]
    return [r["track"] for r in picked], {r["track"]: r["ref_bpm"] for r in rows}


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    weights = str(Path(args[0]).resolve())
    opt = {"songs": 24, "workers": 14, "tag": ""}
    rest = args[1:]
    while rest:
        key = rest.pop(0).lstrip("-")
        if key in opt and rest:
            opt[key] = type(opt[key])(rest.pop(0))

    tracks, bpm = good_songs(opt["songs"])
    # Pair each song with the next one at least 10% away in tempo.
    pairs = []
    for i, a in enumerate(tracks):
        for j in range(1, len(tracks)):
            b = tracks[(i + j) % len(tracks)]
            if abs(bpm[b] / bpm[a] - 1.0) >= 0.10:
                pairs.append((a, b))
                break

    with ProcessPoolExecutor(opt["workers"]) as pool:
        holds = list(pool.map(hold_case, tracks, range(len(tracks)), [weights] * len(tracks)))
        shifts = list(pool.map(shift_case, [a for a, _ in pairs], [b for _, b in pairs], [weights] * len(pairs)))

    pauses = sum(h["pauses"] for h in holds)
    jumps = sum(h["jumps"] for h in holds)
    done = [s for s in shifts if not s.get("skip")]
    found = [s for s in done if s["found"]]
    secs = np.array([s["seconds"] for s in found]) if found else np.array([float("nan")])
    print(f"{opt['tag'] or 'run':<14} hold: {pauses} pauses, {jumps / max(pauses, 1):.2f} stumbles per pause, "
          f"on-beat after {np.mean([h['after'] for h in holds]):.0%}, back in {np.mean([h['back'] for h in holds]):.1f} s   "
          f"shift: found {len(found) / max(len(done), 1):.0%} of {len(done)}, in {np.median(secs):.1f} s median")
    with open(ROOT / "output" / "pause_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"tag": opt["tag"], "weights": weights, "holds": holds, "shifts": shifts}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
