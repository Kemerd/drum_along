"""
evaluate.py - grade the drummer's beats against a reference.

Usage (run with the project venv, which has a working librosa + mir_eval):
    .venv/Scripts/python.exe scripts/evaluate.py <song.wav> <output_prefix> [--drums drum_stem.wav]

Reference:
  - With --drums: the REAL drummer. The isolated drum stem is beat-tracked,
    then every beat is snapped to the attack of the nearest real drum hit
    (~1.5 ms resolution). This is ground truth for "aligned to the beat".
  - Without: librosa's offline beat tracker on the full song. It sees the
    whole song at once (future included), but it is not ground truth.

Reports tempo, beat F-measure (+-70 ms), signed offset, and where our beats
land inside the reference beat (on it, on the "&", elsewhere), and writes
<prefix>_eval.png. Importable: evaluate() returns the numbers as a dict.
"""

import csv
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np


# ===========================================================================
# Loading
# ===========================================================================

def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a numeric CSV into a dict of column arrays."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


# ===========================================================================
# Ground truth from a real drum stem
# ===========================================================================

def drum_attacks(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Attack times of every drum hit in an isolated drum track, ~1.5 ms resolution.

    Amplitude envelope (2.9 ms window, 1.45 ms hop) -> rise per hop -> peaks
    of the rise above an adaptive threshold. The steepest rise marks the
    attack, which is what the ear hears as "the hit".
    """
    from scipy.ndimage import maximum_filter1d, median_filter

    win = max(16, int(round(0.0029 * sr)))
    hop = max(8, int(round(0.00145 * sr)))
    n = 1 + max(0, (len(y) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    env = np.sqrt(np.mean(y[idx] ** 2, axis=1))
    env = np.convolve(env, np.ones(3) / 3, mode="same")
    rise = np.maximum(np.diff(env, prepend=env[0]), 0.0)
    thr = 4.0 * median_filter(rise, size=max(3, int(0.5 * sr / hop))) + 1e-5
    peaks = (rise > thr) & (rise == maximum_filter1d(rise, size=max(3, int(0.03 * sr / hop))))
    return (np.nonzero(peaks)[0] * hop + win / 2) / sr


def drum_stem_beats(path: Path) -> tuple[float, np.ndarray]:
    """Beats of the real drummer: beat-track the stem, snap to real attacks."""
    import librosa

    y, sr = librosa.load(str(path), sr=None, mono=True)
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beats = librosa.frames_to_time(frames, sr=sr)
    attacks = drum_attacks(y, sr)
    if attacks.size >= 2 and beats.size:
        j = np.clip(np.searchsorted(attacks, beats), 1, attacks.size - 1)
        cand = np.where(np.abs(attacks[j - 1] - beats) < np.abs(attacks[j] - beats), attacks[j - 1], attacks[j])
        beats = np.where(np.abs(cand - beats) <= 0.04, cand, beats)
    return float(np.atleast_1d(tempo)[0]), np.sort(beats)


def signed_offsets(est: np.ndarray, ref: np.ndarray, window: float = 0.07) -> np.ndarray:
    """est-minus-ref offsets for estimated beats within +-window of a reference beat."""
    if est.size == 0 or ref.size < 2:
        return np.array([])
    idx = np.clip(np.searchsorted(ref, est), 1, ref.size - 1)
    left, right = ref[idx - 1], ref[idx]
    nearest = np.where(np.abs(est - left) < np.abs(est - right), left, right)
    d = est - nearest
    return d[np.abs(d) <= window]


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate(song_path: Path, prefix: str, drum_stem: Path | None = None,
             plot: bool = True, quiet: bool = False) -> dict:
    """Grade one run. Returns tempo, F-measure, placement and offset numbers."""
    import librosa
    import mir_eval

    # ---- reference ------------------------------------------------------------
    if drum_stem is not None:
        ref_tempo, ref_beats = drum_stem_beats(drum_stem)
        ref_kind = "drums"
    else:
        y, sr = librosa.load(str(song_path), sr=None, mono=True)
        ref_tempo, ref_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        ref_tempo = float(np.atleast_1d(ref_tempo)[0])
        ref_beats = librosa.frames_to_time(ref_frames, sr=sr)
        ref_kind = "mix"

    # Real tempo from the median beat interval (not the tracker's coarse grid).
    if ref_beats.size > 2:
        ref_tempo = 60.0 / float(np.median(np.diff(ref_beats)))

    # ---- ours: the quarter-note steps the drummer actually played --------------
    steps = load_csv(Path(prefix + "_steps.csv"))
    frames = load_csv(Path(prefix + "_frames.csv"))
    if not steps:
        raise RuntimeError(f"no steps logged for {prefix}")
    played = (steps["playing"] == 1) & (steps["step_in_bar"] % 4 == 0)
    est = np.sort(steps["time_s"][played])
    start = float(est[0]) if est.size else 0.0
    ref_eval = ref_beats[ref_beats >= start - 0.1]

    f1 = mir_eval.beat.f_measure(ref_eval, est, f_measure_threshold=0.07) if ref_eval.size and est.size else 0.0
    off = signed_offsets(est, ref_eval)
    tempo = float(np.median(steps["tempo_bpm"][steps["playing"] == 1])) if played.any() else 0.0

    on_beat = off_beat = 0.0
    if est.size and ref_beats.size > 2:
        k = np.clip(np.searchsorted(ref_beats, est) - 1, 0, ref_beats.size - 2)
        ph = (est - ref_beats[k]) / (ref_beats[k + 1] - ref_beats[k])
        ph = np.where(ph > 0.75, ph - 1.0, ph)
        on_beat = float(np.mean(np.abs(ph) < 0.12))
        off_beat = float(np.mean(np.abs(ph - 0.5) < 0.12))

    result = {
        "ref_kind": ref_kind, "ref_tempo": ref_tempo, "tempo": tempo, "f1": float(f1),
        "on_beat": on_beat, "off_beat": off_beat,
        "offset_ms": 1000.0 * float(np.median(off)) if off.size else 0.0,
        "spread_ms": 1000.0 * float(off.std()) if off.size else 0.0,
        "beats_played": int(est.size), "start_s": start,
    }

    if not quiet:
        print(f"reference  {ref_kind}: {ref_tempo:.2f} BPM, {ref_beats.size} beats")
        print(f"drummer    {tempo:.2f} BPM, {est.size} beats played, from {start:.2f} s")
        print(f"F-measure  {f1:.3f} (+-70 ms)")
        print(f"offset     median {result['offset_ms']:+.1f} ms, spread {result['spread_ms']:.1f} ms over {off.size} matched beats")
        print(f"placement  on the beat {on_beat:.0%}, on the off-beat {off_beat:.0%}, elsewhere {1 - on_beat - off_beat:.0%}")

    # ---- plot -------------------------------------------------------------------
    if plot and frames:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return result

        t = frames["time_s"]
        fig, ax = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
        top = max(float(frames["onset"].max()), 1.0)
        ax[0].plot(t, frames["onset"], lw=0.6, color="0.35", label="onset")
        if "harmonic" in frames:
            ax[0].plot(t, -frames["harmonic"], lw=0.6, color="tab:green", label="chord change (down)")
        ax[0].vlines(ref_beats, 0, top * 0.5, color="tab:blue", lw=0.8, label=f"reference beats ({ref_kind})")
        ax[0].vlines(est, top * 0.5, top, color="tab:red", lw=0.8, label="drummer beats")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[1].plot(t, frames["tempo_bpm"], color="tab:red", lw=1, label="drummer tempo")
        ax[1].axhline(ref_tempo, color="tab:blue", ls="--", lw=1, label="reference tempo")
        ax[1].fill_between(t, 0, 250, where=frames["locked"] > 0, color="tab:green", alpha=0.08, label="locked")
        ax[1].set_ylim(40, 220)
        ax[1].legend(loc="upper right", fontsize=8)
        ax[2].plot(t, frames["level"], color="tab:purple", lw=1, label="intensity level")
        ax[2].plot(t, frames["periodicity"], color="tab:orange", lw=0.6, label="periodicity")
        ax[2].set_ylim(-0.1, 1.05)
        ax[2].legend(loc="upper right", fontsize=8)
        ax[2].set_xlabel("seconds")
        fig.tight_layout()
        png = Path(prefix + "_eval.png")
        fig.savefig(png, dpi=110)
        plt.close(fig)
        if not quiet:
            print(f"plot       {png.resolve()}")
    return result


def main() -> int:
    args = sys.argv[1:]
    drum_stem = None
    if "--drums" in args:
        i = args.index("--drums")
        if i + 1 >= len(args):
            print("error: --drums needs a path")
            return 2
        drum_stem = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    if len(args) != 2:
        print("usage: evaluate.py <song.wav> <output_prefix> [--drums drum_stem.wav]")
        return 2
    evaluate(Path(args[0]), args[1], drum_stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
