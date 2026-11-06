"""
make_features.py - turn prepared backings + MIDI into training examples.

Usage (training env, which has pretty_midi):
    .venv-sep/Scripts/python.exe scripts/train/make_features.py <backings_root> <features_root> [--workers 24]

<backings_root> holds <split>/<TrackXXXXX>/{backing_full.flac, backing_sub.flac, all_src.mid}
(as written by scripts/data/slakh_stream_build.py). For every backing this:

  1. decodes it to a temporary WAV,
  2. runs the C++ harness with --features, so training sees *exactly* the
     numbers the plugin will compute,
  3. reads beats, downbeats and drum activity from the song's own MIDI,
  4. writes <features_root>/<split>/<Track>_<variant>.x.npy  (float16 [frames, 135], memory-mappable)
     and    <features_root>/<split>/<Track>_<variant>.npz:
        beat   uint8   [frames]        1 at the frame nearest each beat
        down   uint8   [frames]        1 at the frame nearest each downbeat
        drums  uint8   [frames]        1 where a drummer is playing (+-1 s)
        down_ok bool                   time signature is 4/4 throughout
        bpm    float32                 median tempo from the MIDI

Re-runnable: finished examples are skipped, so it can follow a dataset that
is still streaming in.
"""

import os
import subprocess
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WAV2MID = ROOT / "build" / "Release" / "wav2mid.exe"
VARIANTS = ("full", "sub", "aug")
DRUM_WINDOW_S = 1.0


# ===========================================================================
# Augmentation: make clean synthetic renders sound more like real records
# ===========================================================================

def augment(y: np.ndarray, sr: int, seed: int) -> np.ndarray:
    """
    Random, reproducible "recording chain" on a mono float signal:
      level        peak between -18 and -1 dBFS
      EQ tilt      low shelf (200 Hz) and high shelf (4 kHz), each +-6 dB
      saturation   tanh drive 1.5-4 (half the time)
      room         decaying-noise reverb, RT60 0.3-1.2 s, 10-30% wet (half the time)
      noise        pink-ish noise at 25-45 dB SNR (half the time)
    """
    from scipy.signal import fftconvolve, lfilter

    rng = np.random.default_rng(seed)
    y = y.astype(np.float64)

    def shelf(x, f0, gain_db, high):
        # RBJ cookbook shelving biquad, S = 1.
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * f0 / sr
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / 2 * np.sqrt(2)
        s = 1 if high else -1
        b0 = A * ((A + 1) + s * (A - 1) * cw + 2 * np.sqrt(A) * alpha)
        b1 = -2 * s * A * ((A - 1) + s * (A + 1) * cw)
        b2 = A * ((A + 1) + s * (A - 1) * cw - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - s * (A - 1) * cw + 2 * np.sqrt(A) * alpha
        a1 = 2 * s * ((A - 1) - s * (A + 1) * cw)
        a2 = (A + 1) - s * (A - 1) * cw - 2 * np.sqrt(A) * alpha
        return lfilter([b0 / a0, b1 / a0, b2 / a0], [1, a1 / a0, a2 / a0], x)

    y = shelf(y, 200.0, rng.uniform(-6, 6), high=False)
    y = shelf(y, 4000.0, rng.uniform(-6, 6), high=True)

    if rng.random() < 0.5:
        drive = rng.uniform(1.5, 4.0)
        y = np.tanh(drive * y / (np.max(np.abs(y)) + 1e-9)) / np.tanh(drive)

    if rng.random() < 0.5:
        rt60 = rng.uniform(0.3, 1.2)
        n = int(rt60 * sr)
        ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
        wet = fftconvolve(y, ir)[: len(y)]
        mix = rng.uniform(0.1, 0.3)
        y = (1 - mix) * y + mix * wet * (np.std(y) / (np.std(wet) + 1e-9))

    if rng.random() < 0.5:
        white = rng.standard_normal(len(y))
        pink = lfilter([0.049922, -0.095993, 0.050613, -0.004408], [1, -2.494956, 2.017265, -0.522189], white)
        snr_db = rng.uniform(25, 45)
        pink *= np.std(y) / (np.std(pink) + 1e-9) * 10 ** (-snr_db / 20)
        y = y + pink

    peak = np.max(np.abs(y)) + 1e-9
    level = 10 ** (rng.uniform(-18, -1) / 20)
    return (y * level / peak).astype(np.float32)


# ===========================================================================
# Feature file
# ===========================================================================

def read_daf(path: Path):
    """Read a wav2mid --features file -> (features [T, F], hop, window, sr)."""
    raw = path.read_bytes()
    if raw[:4] != b"DAF1":
        raise ValueError(f"{path}: not a DAF1 file")
    F, hop, win = np.frombuffer(raw[4:16], np.int32)
    sr = float(np.frombuffer(raw[16:24], np.float64)[0])
    x = np.frombuffer(raw[24:], np.float32)
    if x.size % F:
        raise ValueError(f"{path}: truncated feature file")
    return x.reshape(-1, int(F)), int(hop), int(win), sr


def frame_index(times: np.ndarray, hop: int, win: int, sr: float) -> np.ndarray:
    """Frame whose centre ((t+1)*hop - win/2)/sr is nearest each time."""
    return np.rint((times * sr + win / 2.0) / hop - 1.0).astype(np.int64)


# ===========================================================================
# Labels from MIDI
# ===========================================================================

def midi_labels(mid_path: Path, T: int, hop: int, win: int, sr: float) -> dict:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(mid_path))
    beats = pm.get_beats()
    downs = pm.get_downbeats()

    ts = pm.time_signature_changes
    down_ok = all(t.numerator == 4 and t.denominator == 4 for t in ts) if ts else True

    beat = np.zeros(T, np.uint8)
    down = np.zeros(T, np.uint8)
    for arr, times in ((beat, beats), (down, downs)):
        idx = frame_index(np.asarray(times), hop, win, sr)
        idx = idx[(idx >= 0) & (idx < T)]
        arr[idx] = 1

    # Drum activity: frames within +-1 s of any drum onset.
    drum_on = np.array(sorted(n.start for inst in pm.instruments if inst.is_drum for n in inst.notes))
    drums = np.zeros(T, np.uint8)
    if drum_on.size:
        centres = ((np.arange(T) + 1) * hop - win / 2.0) / sr
        j = np.clip(np.searchsorted(drum_on, centres), 1, drum_on.size - 1)
        near = np.minimum(np.abs(centres - drum_on[j - 1]), np.abs(centres - drum_on[j]))
        if drum_on.size == 1:
            near = np.abs(centres - drum_on[0])
        drums[near <= DRUM_WINDOW_S] = 1

    bpm = 60.0 / float(np.median(np.diff(beats))) if len(beats) > 2 else 0.0
    return {"beat": beat, "down": down, "drums": drums, "down_ok": down_ok, "bpm": np.float32(bpm)}


# ===========================================================================
# One example
# ===========================================================================

def build_one(track_dir: str, variant: str, out_path: str) -> str:
    import soundfile as sf

    track_dir, out_path = Path(track_dir), Path(out_path)
    source = "full" if variant == "aug" else variant
    flac = track_dir / f"backing_{source}.flac"
    mid = track_dir / "all_src.mid"
    if not flac.exists() or not mid.exists():
        return f"skip {track_dir.name}/{variant}: missing files"

    with tempfile.TemporaryDirectory(prefix="daf_") as tmp:
        tmp = Path(tmp)
        wav = tmp / "in.wav"
        if variant == "aug":
            y, sr = sf.read(str(flac), dtype="float32", always_2d=True)
            seed = int(track_dir.name[5:]) * 104729 + 3
            sf.write(str(wav), augment(y.mean(axis=1), sr, seed), sr, subtype="PCM_16")
        else:
            audio, sr = sf.read(str(flac), dtype="int16", always_2d=True)
            sf.write(str(wav), audio, sr, subtype="PCM_16")
        daf = tmp / "x.daf"
        res = subprocess.run([str(WAV2MID), str(wav), str(tmp / "run"), "--features", str(daf)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            return f"fail {track_dir.name}/{variant}: {res.stderr.strip()[:200]}"
        x, hop, win, fsr = read_daf(daf)

    labels = midi_labels(mid, x.shape[0], hop, win, fsr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Features as a plain .npy so training can memory-map them (the full set is
    # ~30 GB); labels in a small compressed .npz written last as the "done" mark.
    x_path = out_path.with_suffix(".x.npy")
    tmp_x = out_path.with_suffix(".x.tmp.npy")
    np.save(tmp_x, x.astype(np.float16))
    os.replace(tmp_x, x_path)
    tmp_out = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_out, hop=hop, win=win, sr=fsr, frames=x.shape[0], **labels)
    os.replace(tmp_out, out_path)
    return f"ok {track_dir.parent.name}/{track_dir.name}/{variant} {x.shape[0]} frames {labels['bpm']:.0f} bpm"


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 2
    src_root, dst_root = Path(args[0]), Path(args[1])
    workers = 24
    if "--workers" in args:
        workers = max(1, int(args[args.index("--workers") + 1]))

    jobs = []
    for split_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for track_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            # Only tracks the builder has fully finished (MIDI copied last).
            if not (track_dir / "metadata.yaml").exists():
                continue
            for v in VARIANTS:
                out = dst_root / split_dir.name / f"{track_dir.name}_{v}.npz"
                if not out.exists():
                    jobs.append((str(track_dir), v, str(out)))

    print(f"{len(jobs)} examples to build with {workers} workers")
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_one, *j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                msg = f.result()
            except Exception as exc:
                msg = f"fail: {exc}"
            ok += msg.startswith("ok")
            fail += msg.startswith("fail")
            if msg.startswith("fail") or i % 50 == 0 or i == len(futures):
                print(f"[{i}/{len(futures)}] {msg}", flush=True)
    print(f"done: {ok} built, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
