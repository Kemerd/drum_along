"""
make_ab.py - one listening file: the original drummer, then ours, same excerpt.

Usage:
    python scripts/make_ab.py <original.wav> <ours_preview.wav> <out.wav> [start_s] [length_s]

Layout of the output:
    [original, start..start+length]  [0.8 s gap]  [ours, same span]

Both halves are loudness-matched (RMS) so neither wins just by being louder.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_stereo(path: Path, sr_out: int) -> np.ndarray:
    """Read any WAV as float32 stereo at sr_out."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    if sr != sr_out:
        g = np.gcd(int(sr), sr_out)
        x = resample_poly(x, sr_out // g, int(sr) // g, axis=0).astype(np.float32)
    return x


def main() -> int:
    if len(sys.argv) not in (4, 5, 6):
        print("usage: make_ab.py <original.wav> <ours.wav> <out.wav> [start_s] [length_s]")
        return 2
    orig_path, ours_path, out_path = map(Path, sys.argv[1:4])
    start = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    length = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0

    sr = 44100
    a = load_stereo(orig_path, sr)
    b = load_stereo(ours_path, sr)

    # ------------------------------------------------------------------
    # Cut the same span from both, clamped to what exists.
    # ------------------------------------------------------------------
    s0 = max(0, int(start * sr))
    s1 = min(len(a), len(b), s0 + int(length * sr))
    if s1 <= s0:
        print("error: excerpt is outside the audio")
        return 1
    a, b = a[s0:s1].copy(), b[s0:s1].copy()

    # Loudness match, then a 20 ms fade at every edge so nothing clicks.
    rms_a = float(np.sqrt(np.mean(a ** 2)) + 1e-9)
    rms_b = float(np.sqrt(np.mean(b ** 2)) + 1e-9)
    b *= rms_a / rms_b
    fade = int(0.02 * sr)
    ramp = np.linspace(0.0, 1.0, fade)[:, None]
    for x in (a, b):
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]

    out = np.concatenate([a, np.zeros((int(0.8 * sr), 2), np.float32), b])
    peak = float(np.max(np.abs(out)))
    if peak > 0.98:
        out *= 0.98 / peak

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, sr, subtype="PCM_16")
    print(f"wrote    {out_path.resolve()}  (original {start:.0f}-{start + length:.0f} s, then ours)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
