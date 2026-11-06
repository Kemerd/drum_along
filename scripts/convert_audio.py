"""
convert_audio.py - decode any audio file libsndfile can read (MP3, OGG,
FLAC, AIFF, WAV) into 16-bit PCM WAV for the C++ harness.

Usage:
    python scripts/convert_audio.py <in.mp3> <out.wav>
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import soundfile as sf


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: convert_audio.py <in> <out.wav>")
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    except Exception as exc:
        print(f"error: cannot decode {src}: {exc}")
        return 1
    if x.size == 0:
        print("error: decoded audio is empty")
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), x, sr, subtype="PCM_16")
    print(f"wrote    {dst}  ({sr} Hz, {x.shape[1]} ch, {len(x) / sr:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
