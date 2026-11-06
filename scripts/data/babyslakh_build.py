"""
babyslakh_build.py - smoke-test data: BabySlakh (20 Slakh songs, 16 kHz, CC BY 4.0)
turned into the same layout slakh_stream_build.py produces.

Usage:
    python -I scripts/data/babyslakh_build.py <babyslakh_16k.tar.gz> <out_dir>

Tracks 1-18 go to train/, 19-20 to validation/. Backings are upsampled to
44.1 kHz so features match the real pipeline (the audio itself stops at 8 kHz,
so this is only for testing the pipeline, never for the shipped model).
"""

import io
import random
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

MEMBER_RE = re.compile(r"^babyslakh_16k/(Track\d{5})/(metadata\.yaml|all_src\.mid|stems/(S\d{2})\.wav)$")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    archive, out_root = Path(sys.argv[1]), Path(sys.argv[2])

    # ---- read only the members we need, by validated name ----------------------
    tracks: dict[str, dict] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            mm = MEMBER_RE.match(m.name)
            if not mm or not m.isfile() or m.size > 512 * 1024**2:
                continue
            track, kind, stem = mm.groups()
            data = tar.extractfile(m).read()
            t = tracks.setdefault(track, {"stems": {}})
            if stem:
                t["stems"][stem] = data
            else:
                t[kind] = data

    for i, (track, t) in enumerate(sorted(tracks.items())):
        if "metadata.yaml" not in t or "all_src.mid" not in t:
            continue
        info = yaml.safe_load(t["metadata.yaml"].decode("utf-8")) or {}
        # BabySlakh's metadata marks every stem audio_rendered: False even
        # when the audio exists, so trust the files that are actually there.
        need = sorted(s for s, v in (info.get("stems") or {}).items()
                      if not v.get("is_drum", False) and s in t["stems"])
        if len(need) < 1:
            continue
        audio, sr = {}, None
        for s in need:
            x, r = sf.read(io.BytesIO(t["stems"][s]), dtype="float32", always_2d=True)
            audio[s], sr = x.mean(axis=1), r
        n = max(len(a) for a in audio.values())

        def mix(names):
            y = np.zeros(n, np.float32)
            for s in names:
                y[: len(audio[s])] += audio[s]
            y = resample_poly(y, 441, sr // 100).astype(np.float32) if sr != 44100 else y
            peak = float(np.max(np.abs(y)))
            return y * (0.89 / peak) if peak > 0 else y

        rng = random.Random(int(track[5:]))
        subset = [s for s in need if rng.random() < 0.65] or need[:2]
        split = "train" if i < 18 else "validation"
        out = out_root / split / track
        out.mkdir(parents=True, exist_ok=True)
        sf.write(str(out / "backing_full.flac"), mix(need), 44100, subtype="PCM_16")
        sf.write(str(out / "backing_sub.flac"), mix(subset), 44100, subtype="PCM_16")
        (out / "all_src.mid").write_bytes(t["all_src.mid"])
        (out / "metadata.yaml").write_bytes(t["metadata.yaml"])
        print(f"{split}/{track}: {len(need)} stems, {n / sr:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
