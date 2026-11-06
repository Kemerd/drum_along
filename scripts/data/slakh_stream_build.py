"""
slakh_stream_build.py - build drumless training backings from Slakh2100 (CC BY 4.0).

Usage:
    python -I scripts/data/slakh_stream_build.py <out_dir> [--url URL_or_local_file] [--workers 8]

Recommended: download the archive once with scripts/data/get_slakh.cmd, which
resumes where it stopped, then pass the local file as --url. Direct streaming
cannot resume after the process dies. It also saves little disk, because the
archive is not stored track by track: one song's files are scattered through
all 104 GB, so nearly every stem has to be staged before the first song is
complete. Peak staging is about 65 GB; the build takes 20 minutes from a local file.

What it does, in one pass over the official Zenodo archive:

  - keeps only metadata.yaml, all_src.mid and the stems (skips mix.flac, the
    per-stem MIDI and the 'omitted' split),
  - as soon as a track's metadata arrives, deletes its drum stems and skips any
    drum stems still to come,
  - as soon as a track is complete, a worker mixes its non-drum stems into
        backing_full.flac   every non-drum instrument
        backing_sub.flac    a random subset (variety for training)
    copies all_src.mid + metadata.yaml next to them, and deletes the stems.

Archive contents are untrusted: every member name must match a strict pattern,
and output paths are built from the parsed parts, never from the raw name.

Attribution: Slakh2100, Manilow et al., "Cutting Music Source Separation Some
Slakh" (WASPAA 2019), CC BY 4.0, https://zenodo.org/records/4599666
"""

import io
import json
import os
import random
import re
import shutil
import sys
import time
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_URL = "https://zenodo.org/api/records/4599666/files/slakh2100_flac_redux.tar.gz/content"
USER_AGENT = "drum_along-dataset-builder/1.0"

# Only these member names are accepted. Anything else is skipped.
MEMBER_RE = re.compile(
    r"^slakh2100_flac_redux/(train|validation|test)/(Track\d{5})/"
    r"(metadata\.yaml|all_src\.mid|stems/(S\d{2})\.flac)$"
)

MIN_FREE_BYTES = 12 * 1024**3   # pause reading below this much free disk


# ===========================================================================
# Resumable HTTP stream
# ===========================================================================

class ResumableHTTP(io.RawIOBase):
    """
    Raw stream over one HTTP download that reconnects with a Range request
    after network errors, so a dropped connection does not lose 50 GB of
    progress. One connection at a time: the host's speed limit is respected.
    """

    def __init__(self, url: str, log):
        self.url, self.log = url, log
        self.pos = 0
        self.total = None
        self.resp = None
        self._open()

    def _open(self):
        headers = {"User-Agent": USER_AGENT}
        if self.pos:
            headers["Range"] = f"bytes={self.pos}-"
        req = urllib.request.Request(self.url, headers=headers)
        self.resp = urllib.request.urlopen(req, timeout=120)
        if self.total is None:
            length = self.resp.headers.get("Content-Length")
            self.total = int(length) if length else None

    def readable(self):
        return True

    def readinto(self, buf):
        for attempt in range(40):
            try:
                n = self.resp.readinto(buf)
                if n == 0 and self.total is not None and self.pos < self.total:
                    raise IOError("connection closed early")
                self.pos += n
                return n
            except Exception as exc:  # network hiccup: back off, reconnect at pos
                wait = min(120, 2 ** min(attempt, 7))
                self.log(f"stream error at {self.pos:,} bytes ({exc}); retry in {wait}s")
                time.sleep(wait)
                try:
                    self._open()
                except Exception as exc2:
                    self.log(f"reconnect failed: {exc2}")
        raise IOError("too many stream errors, giving up")


# ===========================================================================
# Per-track mixing (runs in worker processes)
# ===========================================================================

def mix_track(stage: str, out: str, stems: list[str], seed: int) -> dict:
    """Sum non-drum stems into two backings, copy labels, delete the stage dir."""
    import numpy as np
    import soundfile as sf

    stage_p, out_p = Path(stage), Path(out)
    rng = random.Random(seed)
    audio, sr = {}, None
    for s in stems:
        x, r = sf.read(str(stage_p / "stems" / f"{s}.flac"), dtype="float32", always_2d=True)
        audio[s] = x.mean(axis=1)
        sr = sr or r
        if r != sr:
            raise ValueError(f"sample-rate mismatch in {stage}")

    n = max(len(a) for a in audio.values())

    def mix(names):
        y = np.zeros(n, dtype=np.float32)
        for s in names:
            y[: len(audio[s])] += audio[s]
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        return y * (0.89 / peak) if peak > 0 else y   # -1 dBFS

    out_p.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_p / "backing_full.flac"), mix(stems), sr, subtype="PCM_16")

    # Random subset: keep each stem with p = 0.65, at least two of them.
    subset = [s for s in stems if rng.random() < 0.65]
    if len(subset) < 2:
        subset = rng.sample(stems, min(2, len(stems)))
    sf.write(str(out_p / "backing_sub.flac"), mix(subset), sr, subtype="PCM_16")

    shutil.copyfile(stage_p / "all_src.mid", out_p / "all_src.mid")
    shutil.copyfile(stage_p / "metadata.yaml", out_p / "metadata.yaml")
    shutil.rmtree(stage_p, ignore_errors=True)
    return {"split": out_p.parent.name, "track": out_p.name, "seconds": n / sr,
            "stems": stems, "subset": subset}


# ===========================================================================
# Main streaming loop
# ===========================================================================

def main() -> int:
    import yaml

    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    out_root = Path(args.pop(0)).resolve()
    url, workers = DEFAULT_URL, 8
    while args:
        a = args.pop(0)
        if a == "--url" and args:
            url = args.pop(0)
        elif a == "--workers" and args:
            workers = max(1, int(args.pop(0)))

    stage_root = out_root / "_stage"
    stage_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "build.log"
    progress_path = out_root / "progress.json"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Track bookkeeping: key = (split, track)
    meta: dict[tuple, dict] = {}         # parsed stem info once metadata arrived
    have: dict[tuple, set] = {}          # stems present on disk
    has_midi: set = set()
    done: set = set()
    results = []

    def stems_needed(info):
        return sorted(s for s, v in (info.get("stems") or {}).items()
                      if v.get("audio_rendered", False) and not v.get("is_drum", False))

    pool = ProcessPoolExecutor(max_workers=workers)
    futures = []

    def maybe_submit(key):
        if key in done or key not in meta or key not in has_midi:
            return
        need = meta[key]["needed"]
        if not need or not set(need) <= have.get(key, set()):
            return
        split, track = key
        stage = stage_root / split / track
        out = out_root / split / track
        seed = int(track[5:]) * 7919 + 17   # stable across runs, unlike hash()
        futures.append(pool.submit(mix_track, str(stage), str(out), need, seed))
        done.add(key)

    # A local archive (downloaded once with a resumable `curl -C -`) is read
    # straight from disk; anything else is streamed over HTTP.
    if Path(url).is_file():
        raw = open(url, "rb")

        class _Local:
            pos = 0
            total = Path(url).stat().st_size

        stream = _Local()

        class _Counting(io.RawIOBase):
            def readable(self):
                return True

            def readinto(self, b):
                n = raw.readinto(b)
                stream.pos += n
                return n

        buffered = io.BufferedReader(_Counting(), buffer_size=8 * 1024 * 1024)
        log(f"local archive {url} ({stream.total:,} bytes)")
    else:
        stream = ResumableHTTP(url, log)
        log(f"archive size {stream.total:,} bytes" if stream.total else "archive size unknown")
        buffered = io.BufferedReader(stream, buffer_size=8 * 1024 * 1024)

    members = 0
    last_report = 0.0
    t0 = time.time()
    with tarfile.open(fileobj=buffered, mode="r|gz") as tar:
        for m in tar:
            members += 1
            match = MEMBER_RE.match(m.name)
            if not match or not m.isfile() or m.size <= 0 or m.size > 2 * 1024**3:
                continue
            split, track, kind, stem = match.groups()
            key = (split, track)
            stage = stage_root / split / track

            # Skip drum / unrendered stems once we know what they are.
            if stem and key in meta and stem not in meta[key]["needed"]:
                continue

            # Never write without disk headroom: pause until workers free some.
            while shutil.disk_usage(out_root).free < MIN_FREE_BYTES:
                log("low disk space, waiting for mixing workers")
                time.sleep(30)
                futures[:] = [f for f in futures if not f.done()]

            src = tar.extractfile(m)
            if src is None:
                continue
            if stem:
                dst = stage / "stems" / f"{stem}.flac"
            else:
                dst = stage / kind
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                shutil.copyfileobj(src, f, length=4 * 1024 * 1024)

            if stem:
                have.setdefault(key, set()).add(stem)
            elif kind == "all_src.mid":
                has_midi.add(key)
            elif kind == "metadata.yaml":
                try:
                    info = yaml.safe_load(dst.read_text(encoding="utf-8")) or {}
                except Exception as exc:
                    log(f"bad metadata in {split}/{track}: {exc}")
                    info = {}
                meta[key] = {"needed": stems_needed(info)}
                # Drop stems that arrived before we knew they were drums.
                for s in list(have.get(key, set())):
                    if s not in meta[key]["needed"]:
                        (stage / "stems" / f"{s}.flac").unlink(missing_ok=True)
                        have[key].discard(s)

            maybe_submit(key)

            # Collect finished mixes and report progress every 30 s.
            now = time.time()
            if now - last_report > 30:
                last_report = now
                still = []
                for f in futures:
                    if f.done():
                        try:
                            results.append(f.result())
                        except Exception as exc:
                            log(f"mix failed: {exc}")
                    else:
                        still.append(f)
                futures[:] = still
                mb = stream.pos / 1e6
                rate = mb / max(now - t0, 1e-6)
                prog = {
                    "bytes": stream.pos, "total": stream.total,
                    "percent": round(100.0 * stream.pos / stream.total, 2) if stream.total else None,
                    "MBps": round(rate, 2), "members": members,
                    "tracks_mixed": len(results), "tracks_submitted": len(done),
                    "tracks_seen": len(set(have) | set(meta)),
                    "free_GB": round(shutil.disk_usage(out_root).free / 1e9, 1),
                    "eta_hours": round((stream.total - stream.pos) / 1e6 / rate / 3600, 2) if stream.total and rate > 0 else None,
                }
                progress_path.write_text(json.dumps(prog, indent=2), encoding="utf-8")

    # Drain the pool.
    for f in futures:
        try:
            results.append(f.result())
        except Exception as exc:
            log(f"mix failed: {exc}")
    pool.shutdown()

    incomplete = sorted(f"{s}/{t}" for (s, t) in (set(have) | set(meta)) if (s, t) not in done)
    manifest = {"source": "Slakh2100 redux (CC BY 4.0), https://zenodo.org/records/4599666",
                "tracks": sorted(results, key=lambda r: (r["split"], r["track"])), "incomplete": incomplete}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"finished: {len(results)} tracks mixed, {len(incomplete)} incomplete")
    shutil.rmtree(stage_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
