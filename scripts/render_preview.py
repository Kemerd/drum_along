"""
render_preview.py - mix the drummer's MIDI back over the song so you can
listen to it without a DAW.

Usage:
    python scripts/render_preview.py <song.wav> <events.csv> <out.wav> [--kit scripts/kits/ableton_rock.json]

Without --kit, the kit is synthesised from scratch (sine sweeps and filtered
noise): fine for judging timing, not tone. With --kit, a JSON file maps each
GM note to velocity-layered samples on disk (round-robin within a layer), so
the preview sounds like a real kit. In the plugin, your own drum instrument
plays these notes instead.

Also writes <out>_drums.wav with the drums alone.
"""

import csv
import json
import sys

# Keep Windows consoles happy with any characters we print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

RENDER_SR = 44100
RNG = np.random.default_rng(1234)  # deterministic noise, identical renders


# ===========================================================================
# Synthesised fallback kit
# ===========================================================================

def _env(n: int, sr: int, tau: float) -> np.ndarray:
    """Exponential decay envelope with time constant tau seconds."""
    t = np.arange(n) / sr
    return np.exp(-t / tau)


def _sweep(sr: int, f0: float, f1: float, sweep_tau: float, amp_tau: float, dur: float) -> np.ndarray:
    """Sine whose pitch glides exponentially from f0 to f1 (kick / tom body)."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    freq = f1 + (f0 - f1) * np.exp(-t / sweep_tau)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * _env(n, sr, amp_tau)


def _noise(sr: int, dur: float, lo: float | None, hi: float | None, tau: float) -> np.ndarray:
    """Band-limited noise burst with an exponential tail."""
    n = int(dur * sr)
    x = RNG.standard_normal(n)
    if lo and hi:
        sos = butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos")
    elif lo:
        sos = butter(2, lo, btype="highpass", fs=sr, output="sos")
    elif hi:
        sos = butter(2, hi, btype="lowpass", fs=sr, output="sos")
    else:
        sos = None
    if sos is not None:
        x = sosfilt(sos, x)
    x /= max(np.max(np.abs(x)), 1e-9)
    return x * _env(n, sr, tau)


def _metal(sr: int, freqs: list[float], dur: float, tau: float) -> np.ndarray:
    """Inharmonic partials: the 'ping' of a ride or crash."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    x = sum(np.sin(2 * np.pi * f * t + RNG.uniform(0, 2 * np.pi)) for f in freqs)
    return x / len(freqs) * _env(n, sr, tau)


def _layer(*parts: np.ndarray) -> np.ndarray:
    """Sum layers of different lengths, zero-padding the shorter ones."""
    n = max(len(p) for p in parts)
    out = np.zeros(n)
    for p in parts:
        out[:len(p)] += p
    return out


def build_synth_kit(sr: int) -> dict[int, np.ndarray]:
    """One-shot mono sample per GM note, peak-normalised, then given a mix gain."""
    kit = {
        36: _layer(_sweep(sr, 160, 48, 0.035, 0.22, 0.6), 0.15 * _noise(sr, 0.01, 2000, None, 0.003)),
        38: _layer(0.55 * _sweep(sr, 240, 185, 0.02, 0.07, 0.3), 0.8 * _noise(sr, 0.35, 1500, 9000, 0.11)),
        42: _noise(sr, 0.08, 7000, None, 0.025),
        46: _noise(sr, 0.6, 6000, None, 0.22),
        49: _layer(0.85 * _noise(sr, 2.5, 3500, None, 0.9), 0.25 * _metal(sr, [431, 677, 1201, 2397, 3711], 2.5, 0.7)),
        51: _layer(0.6 * _metal(sr, [523, 797, 1033, 2511, 3402, 5123], 1.2, 0.5), 0.35 * _noise(sr, 1.2, 5000, None, 0.35)),
        50: _sweep(sr, 260, 200, 0.05, 0.25, 0.7),
        47: _sweep(sr, 190, 145, 0.05, 0.28, 0.8),
        45: _sweep(sr, 140, 100, 0.06, 0.32, 0.9),
        44: _noise(sr, 0.05, 5000, None, 0.012),
    }
    gains = {36: 1.0, 38: 0.8, 42: 0.32, 46: 0.35, 49: 0.45, 51: 0.3, 50: 0.7, 47: 0.7, 45: 0.75, 44: 0.25}

    out = {}
    for note, x in kit.items():
        x = x / max(np.max(np.abs(x)), 1e-9)
        fade = min(16, len(x))
        x[:fade] *= np.linspace(0.0, 1.0, fade)   # no click at the start
        out[note] = (x * gains[note]).astype(np.float32)
    return out


# ===========================================================================
# Sampled kit
# ===========================================================================

class SampledKit:
    """
    Velocity-layered, round-robin sample kit loaded from a JSON map.

    JSON layout:
        { "root": "<folder>",
          "notes": { "<gm note>": { "gain": g, "layers": [[maxVel, [files...]], ...] } } }

    A hit uses the first layer whose maxVel >= velocity, and cycles through
    that layer's files so repeated hits never sound machine-gunned.
    """

    def __init__(self, path: Path, sr: int):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        root = Path(spec.get("root", "."))
        self.notes: dict[int, tuple[float, list[tuple[int, list[np.ndarray]]]]] = {}
        self._rr: dict[tuple[int, int], int] = {}

        for note_str, entry in spec.get("notes", {}).items():
            layers = []
            for vmax, files in sorted(entry.get("layers", []), key=lambda l: l[0]):
                samples = []
                for f in files:
                    full = root / f
                    try:
                        x, fsr = sf.read(str(full), dtype="float32", always_2d=True)
                    except Exception as exc:  # missing or unreadable file
                        print(f"warning: skipping {full}: {exc}")
                        continue
                    if x.shape[1] == 1:
                        x = np.repeat(x, 2, axis=1)
                    x = x[:, :2]
                    if fsr != sr:
                        g = np.gcd(int(fsr), sr)
                        x = resample_poly(x, sr // g, int(fsr) // g, axis=0).astype(np.float32)
                    samples.append(x)
                if samples:
                    layers.append((int(vmax), samples))
            if layers:
                self.notes[int(note_str)] = (float(entry.get("gain", 1.0)), layers)

    def pick(self, note: int, vel: int) -> tuple[np.ndarray, float] | None:
        """Sample for this hit (velocity layer, then round-robin) and its gain."""
        entry = self.notes.get(note)
        if entry is None:
            return None
        gain, layers = entry
        li = next((i for i, (vmax, _) in enumerate(layers) if vel <= vmax), len(layers) - 1)
        samples = layers[li][1]
        key = (note, li)
        idx = self._rr.get(key, 0)
        self._rr[key] = (idx + 1) % len(samples)
        return samples[idx], gain


# ===========================================================================
# Rendering
# ===========================================================================

def load_events(path: Path) -> list[tuple[float, int, int]]:
    """Note-on events as (seconds, note, velocity)."""
    events = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if int(row["on"]) == 1:
                    events.append((float(row["time_s"]), int(row["note"]), int(row["velocity"])))
            except (KeyError, ValueError):
                continue  # skip malformed rows rather than abort the render
    return events


def render_drums(events, n: int, sr: int, kit_path: Path | None) -> np.ndarray:
    """Stereo drum bus, n x 2."""
    bus = np.zeros((n, 2), dtype=np.float32)
    synth = None if kit_path else build_synth_kit(sr)
    sampled = SampledKit(kit_path, sr) if kit_path else None

    # Open hat is choked by the next closed hat, like a real foot pedal.
    closed_times = np.array(sorted(t for t, note, _ in events if note == 42))

    for t, note, vel in events:
        vel = max(1, min(vel, 127))

        # ---- choose the sample and its level ----------------------------
        if sampled is not None:
            picked = sampled.pick(note, vel)
            if picked is None:
                continue
            sample, kit_gain = picked
            # Layers already change the timbre, so the level curve is gentler.
            gain = kit_gain * (vel / 127.0) ** 1.2
        else:
            mono = synth.get(note)
            if mono is None:
                continue
            sample = np.stack([mono, mono], axis=1)
            gain = (vel / 127.0) ** 1.6

        start = int(round(t * sr))
        if start < 0 or start >= n:
            continue
        length = min(len(sample), n - start)

        # ---- hi-hat choke -----------------------------------------------
        if note == 46 and closed_times.size:
            idx = int(np.searchsorted(closed_times, t + 1e-4))
            if idx < closed_times.size:
                length = min(length, max(1, int((closed_times[idx] - t) * sr)))

        seg = sample[:length] * gain
        if length < len(sample):
            fade = min(256, length)   # short fade so a choked tail does not click
            seg[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
        bus[start:start + length] += seg
    return bus


def main() -> int:
    # ------------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------------
    args = sys.argv[1:]
    kit_path = None
    if "--kit" in args:
        i = args.index("--kit")
        if i + 1 >= len(args):
            print("error: --kit needs a path")
            return 2
        kit_path = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    if len(args) != 3:
        print("usage: render_preview.py <song.wav> <events.csv> <out.wav> [--kit kit.json]")
        return 2
    song_path, events_path, out_path = map(Path, args)

    # ------------------------------------------------------------------
    # Song as stereo at 44.1 kHz (upsampled if needed so hats keep their air)
    # ------------------------------------------------------------------
    song, sr = sf.read(str(song_path), dtype="float32", always_2d=True)
    if song.shape[1] == 1:
        song = np.repeat(song, 2, axis=1)
    song = song[:, :2]
    if sr != RENDER_SR:
        g = np.gcd(int(sr), RENDER_SR)
        song = resample_poly(song, RENDER_SR // g, int(sr) // g, axis=0).astype(np.float32)

    events = load_events(events_path)
    if not events:
        print("error: no note-on events found")
        return 1

    drums = render_drums(events, len(song), RENDER_SR, kit_path)

    # ------------------------------------------------------------------
    # Balance: drums sit at roughly the song's own loudness
    # ------------------------------------------------------------------
    active = np.abs(drums).max(axis=1) > 1e-4
    song_rms = float(np.sqrt(np.mean(song ** 2)) + 1e-9)
    drum_rms = float(np.sqrt(np.mean(drums[active] ** 2)) + 1e-9) if active.any() else 1.0
    drums *= 0.9 * song_rms / drum_rms

    mix = song + drums
    peak = float(np.max(np.abs(mix)))
    if peak > 0.98:
        mix *= 0.98 / peak
        drums *= 0.98 / peak

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mix, RENDER_SR, subtype="PCM_16")
    drums_path = out_path.with_name(out_path.stem + "_drums.wav")
    sf.write(str(drums_path), drums, RENDER_SR, subtype="PCM_16")

    print(f"events   {len(events)} note-ons, kit: {kit_path or 'synthesised'}")
    print(f"wrote    {out_path.resolve()}")
    print(f"wrote    {drums_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
