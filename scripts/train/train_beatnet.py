"""
train_beatnet.py - train the learned beat / downbeat / drums layer.

Usage (training env with CUDA PyTorch):
    .venv-sep/Scripts/python.exe scripts/train/train_beatnet.py <features_root> <out.json>
        [--epochs 40] [--steps 300] [--batch 64] [--chunk 1500] [--channels 32] [--val-split validation]
        [--dilations 1,2,4,...] [--down-weight 0.5] [--drums-weight 0.5] [--val-limit 80]

<features_root>/<split>/*.npz come from make_features.py. Training uses every
split except --val-split (default "validation"); the best epoch by validation
beat F-measure plus downbeat accuracy is exported to <out.json> for
weights/drummer.json to name.

Targets (Boeck-style widening): the frame nearest each beat is 1.0 and its two
neighbours 0.5, so a prediction one frame (10 ms) off still earns most of the
credit. Downbeat loss is masked for songs not in 4/4, and for songs whose MIDI
bar lines disagree with their own drum part (see scripts/diag/grid_census.py).
"""

import json
import math
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beatnet import BeatNet, NUM_FEATURES  # noqa: E402

ATTRIBUTION = ("Slakh2100 (Manilow et al., WASPAA 2019), CC BY 4.0, "
               "https://zenodo.org/records/4599666")


# ===========================================================================
# Data
# ===========================================================================

class FeatureFile:
    """
    One song's features on disk, read a slice at a time.

    The full set is tens of GB and this often runs next to other heavy jobs. A
    memory map would leave every page it touches in this process's working set
    until Windows decides to trim it. Positioned reads keep the process flat:
    the OS file cache still serves repeats, and it hands that RAM back on demand.
    """

    def __init__(self, path: Path):
        self.path = str(path)
        with open(path, "rb") as f:
            major, _ = np.lib.format.read_magic(f)
            read_header = (np.lib.format.read_array_header_1_0 if major == 1
                           else np.lib.format.read_array_header_2_0)
            self.shape, fortran, self.dtype = read_header(f)
            self.offset = f.tell()
        if fortran or len(self.shape) != 2:
            raise ValueError(f"{path}: expected a C-order 2-D array")
        self.ndim = 2
        self.row_bytes = self.shape[1] * self.dtype.itemsize

    def __getitem__(self, rows: slice) -> np.ndarray:
        """Rows [start, stop) as a new in-memory array."""
        start, stop, step = rows.indices(self.shape[0])
        if step != 1 or stop <= start:
            raise ValueError(f"{self.path}: unsupported row range {rows}")
        a = np.fromfile(self.path, dtype=self.dtype, count=(stop - start) * self.shape[1],
                        offset=self.offset + start * self.row_bytes)
        return a.reshape(-1, self.shape[1])


def widen(lbl: np.ndarray) -> np.ndarray:
    """1 at labelled frames, 0.5 at their neighbours."""
    t = lbl.astype(np.float32)
    side = np.zeros_like(t)
    side[1:] = np.maximum(side[1:], t[:-1])
    side[:-1] = np.maximum(side[:-1], t[1:])
    return np.maximum(t, 0.5 * side)


def load_split(files, lazy=True, grid=None):
    """
    Examples as dicts of features + targets. With lazy=True the features stay
    on disk and batches read only their chunks; otherwise they load into RAM.
    grid is the per-song verdict from grid_census.py, or None to trust the MIDI.
    """
    grid = grid or {}
    data = []
    for p in files:
        try:
            z = np.load(p)
            x_path = p.with_suffix(".x.npy")
            if x_path.exists():
                x = FeatureFile(x_path) if lazy else np.load(x_path)
            else:
                x = z["x"]   # older single-file format
            if x.ndim != 2 or x.shape[1] != NUM_FEATURES or x.shape[0] < 600:
                continue
            bpm = float(z["bpm"])
            if not (50.0 <= bpm <= 220.0):
                continue   # MIDI tempo at an implausible metrical level
            data.append({
                "name": p.stem,
                "x": x,
                "beat": widen(z["beat"]),
                "down": widen(z["down"]),
                "drums": z["drums"].astype(np.float32),
                "down_ok": bool(z["down_ok"]) and grid.get(p.stem.split("_")[0], {}).get("kind", "none") in ("backbeat", "none"),
                "beat_frames": np.nonzero(z["beat"])[0],
                "down_frames": np.nonzero(z["down"])[0],
                "hop": int(z["hop"]), "win": int(z["win"]), "sr": float(z["sr"]),
            })
        except Exception as exc:
            print(f"skip {p.name}: {exc}")
    return data


def feature_stats(data, max_frames=2_000_000):
    rng = np.random.default_rng(0)
    rows = []
    per = max(1, max_frames // max(1, len(data)))
    for d in data:
        # One contiguous stretch per song: a seek per frame would cost far more
        # and thousands of 5 s stretches sample the features well enough.
        T = d["x"].shape[0]
        n = min(per, T)
        s = int(rng.integers(0, T - n + 1))
        rows.append(d["x"][s:s + n].astype(np.float32))
    allx = np.concatenate(rows)
    return allx.mean(0), allx.std(0) + 1e-4


def sample_batch(data, batch, chunk, rng):
    xs, ys, ms = [], [], []
    for _ in range(batch):
        d = data[rng.integers(len(data))]
        T = d["x"].shape[0]
        s = int(rng.integers(0, max(1, T - chunk)))
        e = min(T, s + chunk)
        x = d["x"][s:e].astype(np.float32)
        y = np.stack([d["beat"][s:e], d["down"][s:e], d["drums"][s:e]], axis=1)
        m = np.ones_like(y)
        if not d["down_ok"]:
            m[:, 1] = 0.0
        if e - s < chunk:                       # pad short songs
            pad = chunk - (e - s)
            x = np.pad(x, ((0, pad), (0, 0)))
            y = np.pad(y, ((0, pad), (0, 0)))
            m = np.pad(m, ((0, pad), (0, 0)))
        xs.append(x); ys.append(y); ms.append(m)
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ms)))


# ===========================================================================
# Validation: simple peak picking on the beat activation
# ===========================================================================

def peak_pick(act: np.ndarray, thresh=0.3, min_dist=15):
    peaks = []
    last = -10**9
    for t in range(1, len(act) - 1):
        if act[t] >= thresh and act[t] >= act[t - 1] and act[t] > act[t + 1] and t - last >= min_dist:
            peaks.append(t)
            last = t
    return np.array(peaks, dtype=np.int64)


@torch.no_grad()
def validate(model, data, device):
    import mir_eval

    model.eval()
    fs, losses, slots = [], [], []
    for d in data:
        x = torch.from_numpy(d["x"].astype(np.float32))[None].to(device)
        logits = model(x)[0]
        act = torch.sigmoid(logits[:, 0]).cpu().numpy()
        down = torch.sigmoid(logits[:, 1]).cpu().numpy()
        y = torch.from_numpy(d["beat"]).to(device)
        losses.append(F.binary_cross_entropy_with_logits(logits[:, 0], y).item())
        to_s = lambda f: ((f + 1) * d["hop"] - d["win"] / 2.0) / d["sr"]
        ref = to_s(d["beat_frames"].astype(np.float64))
        est = to_s(peak_pick(act).astype(np.float64))
        ref, est = ref[ref > 5.0], est[est > 5.0]      # skip the cold-start seconds
        fs.append(mir_eval.beat.f_measure(ref, est) if ref.size and est.size else 0.0)

        # Downbeat: handed the true beats, does the head pick the right slot?
        beats = d["beat_frames"]
        if d["down_ok"] and beats.size >= 40 and d["down_frames"].size:
            anchor = int(np.argmin(np.abs(beats - d["down_frames"][0])))
            score = np.zeros(4)
            for k, b in enumerate(beats):
                if b >= 500:                           # same cold-start skip as above
                    score[(k - anchor) % 4] += down[max(b - 2, 0):b + 5].max()
            slots.append(int(np.argmax(score)) == 0)
    model.train()
    return float(np.mean(fs)), float(np.mean(losses)), float(np.mean(slots)) if slots else 0.0


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 2
    root, out = Path(args[0]), Path(args[1])
    opt = {"epochs": 40, "steps": 300, "batch": 64, "chunk": 1500, "channels": 32,
           "val-split": "validation", "val-limit": 80, "lr": 2e-3, "seed": 0,
           "dilations": "1,2,4,8,16,32,64,128", "down-weight": 0.5, "drums-weight": 0.5}
    for i in range(2, len(args) - 1, 2):
        key = args[i].lstrip("-")
        if key in opt:
            opt[key] = type(opt[key])(args[i + 1])

    torch.manual_seed(opt["seed"])
    rng = np.random.default_rng(opt["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data ---------------------------------------------------------------
    splits = sorted(p for p in root.iterdir() if p.is_dir())
    val_files = sorted((root / opt["val-split"]).glob("*_full.npz")) if (root / opt["val-split"]).exists() else []
    if opt["val-limit"]:
        val_files = val_files[: opt["val-limit"]]
    train_files = [f for s in splits if s.name not in (opt["val-split"], "test")
                   for f in sorted(s.glob("*.npz")) if not f.name.endswith(".tmp.npz")]
    t0 = time.time()
    grid_path = root / "grid.json"
    grid = json.loads(grid_path.read_text(encoding="utf-8")) if grid_path.exists() else None
    train = load_split(train_files, grid=grid)
    val = load_split(val_files, lazy=False, grid=grid)
    if not train:
        print("error: no training examples")
        return 1
    hours = sum(d["x"].shape[0] for d in train) / 100 / 3600
    print(f"train {len(train)} examples ({hours:.1f} h), val {len(val)}, loaded in {time.time() - t0:.0f}s, device {device}, "
          f"bar lines used {sum(d['down_ok'] for d in train) / len(train):.0%}")

    # ---- model ----------------------------------------------------------------
    dilations = [int(d) for d in opt["dilations"].split(",")]
    model = BeatNet(channels=opt["channels"], dilations=dilations).to(device)
    mean, std = feature_stats(train)
    model.feat_mean.copy_(torch.from_numpy(mean))
    model.feat_std.copy_(torch.from_numpy(std))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} weights, receptive field {model.receptive_field()} frames")

    optim = torch.optim.AdamW(model.parameters(), lr=opt["lr"], weight_decay=1e-4)
    total = opt["epochs"] * opt["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    head_w = torch.tensor([1.0, opt["down-weight"], opt["drums-weight"]], device=device)

    best_score, best_f, best_state, history = -1.0, -1.0, None, []
    for epoch in range(1, opt["epochs"] + 1):
        running = 0.0
        for _ in range(opt["steps"]):
            x, y, m = (t.to(device) for t in sample_batch(train, opt["batch"], opt["chunk"], rng))
            logits = model(x)
            # Beats are rare frames, so they are weighted up. For the drums head
            # the rare case is the opposite one: a drummer is playing 88% of
            # the time, and the moments he is not are the whole point of it.
            frame_w = 1.0 + 4.0 * y
            frame_w[..., 2] = 1.0 + 6.0 * (1.0 - y[..., 2])
            w = frame_w * m * head_w
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()

        vf, vl, vd = validate(model, val, device) if val else (0.0, 0.0, 0.0)
        history.append({"epoch": epoch, "train_loss": running / opt["steps"], "val_beat_f": vf, "val_loss": vl,
                        "val_down_slot": vd})
        mark = ""
        # Same balance as the whole-drummer score in tune_weights.py.
        score = vf + 0.3 * vd
        if score > best_score or not val:
            best_score, best_f = score, vf
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            mark = "  *best"
        print(f"epoch {epoch:3d}  train {running / opt['steps']:.4f}  val loss {vl:.4f}  val beat F {vf:.3f}  "
              f"down slot {vd:.3f}{mark}", flush=True)

    model.load_state_dict(best_state)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.export_json(out, {
        "data": ATTRIBUTION, "examples": len(train), "hours": round(hours, 1),
        "val_examples": len(val), "best_val_beat_f": round(best_f, 4), "options": opt, "history": history,
    })
    torch.save(best_state, out.with_suffix(".pt"))
    print(f"exported {out}  (best val beat F {best_f:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
