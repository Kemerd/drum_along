"""
beatnet.py - the learned layer, mirrored exactly from core/src/BeatNet.cpp.

    z  = clamp((f - mean) / std, -8, 8)
    h0 = elu(Win z + bin)
    h_{i+1} = h_i + elu(causal_dilated_conv_k3(h_i, d_i))
    y  = sigmoid(Whead h + bhead)      -> [beat, downbeat, drums]

Causality: each conv pads 2*d frames on the left only, so output t reads
inputs t-2d, t-d, t (PyTorch tap k=0 is t-2d, k=2 is t). The C++ ring
buffers read the same three frames in the same order.
"""

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_FEATURES = 135
NUM_OUTPUTS = 3
OUTPUT_NAMES = ("beat", "downbeat", "drums")


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):                                  # h: [B, C, T]
        y = self.conv(F.pad(h, (2 * self.dilation, 0)))    # left pad only = causal
        return h + self.drop(F.elu(y))


class BeatNet(nn.Module):
    def __init__(self, channels: int = 32, dilations=(1, 2, 4, 8, 16, 32, 64, 128), dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.dilations = list(dilations)
        self.register_buffer("feat_mean", torch.zeros(NUM_FEATURES))
        self.register_buffer("feat_std", torch.ones(NUM_FEATURES))
        self.inp = nn.Linear(NUM_FEATURES, channels)
        self.blocks = nn.ModuleList(CausalBlock(channels, d, dropout) for d in self.dilations)
        self.head = nn.Linear(channels, NUM_OUTPUTS)

    def forward(self, x):                                  # x: [B, T, F] raw features -> logits [B, T, 3]
        z = ((x - self.feat_mean) / self.feat_std.clamp_min(1e-6)).clamp(-8.0, 8.0)
        h = F.elu(self.inp(z)).transpose(1, 2)             # [B, C, T]
        for blk in self.blocks:
            h = blk(h)
        return self.head(h.transpose(1, 2))

    def receptive_field(self) -> int:
        return 1 + 2 * sum(self.dilations)

    # ---------------------------------------------------------------------
    # Export in the layout BeatNet.cpp / WeightsIO.cpp expect
    # ---------------------------------------------------------------------
    def export_json(self, path, training: dict):
        def flat(t):
            return [float(f"{v:.7g}") for v in t.detach().cpu().reshape(-1).tolist()]

        doc = {
            "format": "drum_along.beatnet",
            "version": 1,
            "numFeatures": NUM_FEATURES,
            "numOutputs": NUM_OUTPUTS,
            "outputs": list(OUTPUT_NAMES),
            "channels": self.channels,
            "dilations": self.dilations,
            "featMean": flat(self.feat_mean),
            "featStd": flat(self.feat_std.clamp_min(1e-6)),
            "inW": flat(self.inp.weight),                  # [C, F] row-major
            "inB": flat(self.inp.bias),
            "blocks": [{"w": flat(b.conv.weight), "b": flat(b.conv.bias)} for b in self.blocks],  # [out, in, k]
            "headW": flat(self.head.weight),               # [3, C]
            "headB": flat(self.head.bias),
            "training": {**training, "exported": time.strftime("%Y-%m-%d %H:%M:%S")},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
