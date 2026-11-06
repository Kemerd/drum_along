/**
 * @file BeatNet.cpp
 * @brief Streaming inference of the learned beat / downbeat / drums layer.
 */

#include "drumalong/BeatNet.h"
#include "drumalong/Kernels.h"

#include <algorithm>
#include <cmath>

namespace drumalong {

namespace {

inline float elu(float x) {
    return x > 0.0f ? x : std::expm1(x);
}

inline float sigmoid(float x) {
    // Clamp so expf can never overflow on a corrupted weight.
    x = std::clamp(x, -30.0f, 30.0f);
    return 1.0f / (1.0f + std::exp(-x));
}

int nextPow2(int n) {
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

} // namespace

// ===========================================================================
// Weights validation
// ===========================================================================

bool BeatNetWeights::valid() const {
    if (channels <= 0 || channels > 256) {
        return false;
    }
    const size_t C = static_cast<size_t>(channels);
    if (dilations.empty() || dilations.size() > 16 ||
        blockW.size() != dilations.size() || blockB.size() != dilations.size()) {
        return false;
    }
    for (size_t i = 0; i < dilations.size(); ++i) {
        if (dilations[i] < 1 || dilations[i] > 512 ||
            blockW[i].size() != C * C * 3 || blockB[i].size() != C) {
            return false;
        }
    }
    return featMean.size() == static_cast<size_t>(kBeatFeatures) &&
           featStd.size()  == static_cast<size_t>(kBeatFeatures) &&
           inW.size()      == C * kBeatFeatures &&
           inB.size()      == C &&
           headW.size()    == C * kNumBeatNetOutputs &&
           headB.size()    == static_cast<size_t>(kNumBeatNetOutputs);
}

// ===========================================================================
// Lifecycle
// ===========================================================================

bool BeatNet::prepare(const BeatNetWeights* weights) {
    w_ = nullptr;
    if (!weights || !weights->valid()) {
        return false;
    }
    w_ = weights;
    C_ = weights->channels;

    z_.assign(kBeatFeatures, 0.0f);
    h_.assign(static_cast<size_t>(C_), 0.0f);
    acc_.assign(static_cast<size_t>(C_), 0.0f);

    rings_.assign(weights->dilations.size(), Ring{});
    for (size_t i = 0; i < rings_.size(); ++i) {
        Ring& r = rings_[i];
        r.len  = nextPow2(2 * weights->dilations[i] + 1);
        r.mask = r.len - 1;
        r.data.assign(static_cast<size_t>(r.len) * C_, 0.0f);
    }
    reset();
    return true;
}

void BeatNet::reset() {
    std::fill(z_.begin(), z_.end(), 0.0f);
    std::fill(h_.begin(), h_.end(), 0.0f);
    for (Ring& r : rings_) {
        std::fill(r.data.begin(), r.data.end(), 0.0f);
        r.head = 0;
    }
}

// ===========================================================================
// One frame
// ===========================================================================

void BeatNet::step(const float* features, float* out) {
    if (!out) {
        return;
    }
    if (!w_ || !features) {
        std::fill(out, out + kNumBeatNetOutputs, 0.0f);
        return;
    }
    const BeatNetWeights& w = *w_;
    const int C = C_;

    // ---- Normalise the features with the training statistics -------------
    for (int f = 0; f < kBeatFeatures; ++f) {
        const float sd = w.featStd[static_cast<size_t>(f)];
        float v = (features[f] - w.featMean[static_cast<size_t>(f)]) / (sd > 1e-6f ? sd : 1.0f);
        if (!std::isfinite(v)) {
            v = 0.0f;
        }
        z_[static_cast<size_t>(f)] = std::clamp(v, -8.0f, 8.0f);
    }

    // ---- Input projection -------------------------------------------------
    for (int c = 0; c < C; ++c) {
        h_[static_cast<size_t>(c)] =
            elu(kernels::dot(w.inW.data() + static_cast<size_t>(c) * kBeatFeatures, z_.data(), kBeatFeatures) +
                w.inB[static_cast<size_t>(c)]);
    }

    // ---- Residual dilated causal blocks ------------------------------------
    for (size_t b = 0; b < rings_.size(); ++b) {
        Ring& r = rings_[b];
        const int d = w.dilations[b];

        // Store this block's input as the newest frame.
        r.head = (r.head + 1) & r.mask;
        std::copy(h_.begin(), h_.end(), r.data.begin() + static_cast<std::ptrdiff_t>(r.head) * C);

        // Taps: k = 2 -> t, k = 1 -> t - d, k = 0 -> t - 2d (PyTorch order).
        const float* x2 = r.data.data() + static_cast<size_t>(r.head) * C;
        const float* x1 = r.data.data() + static_cast<size_t>((r.head - d) & r.mask) * C;
        const float* x0 = r.data.data() + static_cast<size_t>((r.head - 2 * d) & r.mask) * C;

        const float* W = w.blockW[b].data();
        for (int o = 0; o < C; ++o) {
            float s = w.blockB[b][static_cast<size_t>(o)];
            const float* wo = W + static_cast<size_t>(o) * C * 3;   // [in][tap] for this output
            for (int i = 0; i < C; ++i) {
                const float* wi = wo + static_cast<size_t>(i) * 3;
                s += wi[0] * x0[i] + wi[1] * x1[i] + wi[2] * x2[i];
            }
            acc_[static_cast<size_t>(o)] = elu(s);
        }
        for (int c = 0; c < C; ++c) {
            h_[static_cast<size_t>(c)] = x2[c] + acc_[static_cast<size_t>(c)];
        }
    }

    // ---- Heads ------------------------------------------------------------------
    for (int k = 0; k < kNumBeatNetOutputs; ++k) {
        out[k] = sigmoid(kernels::dot(w.headW.data() + static_cast<size_t>(k) * C, h_.data(), C) +
                         w.headB[static_cast<size_t>(k)]);
    }
}

} // namespace drumalong
