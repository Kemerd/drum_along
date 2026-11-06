#pragma once
/**
 * @file BeatNet.h
 * @brief The one *learned* layer of the drummer: a small causal temporal
 *        convolution network that turns per-frame features into three
 *        activations: "a beat is here", "a downbeat is here", and "a drummer
 *        would be playing here".
 *
 * Architecture (identical in scripts/train/beatnet.py):
 *
 *   z  = clamp((f - mean) / std, -8, 8)                   features -> z
 *   h0 = elu(Win z + bin)                                 C channels
 *   for block i, dilation d_i:                            (d = 1, 2, 4 ... 128)
 *     h_{i+1}[t] = h_i[t] + elu( sum_k W_i[:,:,k] h_i[t - (2-k) d_i] + b_i )
 *   y  = sigmoid(Whead h_last + bhead)                    3 outputs
 *
 * Only past frames are read (taps at t, t-d, t-2d), so it streams with zero
 * look-ahead. Receptive field with 8 blocks: 511 frames, about 5 seconds.
 *
 * Trained on Slakh2100 (CC BY 4.0) drumless backings with labels from the
 * songs' own MIDI. The weights live in weights/beatnet.json, separate from the
 * hand-written weights/drummer.json.
 *
 * Real-time contract: prepare() allocates; reset() and step() never allocate.
 */

#include "Config.h"

#include <cstdint>
#include <vector>

namespace drumalong {

/// Number of per-frame features the network reads (see Network::assembleFeatures).
constexpr int kBeatFeatures = 135;

/// Outputs of the learned layer.
enum BeatNetOutput : int {
    kOutBeat = 0,
    kOutDownbeat,
    kOutDrums,
    kNumBeatNetOutputs
};

/**
 * @brief Trained parameters of the beat network. Empty = network disabled.
 */
struct BeatNetWeights {
    int channels = 0;                       ///< C. 0 means "no learned layer".
    std::vector<int>   dilations;           ///< One per block.
    std::vector<float> featMean;            ///< kBeatFeatures.
    std::vector<float> featStd;             ///< kBeatFeatures.
    std::vector<float> inW;                 ///< C x kBeatFeatures, row-major.
    std::vector<float> inB;                 ///< C.
    std::vector<std::vector<float>> blockW; ///< Per block: C x C x 3, layout [out][in][tap].
    std::vector<std::vector<float>> blockB; ///< Per block: C.
    std::vector<float> headW;               ///< kNumBeatNetOutputs x C.
    std::vector<float> headB;               ///< kNumBeatNetOutputs.

    bool enabled() const { return channels > 0; }

    /// Shape check: true when every tensor has the size the architecture needs.
    bool valid() const;
};

/**
 * @brief Streaming inference for BeatNetWeights.
 */
class BeatNet {
public:
    /**
     * @brief Bind to trained weights and allocate the per-block histories.
     * @return false if the weights are missing or malformed.
     */
    bool prepare(const BeatNetWeights* weights);

    /// Clear all history (silence in every ring buffer).
    void reset();

    /**
     * @brief Advance one frame.
     * @param features kBeatFeatures raw (un-normalised) values.
     * @param out      Receives kNumBeatNetOutputs probabilities in [0, 1].
     */
    void step(const float* features, float* out);

    bool isPrepared() const { return w_ != nullptr; }

private:
    const BeatNetWeights* w_ = nullptr;
    int C_ = 0;

    std::vector<float> z_;      ///< Normalised features.
    std::vector<float> h_;      ///< Current activation (C).
    std::vector<float> acc_;    ///< Scratch (C).

    /// Per block: ring of past block inputs, ringLen x C, plus its write head.
    struct Ring {
        std::vector<float> data;
        int len  = 0;           ///< Power of two >= 2*d + 1.
        int mask = 0;
        int head = 0;           ///< Index of the newest frame.
    };
    std::vector<Ring> rings_;
};

} // namespace drumalong
