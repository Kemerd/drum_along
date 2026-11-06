/**
 * @file Network.cpp
 * @brief Forward pass of the hand-programmed drummer network.
 *
 * Every number used here comes from Weights (hand-written defaults, or the
 * editable weights file). This file only knows *how* to apply them:
 * multiply, square, median, log, rectify, integrate, argmax, threshold.
 */

#include "drumalong/Network.h"
#include "drumalong/Kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace drumalong {

namespace {

/// Wrap a phase difference (in beats) into [-0.5, 0.5).
inline double wrapHalf(double x) {
    return x - std::floor(x + 0.5);
}

/// Non-negative modulo 4, safe for negative beat numbers.
inline int mod4(int64_t n) {
    const int r = static_cast<int>(n % 4);
    return (r < 0) ? r + 4 : r;
}

/**
 * @brief Sub-sample peak refinement by fitting a parabola through 3 points.
 * @return Offset in [-0.5, 0.5] to add to the centre index.
 */
inline double parabolicOffset(double left, double centre, double right) {
    const double denom = left - 2.0 * centre + right;
    if (!(std::fabs(denom) > 1e-12)) {
        return 0.0;
    }
    return std::clamp(0.5 * (left - right) / denom, -0.5, 0.5);
}

/// Replace NaN / inf with zero so one bad value cannot poison the state.
inline float finiteOr0(float v) {
    return std::isfinite(v) ? v : 0.0f;
}

/// Median of n >= 1 floats. Reorders the buffer.
inline float medianInPlace(float* buf, int n) {
    std::nth_element(buf, buf + n / 2, buf + n);
    return buf[n / 2];
}

} // namespace

// ===========================================================================
// Lifecycle
// ===========================================================================

bool Network::prepare(const Weights* weights) {
    w_ = nullptr;

    // -----------------------------------------------------------------------
    // Validate every realized tensor before trusting the weights.
    // -----------------------------------------------------------------------
    if (!weights) {
        return false;
    }
    const Weights& w = *weights;
    if (w.windowSamples <= 0 || w.hopSamples <= 0 || w.numBins <= 1) {
        return false;
    }
    if (static_cast<int>(w.dftWindow.size()) != w.windowSamples || w.numBins > w.windowSamples / 2 + 1 ||
        w.mel.rows != kMelBands || w.mel.cols != w.numBins ||
        static_cast<int>(w.lowBand.size()) != w.numBins ||
        static_cast<int>(w.fluxReadout.size()) != kMelBands ||
        static_cast<int>(w.tempoPrior.size()) != kNumPeriods) {
        return false;
    }
    if (w.chromaWindowSamples <= 0 || w.numChromaBins <= 0 ||
        static_cast<int>(w.chromaWindow.size()) != w.chromaWindowSamples ||
        w.chromaBinLo < 0 || w.chromaBinLo + w.numChromaBins > w.chromaWindowSamples / 2 + 1 ||
        w.chroma.rows != kChromaClasses || w.chroma.cols != w.numChromaBins) {
        return false;
    }
    if (!fft_.prepare(w.windowSamples) || !fftChroma_.prepare(w.chromaWindowSamples)) {
        return false;
    }
    if (w.hpssTimeFrames < 1 || w.hpssFreqBins < 1 ||
        w.harmonicSpanFrames < 2 || 2 * w.harmonicSpanFrames > kChromaHist ||
        w.harmonicOffsetFrames < 1 || w.harmonicOffsetFrames >= kMinPeriodFrames - 1) {
        return false;
    }

    // -----------------------------------------------------------------------
    // Allocate every buffer up front: step() must never allocate.
    // -----------------------------------------------------------------------
    w_ = weights;
    const size_t nb = static_cast<size_t>(w.numBins);
    spec_.assign(2 * nb, 0.0f);
    power_.assign(nb, 0.0f);
    percPower_.assign(nb, 0.0f);
    powerHist_.assign(static_cast<size_t>(w.hpssTimeFrames) * nb, 0.0f);
    medScratch_.assign(static_cast<size_t>(std::max(w.hpssTimeFrames, w.hpssFreqBins)), 0.0f);
    melLog_.assign(kMelBands, 0.0f);
    melPrev_.assign(kMelBands, 0.0f);
    melLogPerc_.assign(kMelBands, 0.0f);
    melPrevPerc_.assign(kMelBands, 0.0f);

    chromaSpec_.assign(static_cast<size_t>(2 * w.numChromaBins), 0.0f);
    chromaPower_.assign(static_cast<size_t>(w.numChromaBins), 0.0f);
    chromaHist_.assign(static_cast<size_t>(kChromaHist * kChromaClasses), 0.0f);

    xRaw_.assign(kHist, 0.0f);
    xs_.assign(kHist, 0.0f);
    lowX_.assign(kHist, 0.0f);
    lowXs_.assign(kHist, 0.0f);
    hX_.assign(kHist, 0.0f);
    levelHist_.assign(kHist, 0.5f);

    R_.assign(static_cast<size_t>(kMaxLag + 1), 0.0);
    Rh_.assign(static_cast<size_t>(kMaxHarmLag + 1), 0.0);
    S_.assign(kNumPeriods, 0.0f);
    phaseScore_.assign(static_cast<size_t>(kMaxPeriodFrames + 2), 0.0f);

    // Learned layer (optional) and its feature buffers.
    feat_.assign(kBeatFeatures, 0.0f);
    riseX_.assign(kMelBands, 0.0f);
    riseP_.assign(kMelBands, 0.0f);
    downHist_.assign(kHist, 0.0f);
    useLearned_ = w.beatNet.enabled();
    if (useLearned_ && !beatNet_.prepare(&w.beatNet)) {
        w_ = nullptr;
        return false;
    }

    reset();
    return true;
}

void Network::reset() {
    // Front end.
    std::fill(spec_.begin(), spec_.end(), 0.0f);
    std::fill(power_.begin(), power_.end(), 0.0f);
    std::fill(percPower_.begin(), percPower_.end(), 0.0f);
    std::fill(powerHist_.begin(), powerHist_.end(), 0.0f);
    std::fill(melLog_.begin(), melLog_.end(), 0.0f);
    std::fill(melPrev_.begin(), melPrev_.end(), 0.0f);
    std::fill(melLogPerc_.begin(), melLogPerc_.end(), 0.0f);
    std::fill(melPrevPerc_.begin(), melPrevPerc_.end(), 0.0f);
    lowPrev_       = 0.0f;
    havePrev_      = false;
    powerHistPos_  = 0;
    powerHistFill_ = 0;

    // Chord channel.
    std::fill(chromaSpec_.begin(), chromaSpec_.end(), 0.0f);
    std::fill(chromaPower_.begin(), chromaPower_.end(), 0.0f);
    std::fill(chromaHist_.begin(), chromaHist_.end(), 0.0f);
    chromaFill_ = 0;
    lastHarm_   = 0.0f;
    lastChange_ = 0.0f;

    // Learned layer.
    if (useLearned_) {
        beatNet_.reset();
    }
    std::fill(feat_.begin(), feat_.end(), 0.0f);
    std::fill(riseX_.begin(), riseX_.end(), 0.0f);
    std::fill(riseP_.begin(), riseP_.end(), 0.0f);
    std::fill(downHist_.begin(), downHist_.end(), 0.0f);
    std::fill(std::begin(learnedOut_), std::end(learnedOut_), 0.0f);
    drumsSmooth_ = 1.0f;

    // Histories.
    std::fill(xRaw_.begin(), xRaw_.end(), 0.0f);
    std::fill(xs_.begin(), xs_.end(), 0.0f);
    std::fill(lowX_.begin(), lowX_.end(), 0.0f);
    std::fill(lowXs_.begin(), lowXs_.end(), 0.0f);
    std::fill(hX_.begin(), hX_.end(), 0.0f);

    // Silence gate starts closed: nothing has been heard yet.
    silenceEnv_  = 0.0f;
    silentCount_ = 0;

    // Phase-locked loop.
    B_           = 0.0;
    lastBeatInt_ = -1;

    resetEstimators();

    frame_ = 0;
    state_ = FrameState{};
}

/**
 * Clears everything that summarises "the current song": tempo, lock, bar,
 * intensity range, normalisers. Called by reset() and after a long silence,
 * so the next song starts clean instead of inheriting the last one's tempo.
 */
void Network::resetEstimators() {
    fluxMean_     = 0.0f;
    lowFluxMean_  = 0.0f;
    fluxNormCorr_ = 0.0f;
    harmMean_     = 0.0f;
    harmCorr_     = 0.0f;
    loudMean_     = 0.0f;
    loudMeanCorr_ = 0.0f;

    std::fill(R_.begin(), R_.end(), 0.0);
    std::fill(Rh_.begin(), Rh_.end(), 0.0);
    std::fill(S_.begin(), S_.end(), 0.0f);
    std::fill(phaseScore_.begin(), phaseScore_.end(), 0.0f);

    curL_        = static_cast<float>(60.0 * kFrameRate / 110.0);
    lStar_       = curL_;
    periodicity_ = 0.0f;
    switchCount_ = 0;
    lockCount_   = 0;
    locked_      = false;
    everLocked_  = false;
    soundFrames_ = 0;
    framesSinceLock_ = 0;

    snapCount_   = 0;
    quietFrames_ = 0;
    bandPaused_  = false;
    phaseError_  = 0.0f;
    rebasedFlag_ = false;

    clearDownbeat();
    downbeatSlot_   = 0;
    firstLockEvent_ = false;

    loudFast_  = 0.0f;
    densFast_  = 0.0f;
    loudMin_   = loudMax_ = 0.0f;
    densMin_   = densMax_ = 0.0f;
    fastInit_  = false;
    rangeInit_ = false;
    level_     = 0.5f;
    std::fill(levelHist_.begin(), levelHist_.end(), 0.5f);
    sectionCooldown_ = 0;
    sectionFlag_     = false;
}

/// How strongly the drummer bends toward the band (see Weights::leadership).
float Network::couplingScale() const {
    return 1.0f - 0.8f * std::clamp(w_->leadership, 0.0f, 1.0f);
}

// ===========================================================================
// Forward pass
// ===========================================================================

const FrameState& Network::step(const float* window) {
    FrameState& s = state_;
    if (!w_ || !window) {
        return s;   // Not prepared or no input: report the last state unchanged.
    }
    const Weights& w = *w_;

    s.newBeat       = false;
    s.rebased       = false;
    s.sectionChange = false;

    // The onset DFT reads the newest windowSamples of the long input window.
    const float* shortWindow = window + (w.inputWindowSamples() - w.windowSamples);

    // Front end: audio -> blended flux, sub-bass flux, loudness.
    float flux = 0.0f, lowFlux = 0.0f, loudness = 0.0f;
    runFrontEnd(shortWindow, flux, lowFlux, loudness);

    // Silence gate, chord channel.
    updateSilence(loudness);
    runChord(window);

    // Learned layer: features -> beat / downbeat / drums activations.
    assembleFeatures(lowFlux, loudness);
    runLearned();

    // A real stop in the band reopens the settling window when it comes back.
    updatePause();

    // Normalised onsets (hand-written flux, blended with the learned beat).
    updateOnsets(flux, lowFlux);

    // Tempo.
    updateAutocorrelation();
    bool tempoChanged = false;
    updateTempo(tempoChanged);
    if (tempoChanged && !firstLockEvent_) {
        // A real tempo switch means a new beat grid: old bar accents no longer
        // apply. The first lock is not a switch: the period was already being
        // followed, so the accents gathered so far are still valid evidence.
        clearDownbeat();
    }

    // Phase, bar, intensity.
    updatePhase(tempoChanged);
    updateDownbeat();
    updateIntensity(loudness, flux);

    // -----------------------------------------------------------------------
    // Publish. The engine computes this frame once (frame_ + 1) * hop samples
    // have arrived, and the onset window ends there, so its centre sits N/2
    // samples earlier.
    // -----------------------------------------------------------------------
    // timingOffsetMs: onsets register slightly late in the analysis, so the
    // frame is credited to an earlier instant and every hit moves earlier.
    s.frameIndex    = frame_;
    s.centerSample  = static_cast<double>(frame_ + 1) * w.hopSamples - 0.5 * w.windowSamples -
                      static_cast<double>(w.timingOffsetMs) * 0.001 * w.sampleRate;
    s.flux          = flux;
    s.onset         = xRaw_[static_cast<size_t>(frame_ & kHistMask)];
    s.lowOnset      = lowX_[static_cast<size_t>(frame_ & kHistMask)];
    s.harmonic      = lastHarm_;
    s.loudness      = loudness;
    s.periodicity   = periodicity_;
    s.periodFrames  = curL_;
    s.tempoBpm      = (curL_ > 0.0f) ? 60.0f * kFrameRate / curL_ : 0.0f;
    s.locked        = locked_;
    s.beatPos       = B_;
    s.phaseError    = phaseError_;
    s.rebased       = rebasedFlag_;
    s.downbeatSlot  = downbeatSlot_;
    s.level         = level_;
    s.sectionChange = sectionFlag_;
    s.silent        = silentCount_ >= w.silenceFrames;
    s.beatAct       = learnedOut_[kOutBeat];
    s.downbeatAct   = learnedOut_[kOutDownbeat];
    s.drumsActive   = drumsSmooth_;
    s.features      = feat_.data();
    s.numFeatures   = kBeatFeatures;

    ++frame_;
    return s;
}

// ---------------------------------------------------------------------------
// Learned layer: features and inference
// ---------------------------------------------------------------------------
/**
 * The 135 per-frame features the learned layer reads (and that training
 * exports with wav2mid --features). All are level-independent or relative:
 *
 *   [0, 40)    log-mel spectral shape (each band minus the frame mean)
 *   [40, 80)   plain per-band flux (rectified rise, SuperFlux max filter)
 *   [80, 120)  transient-only per-band flux
 *   120        sub-bass flux
 *   [121, 133) pitch-class profile, normalised to sum 1
 *   133        raw chord change (1 - cosine, before normalisation)
 *   134        loudness relative to its own 3 s running mean
 */
void Network::assembleFeatures(float lowFlux, float loudness) {
    const Weights& w = *w_;
    float* f = feat_.data();

    float mean = 0.0f;
    for (int b = 0; b < kMelBands; ++b) {
        mean += melLog_[static_cast<size_t>(b)];
    }
    mean /= kMelBands;
    for (int b = 0; b < kMelBands; ++b) {
        f[b]                 = melLog_[static_cast<size_t>(b)] - mean;
        f[kMelBands + b]     = riseX_[static_cast<size_t>(b)];
        f[2 * kMelBands + b] = riseP_[static_cast<size_t>(b)];
    }
    f[120] = lowFlux;

    const float* row = chromaHist_.data() + static_cast<size_t>(frame_ % kChromaHist) * kChromaClasses;
    float csum = 0.0f;
    for (int c = 0; c < kChromaClasses; ++c) {
        csum += row[c];
    }
    for (int c = 0; c < kChromaClasses; ++c) {
        f[121 + c] = row[c] / (csum + 1e-6f);
    }
    f[133] = lastChange_;

    const float d = w.fluxNormDecay;
    loudMean_     = d * loudMean_     + (1.0f - d) * loudness;
    loudMeanCorr_ = d * loudMeanCorr_ + (1.0f - d);
    f[134] = loudness - loudMean_ / std::max(loudMeanCorr_, 1e-6f);

    for (int i = 0; i < kBeatFeatures; ++i) {
        f[i] = finiteOr0(f[i]);
    }
}

void Network::runLearned() {
    const Weights& w = *w_;
    const int64_t  t = frame_;
    if (!useLearned_) {
        downHist_[static_cast<size_t>(t & kHistMask)] = 0.0f;
        drumsSmooth_ = 1.0f;   // No opinion: never lays out.
        return;
    }

    // The network always steps, so its history matches training exactly.
    beatNet_.step(feat_.data(), learnedOut_);

    const bool silent = silentCount_ >= w.silenceFrames;
    if (silent) {
        learnedOut_[kOutBeat]     = 0.0f;
        learnedOut_[kOutDownbeat] = 0.0f;
    }
    downHist_[static_cast<size_t>(t & kHistMask)] = learnedOut_[kOutDownbeat];

    const float dd = w.beatDrumsDecay;
    drumsSmooth_ = dd * drumsSmooth_ + (1.0f - dd) * (silent ? 0.0f : learnedOut_[kOutDrums]);
}

// ---------------------------------------------------------------------------
// Front end: DFT, transient split, mel, flux
// ---------------------------------------------------------------------------
void Network::runFrontEnd(const float* window, float& flux, float& lowFlux, float& loudness) {
    const Weights& w = *w_;
    const int nb = w.numBins;

    // DFT layer (evaluated by FFT), then power = re^2 + im^2.
    fft_.forward(window, w.dftWindow.data(), 0, nb, spec_.data());
    for (int k = 0; k < nb; ++k) {
        const float re = spec_[static_cast<size_t>(2 * k)];
        const float im = spec_[static_cast<size_t>(2 * k + 1)];
        power_[static_cast<size_t>(k)] = re * re + im * im;
    }

    // Sub-bass band (kick / bass-note attacks), log-compressed.
    const float lowM = kernels::dot(w.lowBand.data(), power_.data(), nb);
    const float low  = finiteOr0(std::log1p(w.logGain * std::max(lowM, 0.0f)));

    // Transient part of the spectrum.
    runTransientMask();

    // Plain and transient-only flux, blended.
    float loudPlain = 0.0f;
    const float fluxPlain = melFlux(power_,     melLog_,     melPrev_,     havePrev_, &loudPlain, riseX_.data());
    const float fluxPerc  = melFlux(percPower_, melLogPerc_, melPrevPerc_, havePrev_, nullptr,    riseP_.data());
    flux = (1.0f - w.percussiveMix) * fluxPlain + w.percussiveMix * fluxPerc;

    lowFlux   = havePrev_ ? std::max(0.0f, low - lowPrev_) : 0.0f;
    lowPrev_  = low;
    havePrev_ = true;
    loudness  = loudPlain;
}

/**
 * Causal harmonic/percussive split. Sustained = median of each bin over the
 * last hpssTimeFrames frames (this one included); transient = median across
 * hpssFreqBins neighbouring bins of this frame. Soft mask T^2/(T^2 + S^2).
 */
void Network::runTransientMask() {
    const Weights& w = *w_;
    const int nb = w.numBins;
    const int T  = w.hpssTimeFrames;

    // Push this frame's power into the time ring.
    std::copy(power_.begin(), power_.end(),
              powerHist_.begin() + static_cast<std::ptrdiff_t>(powerHistPos_) * nb);
    powerHistPos_  = (powerHistPos_ + 1) % T;
    powerHistFill_ = std::min(powerHistFill_ + 1, T);

    const int half = w.hpssFreqBins / 2;
    float*    med  = medScratch_.data();
    for (int k = 0; k < nb; ++k) {
        // Sustained estimate: median over time.
        for (int i = 0; i < powerHistFill_; ++i) {
            med[i] = powerHist_[static_cast<size_t>(i) * nb + k];
        }
        const float sustained = medianInPlace(med, powerHistFill_);

        // Transient estimate: median across frequency.
        const int lo = std::max(0, k - half);
        const int hi = std::min(nb - 1, k + half);
        for (int j = lo; j <= hi; ++j) {
            med[j - lo] = power_[static_cast<size_t>(j)];
        }
        const float transient = medianInPlace(med, hi - lo + 1);

        const float t2 = transient * transient;
        const float s2 = sustained * sustained;
        const float mask = t2 / (t2 + s2 + 1e-30f);
        percPower_[static_cast<size_t>(k)] = power_[static_cast<size_t>(k)] * finiteOr0(mask);
    }
}

/**
 * Mel bands + log, then the rectified rise against the neighbourhood max of
 * the previous frame (SuperFlux). Updates melPrev for the next call.
 */
float Network::melFlux(const std::vector<float>& power, std::vector<float>& melLog,
                       std::vector<float>& melPrev, bool havePrev, float* loudness, float* bandRise) {
    const Weights& w = *w_;
    const int nb = w.numBins;

    float loudSum = 0.0f;
    for (int b = 0; b < kMelBands; ++b) {
        const float m = kernels::dot(w.mel.row(b), power.data(), nb);
        const float v = finiteOr0(std::log1p(w.logGain * std::max(m, 0.0f)));
        melLog[static_cast<size_t>(b)] = v;
        loudSum += v;
    }

    float flux = 0.0f;
    if (bandRise) {
        std::fill(bandRise, bandRise + kMelBands, 0.0f);
    }
    if (havePrev) {
        const int r = std::max(0, w.fluxMaxRadius);
        for (int b = 0; b < kMelBands; ++b) {
            const int lo = std::max(0, b - r);
            const int hi = std::min(kMelBands - 1, b + r);
            float prevMax = melPrev[static_cast<size_t>(lo)];
            for (int j = lo + 1; j <= hi; ++j) {
                prevMax = std::max(prevMax, melPrev[static_cast<size_t>(j)]);
            }
            const float rise = melLog[static_cast<size_t>(b)] - prevMax;
            if (rise > 0.0f) {
                flux += rise * w.fluxReadout[static_cast<size_t>(b)];
                if (bandRise) {
                    bandRise[b] = rise;
                }
            }
        }
    }

    std::copy(melLog.begin(), melLog.end(), melPrev.begin());
    if (loudness) {
        *loudness = loudSum / kMelBands;
    }
    return flux;
}

// ---------------------------------------------------------------------------
// Chord channel
// ---------------------------------------------------------------------------
void Network::runChord(const float* longWindow) {
    const Weights& w = *w_;
    const int64_t  t = frame_;
    const int      nc = w.numChromaBins;

    // Long DFT layer (by FFT) -> power -> 12 pitch classes -> log.
    fftChroma_.forward(longWindow, w.chromaWindow.data(), w.chromaBinLo, nc, chromaSpec_.data());
    for (int i = 0; i < nc; ++i) {
        const float re = chromaSpec_[static_cast<size_t>(2 * i)];
        const float im = chromaSpec_[static_cast<size_t>(2 * i + 1)];
        chromaPower_[static_cast<size_t>(i)] = re * re + im * im;
    }
    float* row = chromaHist_.data() + static_cast<size_t>(t % kChromaHist) * kChromaClasses;
    for (int c = 0; c < kChromaClasses; ++c) {
        const float v = kernels::dot(w.chroma.row(c), chromaPower_.data(), nc);
        row[c] = finiteOr0(std::log1p(w.chromaLogGain * std::max(v, 0.0f)));
    }
    chromaFill_ = std::min(chromaFill_ + 1, kChromaHist);

    // -----------------------------------------------------------------------
    // Change = 1 - cos(mean of the last D frames, mean of the D before).
    // -----------------------------------------------------------------------
    const int D = w.harmonicSpanFrames;
    float h = 0.0f;
    if (chromaFill_ >= 2 * D) {
        float after[kChromaClasses]  = {};
        float before[kChromaClasses] = {};
        for (int j = 0; j < D; ++j) {
            const float* a = chromaHist_.data() + static_cast<size_t>((t - j) % kChromaHist) * kChromaClasses;
            const float* b = chromaHist_.data() + static_cast<size_t>((t - D - j) % kChromaHist) * kChromaClasses;
            for (int c = 0; c < kChromaClasses; ++c) {
                after[c]  += a[c];
                before[c] += b[c];
            }
        }
        double dotAB = 0.0, nA = 0.0, nB = 0.0;
        for (int c = 0; c < kChromaClasses; ++c) {
            dotAB += static_cast<double>(after[c]) * before[c];
            nA    += static_cast<double>(after[c]) * after[c];
            nB    += static_cast<double>(before[c]) * before[c];
        }
        float change = 0.0f;
        if (nA > 1e-9 && nB > 1e-9) {
            change = static_cast<float>(std::clamp(1.0 - dotAB / std::sqrt(nA * nB), 0.0, 1.0));
        }
        lastChange_ = change;

        // Normalise by its own running mean (bias corrected), rectify.
        const float d = w.harmonicNormDecay;
        harmMean_ = d * harmMean_ + (1.0f - d) * change;
        harmCorr_ = d * harmCorr_ + (1.0f - d);
        const bool silent = silentCount_ >= w.silenceFrames;
        if (!silent) {
            const float mean = harmMean_ / std::max(harmCorr_, 1e-6f);
            h = std::clamp(finiteOr0(change / (mean + w.harmonicNormEps) - w.harmonicBias), 0.0f, 20.0f);
        }
    }

    // Store the change at the moment it happened, harmonicOffsetFrames back.
    const int64_t pos = t - w.harmonicOffsetFrames;
    if (pos >= 0) {
        hX_[static_cast<size_t>(pos & kHistMask)] = h;
    }
    lastHarm_ = h;
}

// ---------------------------------------------------------------------------
// Silence gate
// ---------------------------------------------------------------------------
void Network::updateSilence(float loudness) {
    const Weights& w = *w_;

    silenceEnv_ = w.silenceDecay * silenceEnv_ + (1.0f - w.silenceDecay) * loudness;

    if (silenceEnv_ < w.silenceLevel) {
        // Saturate so the counter can never overflow during a long pause.
        if (silentCount_ < std::numeric_limits<int>::max() - 1) {
            ++silentCount_;
        }
        if (silentCount_ == w.resetSilenceFrames) {
            resetEstimators();   // The song is over: forget its tempo.
        }
    } else {
        silentCount_ = 0;
    }

    if (silentCount_ < w.silenceFrames) {
        ++soundFrames_;
    }
}

// ---------------------------------------------------------------------------
// Onset normalisation and smoothing
// ---------------------------------------------------------------------------
void Network::updateOnsets(float flux, float lowFlux) {
    const Weights& w = *w_;
    const int64_t t  = frame_;
    const bool silent = silentCount_ >= w.silenceFrames;

    // Running means with bias correction. Without the correction the mean
    // starts near zero and the first second of onsets would look enormous.
    const float nd = w.fluxNormDecay;
    fluxMean_     = nd * fluxMean_     + (1.0f - nd) * flux;
    lowFluxMean_  = nd * lowFluxMean_  + (1.0f - nd) * lowFlux;
    fluxNormCorr_ = nd * fluxNormCorr_ + (1.0f - nd);
    const float corr = std::max(fluxNormCorr_, 1e-6f);

    // Divide, subtract the "average moment", rectify, clamp.
    float x  = 0.0f;
    float lx = 0.0f;
    if (!silent) {
        x  = flux    / (fluxMean_    / corr + w.fluxNormEps) - w.onsetBias;
        lx = lowFlux / (lowFluxMean_ / corr + w.fluxNormEps) - w.onsetBias;
        x  = std::clamp(finiteOr0(x),  0.0f, 20.0f);
        lx = std::clamp(finiteOr0(lx), 0.0f, 20.0f);

        // Blend in the learned beat activation: it already knows the beat
        // from the "&", which the hand-written flux cannot tell apart.
        if (useLearned_) {
            const float xb = std::clamp(w.beatOnsetGain * learnedOut_[kOutBeat], 0.0f, 20.0f);
            x = (1.0f - w.beatOnsetMix) * x + w.beatOnsetMix * xb;
        }
    }

    xRaw_[static_cast<size_t>(t & kHistMask)] = x;
    lowX_[static_cast<size_t>(t & kHistMask)] = lx;

    // Centred 3-tap smoothing: the value for frame t-1 is ready now.
    if (t >= 1) {
        const size_t i1 = static_cast<size_t>((t - 1) & kHistMask);
        const size_t i2 = static_cast<size_t>((t - 2) & kHistMask);

        const float a = (t >= 2) ? xRaw_[i2] : 0.0f;
        xs_[i1] = w.smoothKernel[0] * a + w.smoothKernel[1] * xRaw_[i1] + w.smoothKernel[2] * x;

        const float la = (t >= 2) ? lowX_[i2] : 0.0f;
        lowXs_[i1] = w.smoothKernel[0] * la + w.smoothKernel[1] * lowX_[i1] + w.smoothKernel[2] * lx;
    }
}

// ---------------------------------------------------------------------------
// Pauses in the band
// ---------------------------------------------------------------------------
void Network::updatePause() {
    const Weights& w = *w_;

    const float recent  = loudMean_ / std::max(loudMeanCorr_, 1e-6f);
    const bool  stopped = silenceEnv_ < w.pauseLevel * recent;

    if (stopped) {
        if (quietFrames_ < std::numeric_limits<int>::max() - 1) {
            ++quietFrames_;
        }
        if (everLocked_ && static_cast<float>(quietFrames_) >= w.pauseBeats * curL_) {
            bandPaused_ = true;
        }
        return;
    }

    quietFrames_ = 0;
    if (!bandPaused_) {
        return;
    }
    bandPaused_ = false;

    // The band is back. Its next bars can be the same song or a new tempo, so
    // the clock keeps running while the settling window reopens: if the old
    // tempo still fits, nothing moves; if it does not, it can be replaced
    // quickly. Fading the memory stops the old tempo outvoting the new one.
    framesSinceLock_ = 0;
    switchCount_     = 0;
    snapCount_       = 0;
    for (double& r : R_) {
        r *= w.pauseForget;
    }
    for (double& r : Rh_) {
        r *= w.pauseForget;
    }
}

// ---------------------------------------------------------------------------
// Autocorrelation bank
// ---------------------------------------------------------------------------
void Network::updateAutocorrelation() {
    const int64_t t = frame_;
    if (t < 1) {
        return;
    }

    // Newest smoothed sample is frame t-1.
    const int64_t u = t - 1;
    const double  v = xs_[static_cast<size_t>(u & kHistMask)];
    const double  d = w_->acDecay;

    // R(lag) <- d * R(lag) + x[u] * x[u - lag]. One MAC per lag per frame.
    R_[0] = d * R_[0] + v * v;
    for (int L = 1; L <= kMaxLag; ++L) {
        const double prev = (u - L >= 0) ? xs_[static_cast<size_t>((u - L) & kHistMask)] : 0.0;
        R_[static_cast<size_t>(L)] = d * R_[static_cast<size_t>(L)] + v * prev;
    }

    // Same for the chord-change signal, whose newest value is older, with a
    // longer memory and lags long enough to see two-bar chord cycles.
    const int64_t uh = t - w_->harmonicOffsetFrames;
    if (uh >= 0) {
        const double vh = hX_[static_cast<size_t>(uh & kHistMask)];
        const double dh = w_->harmAcDecay;
        Rh_[0] = dh * Rh_[0] + vh * vh;
        for (int L = 1; L <= kMaxHarmLag; ++L) {
            const double prev = (uh - L >= 0) ? hX_[static_cast<size_t>((uh - L) & kHistMask)] : 0.0;
            Rh_[static_cast<size_t>(L)] = dh * Rh_[static_cast<size_t>(L)] + vh * prev;
        }
    }
}

// ---------------------------------------------------------------------------
// Tempo readout, lock and tracking
// ---------------------------------------------------------------------------
void Network::updateTempo(bool& tempoChanged) {
    const Weights& w = *w_;
    tempoChanged = false;

    // -----------------------------------------------------------------------
    // Remove the baseline: the mean of R over the musically relevant lags.
    // -----------------------------------------------------------------------
    double sum = 0.0;
    int    cnt = 0;
    for (int L = kMinPeriodFrames; L <= kMaxLag; ++L) {
        sum += R_[static_cast<size_t>(L)];
        ++cnt;
    }
    const double rBar = (cnt > 0) ? sum / cnt : 0.0;

    // Both autocorrelations in normalised units (0 = no repetition, 1 = all
    // of it), so the onset readout and the chord vote can be added.
    const double denomX = R_[0] - rBar;
    const double normX  = (denomX > 1e-9) ? 1.0 / denomX : 0.0;
    auto rt = [this, rBar, normX](int L) { return (R_[static_cast<size_t>(L)] - rBar) * normX; };

    double sumH = 0.0;
    int    cntH = 0;
    for (int L = 2 * kMinPeriodFrames; L <= kMaxHarmLag; ++L) {
        sumH += Rh_[static_cast<size_t>(L)];
        ++cntH;
    }
    const double rhBar  = (cntH > 0) ? sumH / cntH : 0.0;
    const double denomH = Rh_[0] - rhBar;
    const double normH  = (denomH > 1e-9) ? 1.0 / denomH : 0.0;

    // Chord vote for period L: mean normalised chord periodicity at 2L, 4L,
    // 8L (best of +-1 lag, each clamped to [-1, 1]).
    auto chordVote = [this, rhBar, normH](int L) {
        double acc = 0.0;
        int    n   = 0;
        for (int k : {2, 4, 8}) {
            const int c = k * L;
            if (c + 1 > kMaxHarmLag) {
                break;
            }
            const double peak = std::max({Rh_[static_cast<size_t>(c - 1)], Rh_[static_cast<size_t>(c)],
                                          Rh_[static_cast<size_t>(c + 1)]});
            acc += std::clamp((peak - rhBar) * normH, -1.0, 1.0);
            ++n;
        }
        return (n > 0) ? acc / n : 0.0;
    };

    // -----------------------------------------------------------------------
    // Salience(L) = prior(L) * (R(L) + a*R(2L) + b*R(L/2) + c*chordVote(L)).
    // -----------------------------------------------------------------------
    int   bestIdx = 0;
    float bestVal = -std::numeric_limits<float>::max();
    for (int i = 0; i < kNumPeriods; ++i) {
        const int    L    = kMinPeriodFrames + i;
        const double half = (L % 2 == 0) ? rt(L / 2) : 0.5 * (rt(L / 2) + rt(L / 2 + 1));
        const double dbl  = rt(2 * L);
        const double sal  = w.tempoPrior[static_cast<size_t>(i)] *
                            (rt(L) + w.harmonicDouble * dbl + w.harmonicHalf * half +
                             w.tempoHarmWeight * chordVote(L));
        S_[static_cast<size_t>(i)] = static_cast<float>(sal);
        if (S_[static_cast<size_t>(i)] > bestVal) {
            bestVal = S_[static_cast<size_t>(i)];
            bestIdx = i;
        }
    }

    double frac = 0.0;
    if (bestIdx > 0 && bestIdx < kNumPeriods - 1) {
        frac = parabolicOffset(S_[static_cast<size_t>(bestIdx - 1)],
                               S_[static_cast<size_t>(bestIdx)],
                               S_[static_cast<size_t>(bestIdx + 1)]);
    }
    lStar_       = static_cast<float>(kMinPeriodFrames + bestIdx + frac);
    periodicity_ = periodicityAt(lStar_, rBar);

    // -----------------------------------------------------------------------
    // Lock with hysteresis.
    // -----------------------------------------------------------------------
    firstLockEvent_ = false;
    const bool silent  = silentCount_ >= w.silenceFrames;
    const bool canLock = !silent && soundFrames_ >= w.minSoundFrames;
    if (canLock && periodicity_ >= w.lockOn) {
        lockCount_ = std::min(lockCount_ + 1, 2 * w.lockMinFrames);
    } else if (!canLock || periodicity_ < w.lockOff) {
        lockCount_ = std::max(lockCount_ - 1, 0);
    }

    if (!locked_ && lockCount_ >= w.lockMinFrames) {
        locked_ = true;
        if (!everLocked_) {
            // First lock of this song: adopt the estimate outright.
            everLocked_     = true;
            curL_           = refinePeriod(lStar_, rBar);
            switchCount_    = 0;
            tempoChanged    = true;
            firstLockEvent_ = true;
        }
    } else if (locked_ && lockCount_ == 0) {
        locked_ = false;
    }

    // -----------------------------------------------------------------------
    // Tracking. Before the first lock the period simply follows L*.
    // -----------------------------------------------------------------------
    if (!everLocked_) {
        curL_ = refinePeriod(lStar_, rBar);
    } else if (!tempoChanged) {
        // 1. Follow the nearest salience peak around the current period.
        //    The follow rate shrinks with leadership: a leading drummer only
        //    drifts slowly toward the band.
        const float lo  = curL_ * (1.0f - w.tempoTolerance);
        const float hi  = curL_ * (1.0f + w.tempoTolerance);
        const int   iLo = std::clamp(static_cast<int>(std::ceil(lo)) - kMinPeriodFrames, 0, kNumPeriods - 1);
        const int   iHi = std::clamp(static_cast<int>(std::floor(hi)) - kMinPeriodFrames, 0, kNumPeriods - 1);
        int iLoc = iLo;
        for (int i = iLo + 1; i <= iHi; ++i) {
            if (S_[static_cast<size_t>(i)] > S_[static_cast<size_t>(iLoc)]) {
                iLoc = i;
            }
        }

        // Only follow a real peak. If the window's best point sits on a
        // slope (its outer neighbour is higher), our tempo has no peak in
        // here, and following the slope would let the window creep, frame by
        // frame, all the way to some other tempo.
        const bool realPeak =
            (iLoc == 0 || S_[static_cast<size_t>(iLoc)] >= S_[static_cast<size_t>(iLoc - 1)]) &&
            (iLoc == kNumPeriods - 1 || S_[static_cast<size_t>(iLoc)] >= S_[static_cast<size_t>(iLoc + 1)]);
        if (realPeak) {
            const float localL = refinePeriod(static_cast<float>(kMinPeriodFrames + iLoc), rBar);
            if (std::fabs(localL - curL_) <= w.tempoTolerance * curL_) {
                curL_ += w.tempoFollow * couplingScale() * (localL - curL_);
            }
        }

        // 2. Switch only when the current tempo no longer fits the music and
        //    a clearly periodic challenger has won for a sustained stretch;
        //    or, during the settling window after the first lock, when the
        //    challenger is simply much stronger (an early count was wrong).
        const bool  elsewhere = std::fabs(lStar_ - curL_) > w.tempoTolerance * curL_;
        const float curRho    = periodicityAt(curL_, rBar);
        const float curSal    = std::max(S_[static_cast<size_t>(iLoc)], 0.0f);
        const bool  settling  = framesSinceLock_ < w.tempoSettleFrames;
        const bool  stopped   = curRho < w.lockOff && periodicity_ >= w.lockOn &&
                                bestVal > (1.0f + w.tempoSwitchMargin) * curSal + 1e-12f;
        const bool  recount   = settling && periodicity_ >= w.lockOn &&
                                bestVal > (1.0f + w.tempoSettleMargin) * curSal + 1e-12f;
        if (elsewhere && (stopped || recount)) {
            ++switchCount_;
        } else {
            switchCount_ = std::max(0, switchCount_ - 1);
        }
        const int needed = settling ? std::min(w.tempoSwitchFrames, w.tempoSettleSwitchFrames)
                                    : w.tempoSwitchFrames;
        if (switchCount_ >= needed) {
            curL_        = refinePeriod(lStar_, rBar);
            switchCount_ = 0;
            tempoChanged = true;
        }
    }

    curL_ = std::clamp(curL_, static_cast<float>(kMinPeriodFrames), static_cast<float>(kMaxPeriodFrames));

    // Clock for the settling window (saturating).
    if (everLocked_ && framesSinceLock_ < std::numeric_limits<int>::max()) {
        ++framesSinceLock_;
    }
}

/**
 * Normalised autocorrelation at period L (best of the three nearest lags).
 * ~0 when nothing repeats at L, approaching 1 when everything does.
 */
float Network::periodicityAt(float L, double rBar) const {
    const double denom = R_[0] - rBar;
    if (!(denom > 1e-9)) {
        return 0.0f;
    }
    const int lr = std::clamp(static_cast<int>(std::lround(L)), 2, kMaxLag - 1);
    const double peak = std::max({R_[static_cast<size_t>(lr - 1)],
                                  R_[static_cast<size_t>(lr)],
                                  R_[static_cast<size_t>(lr + 1)]});
    return static_cast<float>(std::clamp((peak - rBar) / denom, -1.0, 1.0));
}

/**
 * Precise period from the autocorrelation peaks at L, 2L, 3L... .
 *
 * The salience curve includes the tempo prior, whose slope drags a parabola
 * fitted to it toward 110 BPM. The raw autocorrelation has no such bias, and
 * its peak at multiple k pins the period down k times more precisely, so we
 * fit p_k = k * L by weighted least squares over every multiple in range.
 */
float Network::refinePeriod(float L, double rBar) const {
    if (!(L >= kMinPeriodFrames)) {
        return L;
    }
    double num = 0.0;
    double den = 0.0;
    for (int k = 1; k <= 6; ++k) {
        const double centre = k * static_cast<double>(L);
        const int    c      = static_cast<int>(std::lround(centre));
        if (c + 3 > kMaxLag) {
            break;
        }
        // Integer peak within +-2 lags of the expected multiple.
        int j = std::max(1, c - 2);
        for (int m = std::max(1, c - 1); m <= c + 2; ++m) {
            if (R_[static_cast<size_t>(m)] > R_[static_cast<size_t>(j)]) {
                j = m;
            }
        }
        const double strength = R_[static_cast<size_t>(j)] - rBar;
        if (strength <= 0.0) {
            continue;   // No periodicity at this multiple: it tells us nothing.
        }
        const double p = j + parabolicOffset(R_[static_cast<size_t>(j - 1)],
                                             R_[static_cast<size_t>(j)],
                                             R_[static_cast<size_t>(j + 1)]);
        num += strength * k * p;
        den += strength * k * k;
    }
    if (!(den > 0.0)) {
        return L;
    }
    const double refined = num / den;
    // Never let the refinement wander to a different peak.
    return (std::fabs(refined - L) <= 1.5) ? static_cast<float>(refined) : L;
}

// ---------------------------------------------------------------------------
// Template bank + phase-locked loop
// ---------------------------------------------------------------------------
float Network::sampleHistory(const std::vector<float>& hist, double pos, int64_t newest) const {
    // Linear interpolation into a history ring; nothing newer than `newest`.
    if (pos < 0.0 || newest < 0 || pos > static_cast<double>(newest) ||
        hist.size() != static_cast<size_t>(kHist)) {
        return 0.0f;
    }
    const int64_t i0 = static_cast<int64_t>(std::floor(pos));
    if (newest - i0 >= kHist - 1) {
        return 0.0f;   // Already overwritten in the ring.
    }
    if (i0 >= newest) {
        return hist[static_cast<size_t>(newest & kHistMask)];
    }
    const float fr = static_cast<float>(pos - static_cast<double>(i0));
    const float a  = hist[static_cast<size_t>(i0 & kHistMask)];
    const float b  = hist[static_cast<size_t>((i0 + 1) & kHistMask)];
    return a + fr * (b - a);
}

void Network::updatePhase(bool tempoChanged) {
    const Weights& w = *w_;
    const int64_t  t = frame_;
    rebasedFlag_ = false;

    // -----------------------------------------------------------------------
    // Free-running oscillator: advance one frame's worth of beat.
    // -----------------------------------------------------------------------
    const double P = std::max(static_cast<double>(curL_), static_cast<double>(kMinPeriodFrames));
    B_ += 1.0 / P;

    // -----------------------------------------------------------------------
    // Template bank: one unit per phase offset phi = 1..maxPhi frames ago.
    // -----------------------------------------------------------------------
    const int64_t newest  = t - 1;                        // newest smoothed onset
    const int64_t newestH = t - w.harmonicOffsetFrames;   // newest chord change
    const int     maxPhi  = std::min(static_cast<int>(std::ceil(P)), kMaxPeriodFrames + 1);

    const double avail = static_cast<double>(t - 2 - maxPhi);
    int K = 0;
    if (avail >= P) {
        K = std::min(w.templateBeats, 1 + static_cast<int>(std::floor(avail / P)));
        while (K > 1 && maxPhi + (K - 1) * P + 2.0 >= kHist - 2) {
            --K;
        }
    }

    if (K >= 2) {
        double sum  = 0.0;
        float  best = -1.0f;
        int    bp   = 1;
        for (int phi = 1; phi <= maxPhi; ++phi) {
            float sc = 0.0f;
            float wk = 1.0f;
            for (int k = 0; k < K; ++k) {
                const double pos = static_cast<double>(t - phi) - k * P;
                float v = sampleHistory(xs_, pos, newest) +
                          w.templateLowWeight * sampleHistory(lowXs_, pos, newest);
                // Chord changes arrive harmonicOffsetFrames late, so the most
                // recent beat (k = 0) is skipped for every phase alike.
                if (k >= 1) {
                    v += w.templateHarmWeight * sampleHistory(hX_, pos, newestH);
                }
                sc += wk * v;
                wk *= w.templateDecay;
            }
            phaseScore_[static_cast<size_t>(phi)] = sc;
            sum += sc;
            if (sc > best) {
                best = sc;
                bp   = phi;
            }
        }
        const double mean     = sum / maxPhi;
        const double contrast = (mean > 1e-9) ? best / mean : 0.0;

        // Circular neighbours within 1..maxPhi, and parabolic refinement.
        auto wrapPhi = [maxPhi](int phi) { return ((phi - 1) % maxPhi + maxPhi) % maxPhi + 1; };
        auto refined = [&](int phi) {
            return phi + parabolicOffset(phaseScore_[static_cast<size_t>(wrapPhi(phi - 1))],
                                         phaseScore_[static_cast<size_t>(phi)],
                                         phaseScore_[static_cast<size_t>(wrapPhi(phi + 1))]);
        };

        // -------------------------------------------------------------------
        // Our current phase, as "frames since the last beat".
        // -------------------------------------------------------------------
        const double frac   = B_ - std::floor(B_);
        double       phiCur = frac * P;
        if (phiCur < 1.0) {
            phiCur += P;
        }
        const int centre = std::clamp(static_cast<int>(std::lround(phiCur)), 1, maxPhi);

        // Strongest unit near our current phase.
        const int halfWin = std::max(1, static_cast<int>(std::lround(w.phaseSnapError * P)));
        int   loc  = centre;
        float sLoc = phaseScore_[static_cast<size_t>(centre)];
        for (int d = -halfWin; d <= halfWin; ++d) {
            const int phi = wrapPhi(centre + d);
            if (phaseScore_[static_cast<size_t>(phi)] > sLoc) {
                sLoc = phaseScore_[static_cast<size_t>(phi)];
                loc  = phi;
            }
        }

        const double eLoc  = wrapHalf(refined(loc) / P - frac);
        const double eGlob = wrapHalf(refined(bp) / P - frac);
        phaseError_ = static_cast<float>(eLoc);

        // -------------------------------------------------------------------
        // Phase-locked loop update.
        // -------------------------------------------------------------------
        if (tempoChanged) {
            B_ += eGlob;             // New tempo: jump straight onto the best phase.
            rebasedFlag_ = true;
            snapCount_   = 0;
        } else if (contrast >= w.templateMinContrast) {
            const bool elsewhere = std::fabs(wrapHalf(eGlob - eLoc)) > w.phaseSnapError;
            if (elsewhere && best > (1.0f + w.phaseSnapMargin) * std::max(sLoc, 1e-9f)) {
                ++snapCount_;
            } else {
                snapCount_ = std::max(0, snapCount_ - 1);
            }

            // Stubborn once settled, as a drummer holds the beat through a
            // syncopated riff; quick only while nothing is known yet.
            const bool settling   = framesSinceLock_ < w.tempoSettleFrames;
            const int  snapNeeded = settling ? std::min(w.phaseSettleSnapFrames, w.phaseSnapFrames)
                                             : w.phaseSnapFrames;
            if (snapCount_ >= snapNeeded) {
                B_ += eGlob;         // A different phase kept winning: move to it.
                rebasedFlag_ = true;
                snapCount_   = 0;
            } else {
                B_ += w.pllPhaseGain * couplingScale() * eLoc;
            }
        }
    } else if (tempoChanged) {
        rebasedFlag_ = true;
    }

    // -----------------------------------------------------------------------
    // Beat boundary detection (monotonic: a backward snap never repeats one).
    // -----------------------------------------------------------------------
    const int64_t n = static_cast<int64_t>(std::floor(B_));
    if (n > lastBeatInt_) {
        lastBeatInt_      = n;
        state_.newBeat    = true;
        state_.beatNumber = n;
        if (numPending_ < kMaxPending) {
            pending_[numPending_++] = PendingBeat{n, t};
        }
    }
}

// ---------------------------------------------------------------------------
// Downbeat slots
// ---------------------------------------------------------------------------
void Network::updateDownbeat() {
    const Weights& w = *w_;
    const int64_t  t = frame_;

    // A beat can be scored once its accent window, including the late chord
    // channel, has fully arrived.
    const int64_t wait = w.accentPost + w.harmonicOffsetFrames;

    int  write    = 0;
    bool measured = false;
    for (int i = 0; i < numPending_; ++i) {
        const PendingBeat pb = pending_[i];
        if (t < pb.frame + wait) {
            pending_[write++] = pb;   // Not ready yet: keep it.
            continue;
        }

        float accent = 0.0f;
        for (int64_t j = pb.frame - w.accentPre; j <= pb.frame + w.accentPost; ++j) {
            if (j < 0 || t - j >= kHist - 1) {
                continue;
            }
            const size_t ij = static_cast<size_t>(j & kHistMask);
            const float a = w.accentLowWeight    * lowX_[ij] +
                            w.accentAllWeight    * xRaw_[ij] +
                            w.accentHarmWeight   * hX_[ij] +
                            w.beatDownbeatWeight * downHist_[ij];
            accent = std::max(accent, a);
        }

        const int slot = mod4(pb.beat);
        slotAccent_[slot] = w.downbeatDecay * slotAccent_[slot] + accent;
        if (slotUpdates_[slot] < std::numeric_limits<int>::max()) {
            ++slotUpdates_[slot];
        }
        measured = true;
    }
    numPending_ = write;

    // Decisions only happen on beats, never on frames in between.
    if (!measured) {
        return;
    }

    // Evidence gate: every slot must have been heard enough times.
    for (int i = 0; i < kBeatsPerBar; ++i) {
        if (slotUpdates_[i] < w.downbeatMinUpdates) {
            return;
        }
    }

    int best = 0;
    for (int i = 1; i < kBeatsPerBar; ++i) {
        if (slotAccent_[i] > slotAccent_[best]) {
            best = i;
        }
    }

    // The first informed choice replaces the default guess immediately.
    if (!slotDecided_) {
        downbeatSlot_  = best;
        slotDecided_   = true;
        slotChallenge_ = 0;
        return;
    }

    // Afterwards a challenger must win clearly on consecutive beats.
    if (best != downbeatSlot_ &&
        slotAccent_[best] > (1.0f + w.downbeatSwitchMargin) * slotAccent_[downbeatSlot_]) {
        if (++slotChallenge_ >= w.downbeatSwitchBeats) {
            downbeatSlot_  = best;
            slotChallenge_ = 0;
        }
    } else {
        slotChallenge_ = 0;
    }
}

void Network::clearDownbeat() {
    numPending_ = 0;
    std::fill(std::begin(slotAccent_), std::end(slotAccent_), 0.0f);
    std::fill(std::begin(slotUpdates_), std::end(slotUpdates_), 0);
    slotChallenge_ = 0;
    slotDecided_   = false;
}

// ---------------------------------------------------------------------------
// Intensity
// ---------------------------------------------------------------------------
void Network::updateIntensity(float loudness, float flux) {
    const Weights& w = *w_;
    const int64_t  t = frame_;
    const bool silent = silentCount_ >= w.silenceFrames;
    sectionFlag_ = false;

    if (!silent) {
        // Fast envelopes start at the first sound, not at zero.
        if (!fastInit_) {
            loudFast_ = loudness;
            densFast_ = flux;
            fastInit_ = true;
        }
        loudFast_ = w.loudFastDecay * loudFast_ + (1.0f - w.loudFastDecay) * loudness;
        densFast_ = w.densFastDecay * densFast_ + (1.0f - w.densFastDecay) * flux;
        const float logDens = std::log(std::max(densFast_, 0.0f) + 1e-4f);

        if (!rangeInit_) {
            // Give the envelopes one second to settle before trusting a range.
            if (soundFrames_ >= 100) {
                loudMin_ = loudMax_ = loudFast_;
                densMin_ = densMax_ = logDens;
                rangeInit_ = true;
            }
        } else {
            // Min/max envelopes that slowly close in on the present.
            loudMax_ = std::max(loudFast_, loudMax_ - w.loudLeak);
            loudMin_ = std::min(loudFast_, loudMin_ + w.loudLeak);
            densMax_ = std::max(logDens,   densMax_ - w.densLeak);
            densMin_ = std::min(logDens,   densMin_ + w.densLeak);

            const float loudSpan = std::max(loudMax_ - loudMin_, w.loudMinSpan);
            const float densSpan = std::max(densMax_ - densMin_, w.densMinSpan);
            const float nl = std::clamp(0.5f + (loudFast_ - 0.5f * (loudMin_ + loudMax_)) / loudSpan, 0.0f, 1.0f);
            const float nd = std::clamp(0.5f + (logDens   - 0.5f * (densMin_ + densMax_)) / densSpan, 0.0f, 1.0f);

            const float raw = w.loudMix * nl + (1.0f - w.loudMix) * nd;
            level_ = w.levelSmoothDecay * level_ + (1.0f - w.levelSmoothDecay) * raw;
        }
    }

    level_ = std::clamp(finiteOr0(level_), 0.0f, 1.0f);
    levelHist_[static_cast<size_t>(t & kHistMask)] = level_;

    // Section change: a clear rise in intensity over the lookback.
    if (sectionCooldown_ > 0) {
        --sectionCooldown_;
    } else if (locked_ && t >= w.sectionLookbackFrames) {
        const float before = levelHist_[static_cast<size_t>((t - w.sectionLookbackFrames) & kHistMask)];
        if (level_ - before > w.sectionJump) {
            sectionFlag_     = true;
            sectionCooldown_ = w.sectionCooldownFrames;
        }
    }
}

} // namespace drumalong
