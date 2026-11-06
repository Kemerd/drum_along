#pragma once
/**
 * @file Network.h
 * @brief Stateful, causal forward pass of the hand-programmed drummer network.
 *
 * One call to step() consumes one 10 ms analysis frame and updates every
 * recurrent state (autocorrelation bank, phase-locked loop, downbeat slots,
 * intensity envelopes, chord history). It never looks at future audio.
 *
 * Real-time contract: prepare() allocates; reset() and step() never allocate,
 * lock or throw, so step() is safe to call on the audio thread.
 */

#include "BeatNet.h"
#include "Config.h"
#include "Fft.h"
#include "Weights.h"

#include <cstdint>
#include <vector>

namespace drumalong {

/**
 * @brief Everything the network knows after one frame.
 *
 * Read by the engine to schedule hits, and by the offline tool for debugging.
 */
struct FrameState {
    int64_t frameIndex   = 0;     ///< 0-based frame counter.
    double  centerSample = 0.0;   ///< Absolute input sample this frame represents.

    // ---- Onsets ------------------------------------------------------------
    float flux       = 0.0f;      ///< Raw blended spectral flux.
    float onset      = 0.0f;      ///< Rectified, normalised broadband onset.
    float lowOnset   = 0.0f;      ///< Rectified, normalised sub-bass onset.
    float harmonic   = 0.0f;      ///< Newest chord-change value (placed harmonicOffsetFrames back).
    float loudness   = 0.0f;      ///< Mean log-mel level.

    // ---- Tempo -------------------------------------------------------------
    float periodicity  = 0.0f;    ///< Normalised autocorrelation at the chosen lag.
    float periodFrames = 0.0f;    ///< Tracked beat period, in frames.
    float tempoBpm     = 0.0f;    ///< Same, in BPM.
    bool  locked       = false;   ///< Tempo is trusted enough to play along.

    // ---- Phase -------------------------------------------------------------
    double  beatPos     = 0.0;    ///< Continuous beat counter at centerSample.
    float   phaseError  = 0.0f;   ///< Error toward the local template peak, in beats.
    bool    rebased     = false;  ///< Phase or tempo jumped this frame.
    bool    newBeat     = false;  ///< A beat boundary was crossed this frame.
    int64_t beatNumber  = 0;      ///< Index of the last crossed beat.

    // ---- Bar ---------------------------------------------------------------
    int downbeatSlot = 0;         ///< Beats with (beatNumber mod 4) == slot are "1".

    // ---- Intensity ---------------------------------------------------------
    float level         = 0.5f;   ///< 0 = softest moment of the song, 1 = loudest.
    bool  sectionChange = false;  ///< Level jumped: crash and restart the phrase.
    bool  silent        = true;   ///< Input is effectively silent.

    // ---- Learned layer (all 0 / 1 when no beatnet.json is loaded) ----------
    float beatAct     = 0.0f;     ///< P(beat at this frame).
    float downbeatAct = 0.0f;     ///< P(downbeat at this frame).
    float drumsActive = 1.0f;     ///< Smoothed P(a drummer would be playing).

    // ---- Training export ------------------------------------------------------
    const float* features    = nullptr;  ///< kBeatFeatures values, valid until the next step().
    int          numFeatures = 0;
};

/**
 * @brief The drummer's ears and sense of time.
 */
class Network {
public:
    Network() = default;

    /**
     * @brief Bind to a realized weight set and allocate all state.
     * @param weights Must outlive this object. Null is rejected.
     * @return false if the weights are missing or inconsistent.
     */
    bool prepare(const Weights* weights);

    /// Forget everything: back to the state of a freshly prepared network.
    void reset();

    /**
     * @brief Run one frame.
     * @param window inputWindowSamples() mono samples, oldest first. The
     *               onset DFT reads the newest windowSamples of them; the
     *               chord DFT reads all of them.
     * @return The updated state. Valid until the next call.
     */
    const FrameState& step(const float* window);

    /// Samples of history step() needs per call.
    int inputWindowSamples() const { return w_ ? w_->inputWindowSamples() : 0; }

    /// Frame hop, in samples.
    int hopSamples() const { return w_ ? w_->hopSamples : 0; }

    /// True once prepare() succeeded.
    bool isPrepared() const { return w_ != nullptr; }

private:
    // Size of every history ring buffer, in frames (10.24 s).
    static constexpr int kHist     = 1024;
    static constexpr int kHistMask = kHist - 1;
    static constexpr int kMaxLag   = 2 * kMaxPeriodFrames;

    // ---- Per-layer helpers ---------------------------------------------------
    void   runFrontEnd(const float* window, float& flux, float& lowFlux, float& loudness);
    void   runTransientMask();
    float  melFlux(const std::vector<float>& power, std::vector<float>& melLog,
                   std::vector<float>& melPrev, bool havePrev, float* loudness, float* bandRise);
    void   runChord(const float* longWindow);
    void   assembleFeatures(float lowFlux, float loudness);
    void   runLearned();
    void   updateSilence(float loudness);
    void   updateOnsets(float flux, float lowFlux);
    void   updatePause();                           ///< Notices the band stopping and coming back.
    void   updateAutocorrelation();
    void   updateTempo(bool& tempoChanged);
    void   updatePhase(bool tempoChanged);
    void   updateDownbeat();
    void   clearDownbeat();
    void   updateIntensity(float loudness, float flux);
    void   resetEstimators();
    float  sampleHistory(const std::vector<float>& hist, double pos, int64_t newest) const;
    float  periodicityAt(float L, double rBar) const;
    float  refinePeriod(float L, double rBar) const;
    float  couplingScale() const;

    const Weights* w_ = nullptr;

    // ---- Onset front end scratch (sized in prepare) --------------------------
    Fft fft_;                        ///< Evaluates the onset DFT layer.
    Fft fftChroma_;                  ///< Evaluates the chord DFT layer.
    std::vector<float> spec_;        ///< 2*numBins: interleaved re/im.
    std::vector<float> power_;       ///< numBins.
    std::vector<float> percPower_;   ///< numBins: transient part of power_.
    std::vector<float> powerHist_;   ///< hpssTimeFrames x numBins ring.
    std::vector<float> medScratch_;  ///< Median workspace.
    std::vector<float> melLog_, melPrev_;          ///< Plain path, kMelBands.
    std::vector<float> melLogPerc_, melPrevPerc_;  ///< Transient path, kMelBands.
    float lowPrev_  = 0.0f;
    bool  havePrev_ = false;
    int   powerHistPos_  = 0;
    int   powerHistFill_ = 0;

    // ---- Chord channel ---------------------------------------------------------
    std::vector<float> chromaSpec_;   ///< 2*numChromaBins.
    std::vector<float> chromaPower_;  ///< numChromaBins.
    std::vector<float> chromaHist_;   ///< kChromaHist x kChromaClasses ring.
    int   chromaFill_ = 0;
    float harmMean_   = 0.0f;
    float harmCorr_   = 0.0f;
    float lastHarm_   = 0.0f;
    float lastChange_ = 0.0f;         ///< Raw (un-normalised) chord change, for the features.

    // ---- Learned layer ------------------------------------------------------------
    BeatNet            beatNet_;
    bool               useLearned_ = false;
    std::vector<float> feat_;         ///< kBeatFeatures.
    std::vector<float> riseX_;        ///< kMelBands: plain per-band rise.
    std::vector<float> riseP_;        ///< kMelBands: transient per-band rise.
    float              learnedOut_[kNumBeatNetOutputs] = {};
    float              loudMean_     = 0.0f;  ///< 3 s running loudness, for the features.
    float              loudMeanCorr_ = 0.0f;
    float              drumsSmooth_  = 1.0f;
    std::vector<float> downHist_;     ///< Learned downbeat activation history.

    // ---- Onset normalisation -----------------------------------------------
    float fluxMean_     = 0.0f;
    float lowFluxMean_  = 0.0f;
    float fluxNormCorr_ = 0.0f;   ///< 1 - decay^n: bias correction for both means.

    // ---- Histories (ring buffers, indexed by frame & kHistMask) -------------
    std::vector<float> xRaw_;      ///< Rectified normalised onset.
    std::vector<float> xs_;        ///< Smoothed onset (valid up to frame t-1).
    std::vector<float> lowX_;      ///< Rectified normalised sub-bass onset.
    std::vector<float> lowXs_;     ///< Smoothed sub-bass onset (valid up to frame t-1).
    std::vector<float> hX_;        ///< Chord change (valid up to t - harmonicOffsetFrames).
    std::vector<float> levelHist_; ///< Intensity level.

    // ---- Tempo ---------------------------------------------------------------
    std::vector<double> R_;          ///< Leaky autocorrelation, lags 0..kMaxLag.
    std::vector<double> Rh_;         ///< Chord-change autocorrelation, lags 0..kMaxHarmLag.
    std::vector<float>  S_;          ///< Salience per candidate period.
    std::vector<float>  phaseScore_; ///< Template score per phase offset.
    float   curL_         = 0.0f;
    float   lStar_        = 0.0f;
    float   periodicity_  = 0.0f;
    int     switchCount_  = 0;
    int     lockCount_    = 0;
    bool    locked_       = false;
    bool    everLocked_   = false;
    int64_t soundFrames_  = 0;
    int     framesSinceLock_ = 0;  ///< Frames since the first lock (settling window).

    // ---- Phase-locked loop ---------------------------------------------------
    double  B_            = 0.0;   ///< Continuous beat position.
    int64_t lastBeatInt_  = -1;
    int     snapCount_    = 0;
    int     quietFrames_  = 0;     ///< Consecutive frames the band has been stopped.
    bool    bandPaused_   = false; ///< The band has stopped and has not come back yet.
    float   phaseError_   = 0.0f;
    bool    rebasedFlag_  = false;

    // ---- Downbeat ------------------------------------------------------------
    struct PendingBeat {
        int64_t beat  = 0;
        int64_t frame = 0;
    };
    static constexpr int kMaxPending = 8;
    PendingBeat pending_[kMaxPending] = {};
    int         numPending_ = 0;
    float       slotAccent_[kBeatsPerBar]  = {};
    int         slotUpdates_[kBeatsPerBar] = {};
    int         slotChallenge_  = 0;
    bool        slotDecided_    = false;
    int         downbeatSlot_   = 0;
    bool        firstLockEvent_ = false;

    // ---- Intensity -------------------------------------------------------------
    float loudFast_  = 0.0f;
    float densFast_  = 0.0f;
    float loudMin_   = 0.0f, loudMax_ = 0.0f;
    float densMin_   = 0.0f, densMax_ = 0.0f;
    bool  fastInit_  = false;
    bool  rangeInit_ = false;
    float level_     = 0.5f;
    int   sectionCooldown_ = 0;
    bool  sectionFlag_     = false;

    // ---- Silence -------------------------------------------------------------
    float silenceEnv_  = 0.0f;
    int   silentCount_ = 0;

    int64_t    frame_ = 0;
    FrameState state_;
};

} // namespace drumalong
