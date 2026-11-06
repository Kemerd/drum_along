#pragma once
/**
 * @file DrumAlongEngine.h
 * @brief Audio in, MIDI out. The piece a VST3 wrapper (or the offline tool)
 *        talks to.
 *
 * ============================================================================
 *  Timing model
 * ============================================================================
 *  The network only ever sees the past, so the drummer never *reacts* to a
 *  beat; it *predicts* the next sixteenth from the tracked tempo and phase and
 *  schedules the hit at an exact future sample. Analysis cost therefore adds
 *  no latency: hits are planned roughly one frame (10 ms) ahead and emitted
 *  inside whichever host block contains their timestamp, with sample-accurate
 *  offsets. A user offset (in ms) shifts every hit, to compensate for the
 *  drum synth and the audio interface.
 *
 *  Real-time contract: prepare() allocates. reset(), process() and the
 *  setters never allocate, lock or throw.
 * ============================================================================
 */

#include "Config.h"
#include "MidiEvent.h"
#include "Network.h"
#include "Weights.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace drumalong {

/**
 * @brief Everything the groove head decided for one sixteenth-note step.
 */
struct StepInfo {
    int64_t globalStep = 0;     ///< Running sixteenth counter.
    double  timeSample = 0.0;   ///< Absolute output sample of the step.
    int     stepInBar  = 0;     ///< 0..15.
    int     phraseBar  = 0;     ///< 0..3 within the 4-bar phrase.
    bool    playing    = false; ///< False while waiting for the first downbeat.
    bool    fill       = false;
    bool    crash      = false;
    float   level      = 0.0f;
    float   tempoBpm   = 0.0f;
    int     hits       = 0;     ///< Notes triggered on this step.
};

/**
 * @brief Optional debug tap. Called synchronously from process(), so a real
 *        plugin should leave it unset. The offline tool uses it to log CSVs.
 */
class IEngineObserver {
public:
    virtual ~IEngineObserver() = default;
    virtual void onFrame(const FrameState& /*state*/) {}
    virtual void onStep(const StepInfo& /*step*/) {}
};

/**
 * @brief Running counters, readable at any time.
 */
struct EngineStats {
    int64_t notesOn      = 0;   ///< Note-ons scheduled.
    int64_t lateEvents   = 0;   ///< Steps emitted after their ideal time.
    int64_t skippedSteps = 0;   ///< Steps too late to play at all.
    int64_t droppedEvents = 0;  ///< Queue overflow (should stay 0).
};

/**
 * @brief The drummer.
 */
class DrumAlongEngine {
public:
    DrumAlongEngine() = default;

    // The network keeps a pointer to weights_, so the engine must stay put.
    DrumAlongEngine(const DrumAlongEngine&)            = delete;
    DrumAlongEngine& operator=(const DrumAlongEngine&) = delete;

    /**
     * @brief Build the weights for @p sampleRate and allocate all state.
     *
     * Starts from the hand-reasoned defaults, overlays @p weightsPath when it
     * is not empty, then realizes the front-end tables for the rate.
     *
     * @param sampleRate   Host rate in Hz.
     * @param maxBlockSize Largest block process() will ever see (informational).
     * @param weightsPath  Optional weights file (JSON). Empty = defaults.
     * @param error        Optional: receives the reason on failure.
     * @return false on an unsupported rate or an invalid weights file.
     */
    bool prepare(double sampleRate, int maxBlockSize,
                 const std::string& weightsPath = std::string(), std::string* error = nullptr);

    /// Forget the song: tempo, phase, bar, intensity, queued notes.
    void reset();

    /**
     * @brief Consume one host block of audio and produce the MIDI for it.
     *
     * @param channels    numChannels pointers to numSamples floats. Null
     *                    channel pointers are skipped.
     * @param numChannels Channel count (mixed to mono).
     * @param numSamples  Block length.
     * @param out         Destination for events, sorted by sampleOffset.
     * @param maxOut      Capacity of @p out. Events that do not fit stay
     *                    queued and come out (late) in the next block.
     * @return Number of events written.
     */
    int process(const float* const* channels, int numChannels, int numSamples,
                MidiEvent* out, int maxOut);

    /// Shift every hit; negative = earlier. Clamped to +-100 ms.
    void setOutputOffsetMs(float ms);

    /**
     * @brief Load an extra groove style (a weights file, usually one of
     *        weights/styles/*.json) for instant switching later.
     *
     * Only the groove head (patterns, fills, swing, dynamics) of a style is
     * used; listening and timing always run on the prepare() weights, so
     * switching style never disturbs the beat. Call after prepare(), never
     * on the audio thread.
     *
     * @return The style's index (1, 2 ...; 0 is the prepare() groove), or -1
     *         with @p error set.
     */
    int addStyle(const std::string& path, std::string* error = nullptr);

    /// Number of selectable styles, including the base groove (index 0).
    int numStyles() const { return 1 + static_cast<int>(styles_.size()); }

    /// Select a style; takes effect on the next sixteenth. Audio-thread safe.
    void setStyle(int index);

    /// Mute or unmute the drums. Listening continues, so it re-enters in time.
    void setPlaying(bool on) { enabled_.store(on, std::memory_order_relaxed); }

    /// Debug tap; pass nullptr to remove. Not for use inside a real host.
    void setObserver(IEngineObserver* observer) { observer_ = observer; }

    /// The plugin adds no latency: hits are predicted, not delayed.
    int latencySamples() const { return 0; }

    const EngineStats& stats() const { return stats_; }
    const Weights&     weights() const { return weights_; }
    bool               isPrepared() const { return prepared_; }

private:
    /// A note waiting for its timestamp.
    struct Scheduled {
        int64_t time     = 0;
        uint8_t note     = 0;
        uint8_t velocity = 0;
        bool    noteOn   = true;
    };

    void runFrame();
    void planStep(int64_t globalStep, double timeSample, const FrameState& s);

    /// Weights the groove head reads: the active style, or the base weights.
    const Weights& grooveWeights() const {
        const int i = activeStyle_.load(std::memory_order_relaxed);
        return (i > 0 && i <= static_cast<int>(styles_.size())) ? styles_[static_cast<size_t>(i - 1)] : weights_;
    }
    void schedule(int64_t time, uint8_t note, uint8_t velocity, bool noteOn);
    float nextJitter();   // Uniform in [-1, 1].

    Weights              weights_;
    std::vector<Weights> styles_;             ///< Extra groove styles (index 1..).
    std::atomic<int>     activeStyle_{0};
    std::atomic<bool>    enabled_{true};
    Network              net_;
    bool             prepared_ = false;
    IEngineObserver* observer_ = nullptr;
    EngineStats      stats_;

    // ---- Input ring: every sample is written twice so the newest window is
    //      always one contiguous, oldest-first span of N floats. ------------
    std::vector<float> ring_;
    int     windowLen_  = 0;   ///< Samples per network input window.
    int     ringPos_    = 0;
    int     hopCounter_ = 0;
    int64_t samplesIn_  = 0;   ///< Absolute index of the next input sample.

    // ---- Output queue, sorted by time. Fixed capacity. ---------------------
    static constexpr int kQueueCapacity = 1024;
    std::vector<Scheduled> queue_;
    int queueCount_ = 0;

    // ---- Planning state ------------------------------------------------------
    bool    planning_         = false;  ///< Tempo locked and sound present.
    bool    armed_            = false;  ///< Will start at the next downbeat.
    bool    playing_          = false;
    int64_t lastPlannedStep_  = 0;
    int64_t barsPlayed_       = 0;
    int64_t phraseOffset_     = 0;
    bool    realignPhrase_    = false;
    bool    pendingCrash_     = false;
    bool    barHadFill_       = false;
    bool    grooveLatched_    = false;  ///< grooveLevel_ holds a value for this run.
    float   grooveLevel_      = 0.5f;   ///< Intensity that picks the pattern, held between phrase lines.
    bool    satOut_           = false;  ///< Sitting out a passage with no drummer; changes on bar lines.
    int     fillVariant_      = -1;     ///< Fill chosen for the current bar, -1 = none.
    int     fillStart_        = 0;      ///< Step where that fill begins.
    float   offsetMs_         = 0.0f;
    double  offsetSamples_    = 0.0;
    int64_t noteLenSamples_   = 0;
    uint32_t rng_             = 0x9E3779B9u;
};

} // namespace drumalong
