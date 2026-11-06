/**
 * @file DrumAlongEngine.cpp
 * @brief Streaming wrapper around the network: input ring, frame clock,
 *        groove head and the sample-accurate MIDI scheduler.
 */

#include "drumalong/DrumAlongEngine.h"
#include "drumalong/WeightProgrammer.h"
#include "drumalong/WeightsIO.h"

#include <algorithm>
#include <cmath>

namespace drumalong {

namespace {

/// Floor division that stays correct for negative numerators.
inline int64_t floorDiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) {
        --q;
    }
    return q;
}

/// Non-negative modulo 4.
inline int mod4(int64_t n) {
    const int r = static_cast<int>(n % 4);
    return (r < 0) ? r + 4 : r;
}

} // namespace

// ===========================================================================
// Lifecycle
// ===========================================================================

bool DrumAlongEngine::prepare(double sampleRate, int maxBlockSize,
                              const std::string& weightsPath, std::string* error) {
    prepared_ = false;
    (void)maxBlockSize;   // Blocks of any size are handled sample by sample.

    // -----------------------------------------------------------------------
    // Weights: hand-reasoned defaults, optionally overlaid by the editable
    // weights file, then realized for this sample rate.
    // -----------------------------------------------------------------------
    Weights fresh;
    programDefaults(fresh);
    if (!weightsPath.empty() && !loadWeightsFile(weightsPath, fresh, error)) {
        return false;
    }
    if (!realize(fresh, sampleRate, error)) {
        return false;
    }
    weights_ = std::move(fresh);
    styles_.clear();          // Styles are realized per sample rate: re-add after prepare().
    activeStyle_.store(0, std::memory_order_relaxed);

    if (!net_.prepare(&weights_)) {
        if (error) *error = "network rejected the realized weights";
        return false;
    }

    // Allocate everything process() will touch. The ring holds the longest
    // window any layer reads (the chord channel's), written twice.
    windowLen_ = weights_.inputWindowSamples();
    ring_.assign(static_cast<size_t>(2 * windowLen_), 0.0f);
    queue_.assign(kQueueCapacity, Scheduled{});
    noteLenSamples_ = std::max<int64_t>(1, std::llround(weights_.noteLengthSeconds * sampleRate));
    offsetSamples_  = static_cast<double>(offsetMs_) * 0.001 * sampleRate;

    prepared_ = true;
    reset();
    return true;
}

void DrumAlongEngine::reset() {
    if (!prepared_) {
        return;
    }
    net_.reset();

    std::fill(ring_.begin(), ring_.end(), 0.0f);
    ringPos_    = 0;
    hopCounter_ = 0;
    samplesIn_  = 0;
    queueCount_ = 0;

    planning_        = false;
    armed_           = false;
    playing_         = false;
    lastPlannedStep_ = 0;
    barsPlayed_      = 0;
    phraseOffset_    = 0;
    realignPhrase_   = false;
    pendingCrash_    = false;
    barHadFill_      = false;
    grooveLatched_   = false;
    grooveLevel_     = 0.5f;
    satOut_          = false;
    fillVariant_     = -1;
    fillStart_       = 0;
    rng_             = 0x9E3779B9u;
    stats_           = EngineStats{};
}

int DrumAlongEngine::addStyle(const std::string& path, std::string* error) {
    if (!prepared_) {
        if (error) *error = "prepare() the engine before adding styles";
        return -1;
    }
    // A style is a full weights file (usually "extends" the base); load and
    // validate it at the engine's rate, then keep only for its groove head.
    Weights style;
    programDefaults(style);
    if (!loadWeightsFile(path, style, error) || !realize(style, weights_.sampleRate, error)) {
        return -1;
    }
    styles_.push_back(std::move(style));
    return static_cast<int>(styles_.size());
}

void DrumAlongEngine::setStyle(int index) {
    activeStyle_.store(std::clamp(index, 0, static_cast<int>(styles_.size())), std::memory_order_relaxed);
}

void DrumAlongEngine::setOutputOffsetMs(float ms) {
    offsetMs_ = std::isfinite(ms) ? std::clamp(ms, -100.0f, 100.0f) : 0.0f;
    if (prepared_) {
        offsetSamples_ = static_cast<double>(offsetMs_) * 0.001 * weights_.sampleRate;
    }
}

// ===========================================================================
// Audio thread
// ===========================================================================

int DrumAlongEngine::process(const float* const* channels, int numChannels, int numSamples,
                             MidiEvent* out, int maxOut) {
    if (!prepared_ || numSamples <= 0) {
        return 0;
    }

    const int     N          = windowLen_;
    const int     hop        = weights_.hopSamples;
    const int64_t blockStart = samplesIn_;

    // -----------------------------------------------------------------------
    // 1. Feed samples one at a time; every hop samples, run one frame.
    //    Frames fire mid-block exactly where they would in a stream, so the
    //    result never depends on the host's block size.
    // -----------------------------------------------------------------------
    for (int i = 0; i < numSamples; ++i) {
        // Mono mix of whichever channels exist.
        float mono = 0.0f;
        int   used = 0;
        if (channels && numChannels > 0) {
            for (int c = 0; c < numChannels; ++c) {
                if (channels[c]) {
                    mono += channels[c][i];
                    ++used;
                }
            }
        }
        mono = (used > 0) ? mono / static_cast<float>(used) : 0.0f;
        if (!std::isfinite(mono)) {
            mono = 0.0f;   // A NaN from upstream must not poison the network.
        }
        mono = std::clamp(mono, -4.0f, 4.0f);

        // Double-write: window = ring_[ringPos_ .. ringPos_ + N), oldest first.
        ring_[static_cast<size_t>(ringPos_)]     = mono;
        ring_[static_cast<size_t>(ringPos_ + N)] = mono;
        ringPos_ = (ringPos_ + 1 == N) ? 0 : ringPos_ + 1;
        ++samplesIn_;

        if (++hopCounter_ >= hop) {
            hopCounter_ = 0;
            runFrame();
        }
    }

    // -----------------------------------------------------------------------
    // 2. Emit every queued event that falls inside this block.
    // -----------------------------------------------------------------------
    const int64_t blockEnd = blockStart + numSamples;
    int written  = 0;
    int consumed = 0;
    while (consumed < queueCount_ && queue_[static_cast<size_t>(consumed)].time < blockEnd) {
        if (!out || written >= maxOut) {
            if (!out || maxOut <= 0) {
                // Nowhere to write at all: discard instead of piling up.
                ++consumed;
                ++stats_.droppedEvents;
                continue;
            }
            break;   // Output full: the rest go out next block, slightly late.
        }
        const Scheduled& ev = queue_[static_cast<size_t>(consumed)];
        MidiEvent& m   = out[written++];
        m.sampleOffset = static_cast<int32_t>(std::clamp<int64_t>(ev.time - blockStart, 0, numSamples - 1));
        m.note         = ev.note;
        m.velocity     = ev.velocity;
        m.noteOn       = ev.noteOn;
        ++consumed;
    }

    // Slide the remaining events to the front.
    if (consumed > 0) {
        std::move(queue_.begin() + consumed, queue_.begin() + queueCount_, queue_.begin());
        queueCount_ -= consumed;
    }
    return written;
}

// ---------------------------------------------------------------------------
// One analysis frame: run the network, then plan any steps due before the
// next frame.
// ---------------------------------------------------------------------------
void DrumAlongEngine::runFrame() {
    const FrameState& s = net_.step(ring_.data() + ringPos_);
    if (observer_) {
        observer_->onFrame(s);
    }

    // A jump in intensity: crash on the next 1 and start a new phrase there.
    if (s.sectionChange) {
        pendingCrash_  = true;
        realignPhrase_ = true;
    }

    // -----------------------------------------------------------------------
    // No trusted tempo or no sound: stop, and re-arm when both come back.
    // -----------------------------------------------------------------------
    if (!s.locked || s.silent || !(s.periodFrames > 0.0f)) {
        planning_ = false;
        armed_    = false;
        playing_  = false;
        return;
    }

    const double B              = s.beatPos;
    const double c              = s.centerSample;
    const double hop            = weights_.hopSamples;
    const double samplesPerBeat = static_cast<double>(s.periodFrames) * hop;
    const double stepSamples    = samplesPerBeat / kStepsPerBeat;
    const int64_t gridStep      = static_cast<int64_t>(std::floor(B * kStepsPerBeat));

    if (!planning_) {
        // Fresh lock: start counting from the current grid position and
        // wait for the next downbeat before playing anything.
        planning_        = true;
        armed_           = true;
        lastPlannedStep_ = gridStep;
    } else if (s.rebased) {
        // Grid jumped forward: skip steps now in the past. Jumped backward:
        // keep the counter so no step is ever played twice.
        lastPlannedStep_ = std::max(lastPlannedStep_, gridStep);
    }

    // -----------------------------------------------------------------------
    // Plan every step whose output time lands before the next frame.
    // T(g) = centre + (g/4 - B) * samplesPerBeat, extrapolated from the
    // newest phase estimate (about 20 ms of extrapolation, never more).
    // -----------------------------------------------------------------------
    const double now     = static_cast<double>(samplesIn_);
    const double horizon = now + hop - offsetSamples_;

    for (int guard = 0; guard < 64; ++guard) {
        const int64_t g = lastPlannedStep_ + 1;

        // Swing: the "&" moves toward the last triplet of the beat (up to
        // +1/6 beat), the "e" and "a" toward theirs (up to +1/12 beat).
        // Delays only, and always smaller than the gap to the next step, so
        // steps stay in order.
        const int stepInBeat = static_cast<int>(((g % kStepsPerBeat) + kStepsPerBeat) % kStepsPerBeat);
        const Weights& gw = grooveWeights();
        double swing = 0.0;
        if (stepInBeat == 2) {
            swing = gw.swing8 / 6.0;
        } else if (stepInBeat == 1 || stepInBeat == 3) {
            swing = gw.swing16 / 12.0;
        }
        const double T = c + (static_cast<double>(g) / kStepsPerBeat + swing - B) * samplesPerBeat;
        if (T >= horizon) {
            break;
        }
        lastPlannedStep_ = g;

        double tOut = T + offsetSamples_;
        if (tOut < now) {
            // Already behind us (phase correction or offset). Play it now if
            // it is only a little late, otherwise drop it: a hit half a
            // sixteenth late sounds worse than no hit.
            if (now - tOut > weights_.lateToleranceSteps * stepSamples) {
                ++stats_.skippedSteps;
                continue;
            }
            ++stats_.lateEvents;
            tOut = now;
        }
        planStep(g, tOut, s);
    }
}

// ---------------------------------------------------------------------------
// The groove head: which drums fire on this sixteenth, and how hard.
// ---------------------------------------------------------------------------
void DrumAlongEngine::planStep(int64_t globalStep, double timeSample, const FrameState& s) {
    // The groove head reads the active style; listening used the base weights.
    const Weights& w = grooveWeights();

    // -----------------------------------------------------------------------
    // Where are we in the bar and in the phrase?
    // -----------------------------------------------------------------------
    const int64_t beat       = floorDiv(globalStep, kStepsPerBeat);
    const int     stepInBeat = static_cast<int>(globalStep - beat * kStepsPerBeat);
    const int64_t relBeat    = beat - s.downbeatSlot;
    const int     beatInBar  = mod4(relBeat);
    const int64_t barIndex   = floorDiv(relBeat, kBeatsPerBar);
    const int     stepInBar  = beatInBar * kStepsPerBeat + stepInBeat;

    bool crashFromFill = false;
    if (stepInBar == 0) {
        // Start of a bar: start playing if armed, count bars, maybe realign.
        if (armed_ && !playing_) {
            playing_    = true;
            armed_      = false;
            barsPlayed_ = 0;
        } else if (playing_) {
            ++barsPlayed_;
        }
        if (realignPhrase_) {
            phraseOffset_  = barIndex;
            realignPhrase_ = false;
        }
        crashFromFill = barHadFill_;
        barHadFill_   = false;
    }

    // Position in the 8-bar phrase (non-negative modulo).
    int phraseBar = static_cast<int>((barIndex - phraseOffset_) % kBarsPerPhrase);
    if (phraseBar < 0) {
        phraseBar += kBarsPerPhrase;
    }

    // -----------------------------------------------------------------------
    // Decisions a drummer makes on a bar line, never in the middle of one.
    // -----------------------------------------------------------------------
    if (stepInBar == 0) {
        // The pattern: only on the phrase grid, and only for a real change of
        // intensity. A section change lands here too, since it resets phraseBar.
        const bool phraseLine = phraseBar % w.grooveHoldBars == 0;
        if (!grooveLatched_ || (phraseLine && std::fabs(s.level - grooveLevel_) >= w.grooveHoldChange)) {
            grooveLevel_   = s.level;
            grooveLatched_ = true;
        }

        // Sitting out, with a gap between the two thresholds so a value near
        // one of them cannot switch every bar. Coming back gets a crash.
        const bool wasOut = satOut_;
        satOut_ = w.beatLayOutBelow > 0.0f &&
                  s.drumsActive < (satOut_ ? w.beatLayInAbove : w.beatLayOutBelow);
        if (wasOut && !satOut_) {
            pendingCrash_ = true;
        }
    }

    // -----------------------------------------------------------------------
    // The fill for the last bar of the phrase is chosen once, on its 1, from
    // the intensity at that moment, so it cannot change pattern halfway.
    // -----------------------------------------------------------------------
    if (stepInBar == 0) {
        fillVariant_ = -1;
        const float lv = s.level;
        if (playing_ && phraseBar == kBarsPerPhrase - 1 &&
            barsPlayed_ + 1 >= w.fillMinBarsPlayed && lv >= w.fillMinLevel) {
            if (lv < w.fillSnareLevel) {
                fillVariant_ = kFillPickup;
                fillStart_   = w.fillStartShort;
            } else if (lv < w.fillTomLevel) {
                fillVariant_ = kFillSnare;
                fillStart_   = w.fillStartShort;
            } else {
                // Alternate the two tom fills by phrase so they never repeat back to back.
                const int64_t phrase = floorDiv(barIndex - phraseOffset_, kBarsPerPhrase);
                fillVariant_ = (phrase % 2 == 0) ? kFillSnareToms : kFillToms;
                fillStart_   = (lv >= w.fillLevelLong) ? w.fillStartLong : w.fillStartShort;
            }
        }
    }

    StepInfo info;
    info.globalStep = globalStep;
    info.timeSample = timeSample;
    info.stepInBar  = stepInBar;
    info.phraseBar  = phraseBar;
    info.playing    = playing_;
    info.level      = s.level;
    info.tempoBpm   = s.tempoBpm;

    // Not started yet, the song is quiet enough that this drummer lays out, or
    // the learned layer hears a passage where a drummer would sit out.
    const bool layOut = !enabled_.load(std::memory_order_relaxed) ||
                        s.level < w.layOutLevel || satOut_;
    if (!playing_ || layOut) {
        info.playing = playing_ && !layOut;
        if (observer_) {
            observer_->onStep(info);
        }
        return;
    }

    // -----------------------------------------------------------------------
    // Fill and crash decisions.
    // -----------------------------------------------------------------------
    const float level  = s.level;
    const bool  inFill = fillVariant_ >= 0 && phraseBar == kBarsPerPhrase - 1 &&
                         stepInBar >= fillStart_;
    const bool  crash  = stepInBar == 0 &&
                         (pendingCrash_ || (crashFromFill && level >= w.crashLevel));
    if (crash) {
        pendingCrash_ = false;
    }

    // -----------------------------------------------------------------------
    // Fire the hits.
    // -----------------------------------------------------------------------
    const int64_t t0   = std::llround(timeSample);
    int           hits = 0;
    auto hit = [&](int drum, float baseVelocity) {
        float v = baseVelocity + w.velLevelGain * (level - 0.5f) + w.velHumanize * nextJitter();
        v = std::clamp(v, 1.0f, 127.0f);
        const uint8_t note = kDrumNotes[drum];
        schedule(t0, note, static_cast<uint8_t>(std::lround(v)), true);
        schedule(t0 + noteLenSamples_, note, 0, false);
        ++hits;
    };

    if (inFill) {
        const Matrix& table = w.fillVelocity[static_cast<size_t>(fillVariant_)];
        for (int d = 0; d < kNumDrums; ++d) {
            const float v = table.at(stepInBar, d);
            if (v > 0.0f) {
                hit(d, v);
            }
        }
        // A pickup is too small to deserve a crash after it.
        barHadFill_ = fillVariant_ != kFillPickup;
    } else {
        for (int d = 0; d < kNumDrums; ++d) {
            // A crash replaces the cymbal that would have played on the 1.
            if (crash && (d == kClosedHat || d == kOpenHat || d == kRide)) {
                continue;
            }
            const bool on = grooveLevel_ >= w.hitOnLevel.at(stepInBar, d) &&
                            grooveLevel_ <  w.hitOffLevel.at(stepInBar, d);
            if (on) {
                hit(d, w.velocity.at(stepInBar, d));
            }
        }
        if (crash) {
            hit(kCrash, w.crashVelocity);
        }
    }

    info.fill  = inFill;
    info.crash = crash;
    info.hits  = hits;
    if (observer_) {
        observer_->onStep(info);
    }
}

// ---------------------------------------------------------------------------
// Sorted insert into the fixed-capacity queue.
// ---------------------------------------------------------------------------
void DrumAlongEngine::schedule(int64_t time, uint8_t note, uint8_t velocity, bool noteOn) {
    if (queueCount_ >= kQueueCapacity) {
        ++stats_.droppedEvents;
        return;
    }

    // Walk back from the end; events usually arrive in time order, so this
    // is almost always zero or one comparison.
    int i = queueCount_;
    while (i > 0 && queue_[static_cast<size_t>(i - 1)].time > time) {
        queue_[static_cast<size_t>(i)] = queue_[static_cast<size_t>(i - 1)];
        --i;
    }
    queue_[static_cast<size_t>(i)] = Scheduled{time, note, velocity, noteOn};
    ++queueCount_;

    if (noteOn) {
        ++stats_.notesOn;
    }
}

float DrumAlongEngine::nextJitter() {
    // xorshift32: fast, deterministic, allocation-free.
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return static_cast<float>(rng_) * (2.0f / 4294967295.0f) - 1.0f;
}

} // namespace drumalong
