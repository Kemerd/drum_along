#pragma once
/**
 * @file Config.h
 * @brief Fixed *architecture* of the drummer network: layer sizes and the
 *        time base. Everything that is a learned-or-written number lives in
 *        Weights (and in the editable weights file) instead.
 *
 * Everything is sized in *time*, not samples: the frame rate is 100 Hz at any
 * host sample rate, so the recurrent layers (autocorrelation, phase-locked
 * loop, leaky integrators) see identical timing at 44.1 kHz and 96 kHz. Only
 * the front-end tables depend on the sample rate, and those are regenerated
 * from the weights spec whenever the host rate changes.
 */

#include <cstdint>

namespace drumalong {

// ---------------------------------------------------------------------------
// Time base
// ---------------------------------------------------------------------------

/// Analysis frames per second. 100 Hz = one frame every 10 ms.
constexpr int kFrameRate = 100;

// ---------------------------------------------------------------------------
// Spectral front-end sizes
// ---------------------------------------------------------------------------

/// Number of mel bands in the onset front end.
constexpr int kMelBands = 40;

/// Pitch classes in the chord channel (C, C#, D ... B).
constexpr int kChromaClasses = 12;

/// Frames of chroma history kept for the chord-change unit (0.64 s).
constexpr int kChromaHist = 64;

// ---------------------------------------------------------------------------
// Tempo bank
// ---------------------------------------------------------------------------

/// Beat periods (in frames) covered by the tempo readout.
/// 29 frames = 206.9 BPM, 120 frames = 50 BPM.
constexpr int kMinPeriodFrames = 29;
constexpr int kMaxPeriodFrames = 120;
constexpr int kNumPeriods = kMaxPeriodFrames - kMinPeriodFrames + 1;

/// Longest lag of the chord-change autocorrelation: 8 beats at the slowest
/// tempo, so two-bar chord cycles are visible to the tempo vote.
constexpr int kMaxHarmLag = 8 * kMaxPeriodFrames;

// ---------------------------------------------------------------------------
// Rhythm structure
// ---------------------------------------------------------------------------

/// We assume 4/4. Triple meters are an honest known limitation of v1.
constexpr int kBeatsPerBar   = 4;
constexpr int kStepsPerBeat  = 4;     ///< Sixteenth-note grid.
constexpr int kStepsPerBar   = kBeatsPerBar * kStepsPerBeat;
constexpr int kBarsPerPhrase = 8;     ///< Fills land at the end of an 8-bar phrase.

// ---------------------------------------------------------------------------
// Drum vocabulary (General MIDI percussion map)
// ---------------------------------------------------------------------------

/// Index into the groove tables. Order matters: the tables are written
/// against this ordering, and the weights file uses the names below.
enum DrumIndex : int {
    kKick = 0,
    kSnare,
    kClosedHat,
    kOpenHat,
    kCrash,
    kRide,
    kTomHi,
    kTomMid,
    kTomLo,
    kPedalHat,
    kNumDrums
};

/// GM note numbers, indexed by DrumIndex.
constexpr uint8_t kDrumNotes[kNumDrums] = {
    36,  // kick
    38,  // snare
    42,  // closed hat
    46,  // open hat
    49,  // crash
    51,  // ride
    50,  // tom hi
    47,  // tom mid
    45,  // tom lo
    44   // hi-hat foot (pedal "chick")
};

/// Names used for each drum in the weights file.
constexpr const char* kDrumNames[kNumDrums] = {
    "kick", "snare", "closedHat", "openHat", "crash", "ride", "tomHi", "tomMid", "tomLo", "pedalHat"
};

} // namespace drumalong
