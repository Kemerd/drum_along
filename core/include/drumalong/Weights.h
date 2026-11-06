#pragma once
/**
 * @file Weights.h
 * @brief Every tensor and constant in the drummer network.
 *
 * Nothing in here is trained. WeightProgrammer writes the defaults by
 * reasoning about what each layer must compute, and the weights file
 * (weights/drummer.json) can override any of them without recompiling.
 *
 * Two kinds of fields live here:
 *
 *   SPEC      Every musical decision: groove tables, fill tables, tempo
 *             prior, time constants, channel weights, front-end ranges.
 *             Sample-rate independent. This is what the weights file holds.
 *
 *   REALIZED  The big front-end tables (DFT rows, mel and chroma filters).
 *             Pure math derived from the SPEC for the host sample rate, the
 *             way a decoder derives its cosine tables. Never stored in the
 *             file: they would be ~1M numbers that nobody should hand-edit.
 *
 * Layout convention: matrices are row-major, rows x cols, applied as
 * y = W * x with x of length cols.
 */

#include "BeatNet.h"
#include "Config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace drumalong {

/**
 * @brief Small row-major dense matrix.
 *
 * Deliberately trivial so the compiler can see straight through it and
 * vectorise the inner dot-product loop.
 */
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;

    /// Allocate and zero-fill. Never called on the audio thread.
    void resize(int r, int c) {
        rows = (r > 0) ? r : 0;
        cols = (c > 0) ? c : 0;
        data.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
    }

    float&       at(int r, int c)       { return data[static_cast<size_t>(r) * cols + c]; }
    const float& at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
    const float* row(int r) const       { return data.data() + static_cast<size_t>(r) * cols; }
    bool         empty() const          { return data.empty(); }
};

/// Fill patterns, ordered by how much they say. Intensity picks the tier.
enum FillVariant : int {
    kFillPickup = 0,   ///< Two snare notes into the 1. Barely a fill.
    kFillSnare,        ///< Snare build on beat 4.
    kFillSnareToms,    ///< Snare into toms.
    kFillToms,         ///< Down the toms.
    kNumFillVariants
};

/// Names used for each fill in the weights file.
constexpr const char* kFillNames[kNumFillVariants] = {"pickup", "snare", "snareToms", "toms"};

/**
 * @brief The complete parameter set of the drummer network.
 */
struct Weights {
    // =======================================================================
    // SPEC: front end
    // =======================================================================
    float windowSeconds = 0.0f;   ///< Onset DFT window (rounded to a power of two).
    float melMinHz      = 0.0f;   ///< Lowest mel edge.
    float melMaxHz      = 0.0f;   ///< Highest mel edge (capped below Nyquist).
    float lowBandMinHz  = 0.0f;   ///< Sub-bass band used for kick/bass accents.
    float lowBandMaxHz  = 0.0f;
    float logGain       = 1.0f;   ///< m' = log(1 + logGain * m).

    // =======================================================================
    // SPEC: transient channel (causal harmonic/percussive split)
    // =======================================================================
    int   hpssTimeFrames = 0;     ///< Median over this many past frames = sustained part.
    int   hpssFreqBins   = 0;     ///< Median over this many neighbouring bins = transient part.
    float percussiveMix  = 0.0f;  ///< 0 = plain flux, 1 = transient-only flux.

    // =======================================================================
    // SPEC: onset flux and normalisation
    // =======================================================================
    std::vector<float> fluxReadout;         ///< kMelBands, weight per band.
    int   fluxMaxRadius = 1;                ///< Neighbour max on the previous frame.
    float fluxNormDecay = 0.0f;             ///< Leaky-mean coefficient of the running flux level.
    float fluxNormEps   = 0.0f;             ///< Floor on the running mean.
    float onsetBias     = 1.0f;             ///< x = max(0, flux/mean - bias).
    float smoothKernel[3] = {0.25f, 0.5f, 0.25f};

    // =======================================================================
    // SPEC: chord-change channel
    // =======================================================================
    float chromaWindowSeconds = 0.0f;  ///< Long DFT window for pitch resolution.
    float chromaMinHz         = 0.0f;  ///< Pitch range read by the chroma filters.
    float chromaMaxHz         = 0.0f;
    float chromaSigma         = 0.0f;  ///< Pitch-class spread, in semitones.
    float chromaLogGain       = 1.0f;
    int   harmonicSpanFrames  = 0;     ///< Each side of the before/after comparison.
    float harmonicNormDecay   = 0.0f;  ///< Running mean of the change signal.
    float harmonicNormEps     = 0.0f;
    float harmonicBias        = 1.0f;  ///< h = max(0, change/mean - bias).

    // =======================================================================
    // SPEC: tempo
    // =======================================================================
    float              acDecay           = 0.0f;  ///< Per-frame leak of every lag accumulator.
    std::vector<float> tempoPrior;                ///< kNumPeriods multiplicative prior.
    float              harmonicDouble    = 0.0f;  ///< Weight of R(2L) in the salience of L.
    float              harmonicHalf      = 0.0f;  ///< Weight of R(L/2) in the salience of L.
    float              tempoHarmWeight   = 0.0f;  ///< Weight of chord-change periodicity at 2L/4L/8L.
    float              harmAcDecay       = 0.0f;  ///< Per-frame leak of the chord-change autocorrelation.
    float              tempoFollow       = 0.0f;  ///< Per-frame pull toward the local peak.
    float              tempoTolerance    = 0.0f;  ///< Relative window that counts as "same tempo".
    float              tempoSwitchMargin = 0.0f;  ///< Challenger must be this much more salient...
    int                tempoSwitchFrames = 0;     ///< ...for this many frames.
    int                tempoSettleFrames = 0;     ///< After the first lock, a re-count window...
    float              tempoSettleMargin = 0.0f;  ///< ...where a challenger this much stronger...
    int                tempoSettleSwitchFrames = 0; ///< ...for this long replaces an early count.
    float              pauseBeats  = 0.0f;  ///< Beats below pauseLevel that count as the band stopping.
    float              pauseLevel  = 0.0f;  ///< Loudness, relative to the last 3 s, that counts as stopped.
    float              pauseForget = 0.0f;  ///< Share of the tempo memory kept when the band comes back.
    float              lockOn            = 0.0f;  ///< Periodicity needed to count toward lock.
    float              lockOff           = 0.0f;  ///< Periodicity below which lock erodes.
    int                lockMinFrames     = 0;     ///< Sustained frames before declaring lock.
    int                minSoundFrames    = 0;     ///< Audio needed before any lock is allowed.

    // =======================================================================
    // SPEC: beat phase (template bank + phase-locked loop)
    // =======================================================================
    float templateLowWeight   = 0.0f;  ///< Sub-bass onsets in each template unit.
    float templateHarmWeight  = 0.0f;  ///< Chord changes in each template unit.
    int   templateBeats       = 0;     ///< How many past beats the template spans.
    float templateDecay       = 0.0f;  ///< Weight falloff per beat into the past.
    float templateMinContrast = 0.0f;  ///< Peak/mean template score needed to steer.
    float pllPhaseGain        = 0.0f;  ///< Fraction of phase error corrected per frame...
    float leadership          = 0.0f;  ///< ...scaled by (1 - 0.8*leadership); also scales tempo follow.
                                       ///< 0 = chases the band, 1 = holds time like a metronome.
    float phaseSnapError      = 0.0f;  ///< Half-width (beats) of the "same phase" window.
    float phaseSnapMargin     = 0.0f;  ///< A different phase must beat ours by this ratio...
    int   phaseSnapFrames     = 0;     ///< ...for this many frames before we jump to it.
    int   phaseSettleSnapFrames = 0;   ///< The same hold during a settling window.
    float timingOffsetMs      = 0.0f;  ///< Onsets register this late in the analysis; hits move earlier by it.

    // =======================================================================
    // SPEC: downbeat
    // =======================================================================
    int   accentPre            = 0;    ///< Frames before the beat searched for accent.
    int   accentPost           = 0;    ///< Frames after the beat searched for accent.
    float accentLowWeight      = 0.0f; ///< Sub-bass onsets (kick / bass note).
    float accentAllWeight      = 0.0f; ///< Broadband onsets.
    float accentHarmWeight     = 0.0f; ///< Chord changes (the strongest bar cue).
    float downbeatDecay        = 0.0f; ///< Leak per update of each slot.
    float downbeatSwitchMargin = 0.0f; ///< New slot must beat current by this ratio...
    int   downbeatSwitchBeats  = 0;    ///< ...on this many consecutive beats.
    int   downbeatMinUpdates   = 0;    ///< Accents every slot needs before any decision.

    // =======================================================================
    // SPEC: intensity
    // =======================================================================
    float loudFastDecay    = 0.0f;
    float densFastDecay    = 0.0f;
    float levelSmoothDecay = 0.0f;
    float loudMinSpan      = 0.0f;  ///< Range narrower than this is treated as "steady".
    float densMinSpan      = 0.0f;
    float loudLeak         = 0.0f;  ///< Per-frame closing of the min/max envelope.
    float densLeak         = 0.0f;
    float loudMix          = 0.0f;  ///< level = loudMix*loud + (1-loudMix)*density.
    int   sectionLookbackFrames = 0;///< Compare level against this far back...
    float sectionJump      = 0.0f;  ///< ...and flag a section change above this rise.
    int   sectionCooldownFrames = 0;

    float silenceDecay       = 0.0f; ///< Envelope of loudness used for the gate.
    float silenceLevel       = 0.0f; ///< Below this the input counts as silent.
    int   silenceFrames      = 0;    ///< Silent frames before the drummer stops.
    int   resetSilenceFrames = 0;    ///< Silent frames before all estimators reset.

    // =======================================================================
    // SPEC: groove head. Each (step, drum) cell fires when
    // onLevel <= level < offLevel. Velocities are readout weights.
    // =======================================================================
    Matrix hitOnLevel;     ///< kStepsPerBar x kNumDrums.
    Matrix hitOffLevel;    ///< kStepsPerBar x kNumDrums.
    Matrix velocity;       ///< kStepsPerBar x kNumDrums, at level 0.5.
    std::vector<Matrix> fillVelocity; ///< kNumFillVariants tables, 0 = no hit.

    float swing8            = 0.0f;  ///< 0 = straight eighths, 1 = full triplet swing on every "&".
    float swing16           = 0.0f;  ///< Same for the "e" and "a" sixteenths.
    float velLevelGain      = 0.0f;  ///< Velocity added per unit of (level - 0.5).
    float velHumanize       = 0.0f;  ///< +- uniform velocity jitter.
    float layOutLevel       = 0.0f;  ///< Below this intensity the drummer lays out.
    int   grooveHoldBars    = 0;     ///< The pattern may only change on this bar grid.
    float grooveHoldChange  = 0.0f;  ///< Intensity change needed to move to another pattern.
    float fillMinLevel      = 0.0f;  ///< No fills below this intensity.
    float fillSnareLevel    = 0.0f;  ///< Below this, only a pickup; above, a snare build.
    float fillTomLevel      = 0.0f;  ///< At or above this, fills may use toms.
    float fillLevelLong     = 0.0f;  ///< At or above this, tom fills start on beat 3.
    int   fillStartShort    = 0;     ///< Step where a short fill begins (beat 4).
    int   fillStartLong     = 0;     ///< Step where a long fill begins (beat 3).
    int   fillMinBarsPlayed = 0;     ///< Bars of groove required before the first fill.
    float crashLevel        = 0.0f;  ///< Post-fill crash only at or above this level.
    float crashVelocity     = 0.0f;
    float noteLengthSeconds = 0.0f;  ///< Gap between note-on and note-off.
    float lateToleranceSteps = 0.0f; ///< A step later than this is dropped, not played.

    // =======================================================================
    // SPEC: how the learned layer is used (the learned numbers themselves are
    // in a separate file, named here, so hand-written and trained weights
    // never mix in one place)
    // =======================================================================
    std::string beatNetFile;              ///< Relative to the weights file. Empty = no learned layer.
    float beatOnsetMix       = 0.0f;      ///< 0 = hand-written onsets only, 1 = learned beat activation only.
    float beatOnsetGain      = 0.0f;      ///< Scales the 0..1 activation into onset units.
    float beatDownbeatWeight = 0.0f;      ///< Learned downbeat activation in the accent slots.
    float beatLayOutBelow    = 0.0f;      ///< Lay out while "drummer would play" is below this (0 = never).
    float beatLayInAbove     = 0.0f;      ///< Once out, come back when "drummer would play" passes this.
    float beatDrumsDecay     = 0.0f;      ///< Smoothing of the "drummer would play" activation.

    /// Trained parameters, LOADED from beatNetFile (never hand-written).
    BeatNetWeights beatNet;

    // =======================================================================
    // REALIZED for the host sample rate (never stored in the file)
    // =======================================================================
    double sampleRate    = 0.0;
    int    hopSamples    = 0;   ///< sampleRate / 100.
    int    windowSamples = 0;   ///< Onset DFT length N (power of two).
    int    numBins       = 0;   ///< Onset DFT bins kept (DC .. top mel edge).
    /// The DFT layer is the linear map X_k = sum_n window[n] x[n] e^{-2 pi i k n / N}.
    /// Only its window (Hann * scale) is stored; an FFT evaluates the map.
    std::vector<float> dftWindow;  ///< N.
    Matrix mel;                 ///< kMelBands x numBins, rows sum to 1.
    std::vector<float> lowBand; ///< numBins, averages the sub-bass bins.

    int    chromaWindowSamples  = 0;  ///< Chord DFT length (power of two).
    int    chromaBinLo          = 0;  ///< First DFT bin read by the chroma filters.
    int    numChromaBins        = 0;
    std::vector<float> chromaWindow;  ///< chromaWindowSamples: the chord DFT layer's window.
    Matrix chroma;                    ///< kChromaClasses x numChromaBins.
    int    harmonicOffsetFrames = 0;  ///< How far back a chord change is placed in time.

    /// Samples of history the network needs per frame (the longer window).
    int inputWindowSamples() const {
        return windowSamples > chromaWindowSamples ? windowSamples : chromaWindowSamples;
    }
};

} // namespace drumalong
