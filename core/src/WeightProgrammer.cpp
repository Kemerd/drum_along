/**
 * @file WeightProgrammer.cpp
 * @brief Hand-written weights for every layer of the drummer.
 *
 * ============================================================================
 *  How to read this file
 * ============================================================================
 *  Each defaultsXxx() function owns one layer. Its comment block says what
 *  the layer must compute, and the code writes the numbers that make it
 *  compute exactly that. Nothing is fitted to data. Every value can be
 *  overridden in weights/drummer.json without recompiling.
 *
 *  The realizeXxx() functions turn the spec into the big front-end tables
 *  (DFT rows, mel and chroma filters) for the host sample rate.
 *
 *  Data flow per 10 ms frame:
 *
 *    audio --[DFT]--> power --[transient split]--> percussive power
 *      --[mel + log]--> log-mel --[flux]--> onset strength x
 *    audio --[long DFT]--> power --[chroma]--> 12 pitch classes
 *      --[before/after compare]--> chord-change strength h
 *    x --[autocorrelation bank]--> tempo
 *    x, sub-bass, h --[template bank + PLL]--> beat phase
 *    sub-bass, x, h --[accent slots]--> which beat is "1"
 *    loudness, density --[song-relative range]--> intensity
 *    intensity, bar position --[groove head]--> drum hits
 * ============================================================================
 */

#include "drumalong/WeightProgrammer.h"

#include <algorithm>
#include <cmath>

namespace drumalong {

namespace {

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// HTK mel scale.
double hzToMel(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }
double melToHz(double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); }

/**
 * @brief Power of two nearest to @p x in the log domain.
 *
 * 0.023 s is 1014 samples at 44.1 kHz (-> 1024) and 1104 at 48 kHz
 * (-> 1024, not 2048), which keeps the window close to its nominal length.
 */
int nearestPow2(double x) {
    if (!(x >= 2.0)) {
        return 2;
    }
    int p = 1;
    while (static_cast<double>(p) * 2.0 <= x) {
        p *= 2;
    }
    return (x / p < (2.0 * p) / x) ? p : 2 * p;
}

/// Per-frame leak coefficient for a leaky integrator with time constant tau.
float decayForSeconds(double tauSeconds) {
    if (!(tauSeconds > 0.0)) {
        return 0.0f;
    }
    return static_cast<float>(std::exp(-1.0 / (tauSeconds * kFrameRate)));
}

/**
 * @brief The DFT layer's window: periodic Hann times the scale that makes a
 *        full-scale sine centred on a bin read magnitude 1.
 *
 * The layer itself is X_k = sum_n window[n] x[n] e^{-2 pi i k n / N}: row 2k
 * of its weight matrix is window[n] cos(2 pi k n / N) and row 2k+1 is
 * -window[n] sin(...). Those rows are fully determined by the window, so the
 * window is all we store; an FFT evaluates the same map in O(N log N).
 */
void writeDftWindow(std::vector<float>& window, int N) {
    std::vector<double> hann(static_cast<size_t>(N));
    double windowSum = 0.0;
    for (int n = 0; n < N; ++n) {
        hann[static_cast<size_t>(n)] = 0.5 - 0.5 * std::cos(2.0 * kPi * n / N);
        windowSum += hann[static_cast<size_t>(n)];
    }
    const double scale = (windowSum > 0.0) ? 2.0 / windowSum : 1.0;
    window.resize(static_cast<size_t>(N));
    for (int n = 0; n < N; ++n) {
        window[static_cast<size_t>(n)] = static_cast<float>(scale * hann[static_cast<size_t>(n)]);
    }
}

// ===========================================================================
// Front end
// ===========================================================================
/**
 * The front end turns 23 ms of audio into "how much energy is in each
 * perceptual band right now".
 *
 *  - A DFT is a linear map: one dense layer of Hann-windowed cosines and
 *    sines. Only bins up to the top mel edge are kept.
 *  - Triangular mel filters, each normalised to sum to 1, so a band reads
 *    average power density and wide bands do not drown narrow ones.
 *  - A flat 35-140 Hz band isolates kick drums and bass-note attacks.
 *  - log(1 + 1e5 * m) puts the compression knee near -50 dB per bin, so
 *    flux becomes a level-independent ratio for everything audible.
 */
void defaultsFrontEnd(Weights& w) {
    w.windowSeconds = 0.023f;
    w.melMinHz      = 30.0f;
    w.melMaxHz      = 16000.0f;
    w.lowBandMinHz  = 35.0f;
    w.lowBandMaxHz  = 140.0f;
    w.logGain       = 1.0e5f;
}

// ===========================================================================
// Transient channel
// ===========================================================================
/**
 * Singing, held guitar chords and pads all change their spectrum slowly, and
 * vibrato makes the plain flux twitch. Drum-like attacks are the opposite:
 * broadband in one instant. The classic separation (Fitzgerald, 2010) uses
 * two median filters on the spectrogram:
 *
 *   sustained  = median of each bin over the last 9 frames (90 ms, causal)
 *   transient  = median across 17 neighbouring bins in this frame
 *
 * A soft mask T^2 / (T^2 + S^2) keeps the transient part of every bin. The
 * onset flux is then a 50/50 blend of plain flux and transient-only flux:
 * pure transient flux would go deaf to a soft strum, pure plain flux hears
 * every vowel.
 */
void defaultsTransient(Weights& w) {
    w.hpssTimeFrames = 9;
    w.hpssFreqBins   = 17;
    w.percussiveMix  = 0.5f;
}

// ===========================================================================
// Onset detection
// ===========================================================================
/**
 * Flux: for each band, how much did the log energy *rise* since the previous
 * frame? Comparing against the max of the three neighbouring bands of the
 * previous frame (the SuperFlux trick) stops vibrato from sliding between
 * bands and faking onsets. All bands count equally.
 *
 * Dividing by its own 3 s running mean makes "1.0" mean "an average moment
 * of this song", so a bias of 1.0 keeps only above-average events. Then a
 * [1/4, 1/2, 1/4] kernel widens each spike to three frames, so onsets that
 * wobble by +-10 ms still line up in the autocorrelation and the template.
 */
void defaultsOnset(Weights& w) {
    w.fluxReadout.assign(kMelBands, 1.0f / kMelBands);
    w.fluxMaxRadius = 1;
    w.fluxNormDecay = decayForSeconds(3.0);
    w.fluxNormEps   = 0.002f;
    w.onsetBias     = 1.0f;
    w.smoothKernel[0] = 0.25f;
    w.smoothKernel[1] = 0.50f;
    w.smoothKernel[2] = 0.25f;
}

// ===========================================================================
// Chord-change channel
// ===========================================================================
/**
 * Without drums, guitars and bass hit the beat and the "&" equally hard, so
 * onsets alone cannot tell them apart. Harmony can: chords change on beats
 * (most often on the 1), almost never on an off-beat eighth.
 *
 *  - Pitch needs frequency resolution a 23 ms window cannot give, so this
 *    channel has its own 93 ms DFT (10.8 Hz bins at 44.1 kHz), read only
 *    from 80 Hz to 1.6 kHz where chord tones live.
 *  - Every bin is folded onto the 12 pitch classes with a Gaussian of 0.5
 *    semitones. Low bins that cannot resolve a semitone get a wider Gaussian,
 *    so they spread evenly and cannot fake a change.
 *  - Change = 1 - cosine similarity between the average chroma of the last
 *    120 ms and of the 120 ms before it. Two whole windows on each side make
 *    a sung melody note (a partial change) score far below a new chord (a
 *    total change), and re-strumming the same chord scores nothing at all.
 *  - Like the flux, it is divided by its own running mean (4 s) and
 *    rectified above 1.0, then stored at the moment the change happened,
 *    which is about 15 frames in the past.
 */
void defaultsChord(Weights& w) {
    w.chromaWindowSeconds = 0.093f;
    w.chromaMinHz         = 80.0f;
    w.chromaMaxHz         = 1600.0f;
    w.chromaSigma         = 0.5f;
    w.chromaLogGain       = 1.0e4f;
    w.harmonicSpanFrames  = 12;
    w.harmonicNormDecay   = decayForSeconds(4.0);
    w.harmonicNormEps     = 0.01f;
    w.harmonicBias        = 1.0f;
}

// ===========================================================================
// Tempo
// ===========================================================================
/**
 * A leaky autocorrelation R(lag) of the onset signal, one accumulator per lag
 * from 1 to 240 frames, with a 4 s memory. Music that repeats every L frames
 * peaks at R(L), and also at 2L, 3L, and at L/2 when eighths are busy.
 *
 *  1. Harmonic readout. Salience(L) = R(L) + 0.5*R(2L) + 0.25*R(L/2).
 *  2. A log-Gaussian prior centred on 110 BPM, 0.8 octaves wide: when L and
 *     2L tie, it picks the one a drummer would count.
 *
 *  3. A chord vote. Chords change on bar lines (every 2, 4 or 8 beats), so
 *     a second autocorrelation, on the chord-change signal with an 8 s
 *     memory and lags up to 960 frames, adds 0.5 x the mean of its
 *     normalised peaks at 2L, 4L and 8L. A riff that repeats every three
 *     sixteenths (a dotted-eighth guitar figure) fools the onset readout
 *     into 4/3 of the real tempo, but its multiples never land on the
 *     chord changes, so the vote pulls the drummer back to the real beat.
 *
 * Lock: periodicity above 0.2 for 0.8 s, and only after 2.5 s of audio.
 *
 * Small drifts are followed (the nearest peak, refined over several
 * harmonics). A real tempo change needs three things at once, for 3 s: the
 * current tempo has stopped repeating (periodicity below lockOff), the
 * challenger clearly repeats (above lockOn), and it is 20% more salient.
 * Quiet breakdowns and half-time passages fail the first test, so the
 * drummer keeps counting through them.
 */
void defaultsTempo(Weights& w) {
    w.acDecay = decayForSeconds(4.0);

    const double centrePeriod = 60.0 * kFrameRate / 110.0;   // 54.5 frames.
    const double widthOctaves = 0.8;
    w.tempoPrior.assign(kNumPeriods, 0.0f);
    for (int i = 0; i < kNumPeriods; ++i) {
        const double L   = kMinPeriodFrames + i;
        const double oct = std::log2(L / centrePeriod);
        w.tempoPrior[static_cast<size_t>(i)] =
            static_cast<float>(std::exp(-0.5 * (oct / widthOctaves) * (oct / widthOctaves)));
    }

    w.harmonicDouble    = 0.5f;
    w.harmonicHalf      = 0.25f;
    w.tempoHarmWeight   = 0.5f;
    w.harmAcDecay       = decayForSeconds(8.0);
    w.tempoFollow       = 0.1f;
    w.tempoTolerance    = 0.06f;
    w.tempoSwitchMargin = 0.2f;
    w.tempoSwitchFrames = 300;

    // Settling: the first lock happens after 3-6 s, before the chord vote has
    // seen a full chord cycle. For 12 s after it, a challenger 30% stronger
    // for 2 s may replace the early count, even if the early count still
    // "fits". Like a drummer re-counting in the first bars, then holding.
    w.tempoSettleFrames       = 1200;
    w.tempoSettleMargin       = 0.3f;
    w.tempoSettleSwitchFrames = 200;

    // A pause in the band is the one moment a new tempo is likely, so it
    // reopens the settling window. Loudness decides, not onsets: a dense wall
    // of guitars has almost no attacks either, and must not count. On the log
    // scale of this loudness, 0.6 of the last 3 seconds is a real collapse, and
    // holding it for a beat excludes the gaps between phrases. When the band
    // returns, half of the tempo memory goes: enough for a new tempo to surface
    // in a few seconds, while the same tempo simply carries on.
    w.pauseBeats  = 1.0f;
    w.pauseLevel  = 0.6f;
    w.pauseForget = 0.5f;
    w.lockOn            = 0.20f;
    w.lockOff           = 0.10f;
    w.lockMinFrames     = 80;
    w.minSoundFrames    = 250;
}

// ===========================================================================
// Beat phase
// ===========================================================================
/**
 * The template bank has one unit per candidate phase phi ("the beats were
 * phi, phi+P, phi+2P... frames ago"). Each unit sums three histories at
 * those positions, weighted 0.8^k so recent beats count most:
 *
 *   broadband onsets  (1.0)  where things are struck,
 *   sub-bass onsets   (1.0)  kicks and bass notes land on beats,
 *   chord changes     (1.0)  harmony moves on beats, not on the "&".
 *
 * The phase-locked loop follows the strongest unit *near its current phase*
 * (within 0.18 beats), slewing 4% of the error per frame. It only jumps to a
 * different phase when that phase beats ours by 15% for a full second: the
 * way a human drummer resets after realising they flipped the beat, without
 * flickering every time the "&" gets a loud strum.
 *
 * Leadership. In a real band the drummer and the players are two coupled
 * clocks, but the drummer carries most of the inertia: the band mostly
 * follows the drums, the drums only drift gently toward the band. So the
 * phase correction and the tempo follow rate are both scaled by
 * (1 - 0.8 * leadership). At 0.5 the drummer corrects 2.4% of its error per
 * frame (about a 0.4 s time constant): it holds its own time through a
 * rushed fill or a sloppy strum, but still lands with the band within a bar.
 */
void defaultsPhase(Weights& w) {
    w.leadership          = 0.5f;
    w.templateLowWeight   = 1.0f;
    w.templateHarmWeight  = 1.0f;
    w.templateBeats       = 6;
    w.templateDecay       = 0.8f;
    w.templateMinContrast = 1.25f;
    w.pllPhaseGain        = 0.04f;
    w.phaseSnapError      = 0.18f;
    w.phaseSnapMargin     = 0.15f;
    w.phaseSnapFrames     = 100;

    // Right after the first lock or a pause, nothing is known yet about where
    // the beat sits, so a quick correction is worth its risk.
    w.phaseSettleSnapFrames = 100;

    // Measured, not assumed: averaged over 8,466 Slakh beats, the onset flux
    // peaks one frame after the true note start, and on real recorded drum
    // stems the drummer landed 8-18 ms late. Calibrated on the Slakh
    // validation split once the learned layer is in (see docs/NOTES.md);
    // 0 until then.
    w.timingOffsetMs      = 0.0f;
}

// ===========================================================================
// Downbeat
// ===========================================================================
/**
 * Which of the four beats is "1"? Three cues, each read in a window around
 * every beat and added into one of four leaky slots:
 *
 *   sub-bass onsets (1.0)  kick on 1, new bass note on 1,
 *   chord changes   (1.0)  chords change on the 1 far more than anywhere,
 *   broadband       (0.3)  a tie-breaker only: snares on 2 and 4 are
 *                          broadband and would mislead it.
 *
 * Evidence rules: no decision until every slot was measured twice; the
 * first informed decision is free; afterwards a challenger must win by 15%
 * on 8 consecutive beats. Each accent is only measurable after its beat has
 * passed, so switching on one beat always lands the "1" in the past.
 */
void defaultsDownbeat(Weights& w) {
    w.accentPre            = 2;
    w.accentPost           = 4;
    w.accentLowWeight      = 1.0f;
    w.accentAllWeight      = 0.3f;
    w.accentHarmWeight     = 1.0f;
    w.downbeatDecay        = 0.8f;
    w.downbeatSwitchMargin = 0.15f;
    w.downbeatSwitchBeats  = 8;
    w.downbeatMinUpdates   = 2;
}

// ===========================================================================
// Intensity
// ===========================================================================
/**
 * "You dig in, he follows." Intensity is where the input sits right now
 * within *this song's own* dynamic range. Two fast (1.5 s) signals, mean
 * log-mel loudness and log onset density, each feed a min/max envelope that
 * slowly forgets, and map to 0.5 + (value - mid) / span. A steady song sits
 * near 0.5: a mid groove, exactly what a session drummer would play.
 * A rise of 0.22 within 2 s flags a section change: crash on the next 1.
 */
void defaultsIntensity(Weights& w) {
    w.loudFastDecay    = decayForSeconds(1.5);
    w.densFastDecay    = decayForSeconds(1.5);
    w.levelSmoothDecay = decayForSeconds(0.4);

    w.loudMinSpan = 1.5f;     // ln-power units: ~6.5 dB.
    w.densMinSpan = 0.8f;     // ln units: a factor of ~2.2 in onset activity.
    w.loudLeak    = 0.00025f;
    w.densLeak    = 0.00015f;
    w.loudMix     = 0.65f;

    w.sectionLookbackFrames = 200;
    w.sectionJump           = 0.22f;
    w.sectionCooldownFrames = 400;

    w.silenceDecay       = decayForSeconds(0.2);
    w.silenceLevel       = 0.1f;
    w.silenceFrames      = 40;
    w.resetSilenceFrames = 300;
}

// ===========================================================================
// Groove head
// ===========================================================================
/**
 * A 16-step bar (sixteenth notes):
 *
 *     beat 1        beat 2        beat 3        beat 4
 *     0  1  2  3    4  5  6  7    8  9  10 11   12 13 14 15
 *     1  e  &  a    2  e  &  a    3  e  &  a    4  e  &  a
 *
 * Each cell fires for a band of intensity [on, off). Low to high:
 *
 *   0.00  Kick 1 and 3, snare 2 and 4, closed hat on quarters.
 *   0.20  Hats fill in to eighths. The default pop/rock groove.
 *   0.35  Kick on the "& of 3".
 *   0.45  Open hat on the "& of 4", closed again on the next 1.
 *   0.55  Ghost snare on the "a of 2".
 *   0.60  Kick on the "& of 2".
 *   0.62  Sixteenth hat accents on the "a of 1" and "a of 3".
 *   0.70  Second ghost on the "a of 4".
 *   0.82  Hats give way to ride eighths: the big chorus.
 *   0.85  Kick on the "& of 4" too.
 *
 * Fills use the last bar of every 8-bar phrase, sized by intensity:
 *   below 0.15 none; to 0.40 a two-note pickup; to 0.62 a snare build;
 *   above that tom fills, from beat 3 above 0.72. Toms are the loudest
 *   statement a kit can make, so they only appear when the song drives.
 *
 * layOutLevel (default 0, off): below it the drummer stops playing, for
 * styles where a quiet intro or breakdown should have no drums at all.
 */
void defaultsGroove(Weights& w) {
    constexpr float kNever  = 2.0f;    // Above any reachable level.
    constexpr float kAlways = 0.0f;
    constexpr float kTop    = 2.0f;    // "No upper limit".

    w.hitOnLevel.resize(kStepsPerBar, kNumDrums);
    w.hitOffLevel.resize(kStepsPerBar, kNumDrums);
    w.velocity.resize(kStepsPerBar, kNumDrums);
    for (int s = 0; s < kStepsPerBar; ++s) {
        for (int d = 0; d < kNumDrums; ++d) {
            w.hitOnLevel.at(s, d)  = kNever;
            w.hitOffLevel.at(s, d) = kTop;
        }
    }

    auto cell = [&w](int step, int drum, float on, float off, float vel) {
        w.hitOnLevel.at(step, drum)  = on;
        w.hitOffLevel.at(step, drum) = off;
        w.velocity.at(step, drum)    = vel;
    };

    // ---- Kick ---------------------------------------------------------
    cell(0,  kKick, kAlways, kTop, 112.0f);   // 1
    cell(8,  kKick, kAlways, kTop, 100.0f);   // 3
    cell(10, kKick, 0.35f,   kTop,  92.0f);   // & of 3
    cell(6,  kKick, 0.60f,   kTop,  88.0f);   // & of 2
    cell(14, kKick, 0.85f,   kTop,  86.0f);   // & of 4

    // ---- Snare --------------------------------------------------------
    cell(4,  kSnare, kAlways, kTop, 104.0f);  // 2
    cell(12, kSnare, kAlways, kTop, 108.0f);  // 4
    cell(7,  kSnare, 0.55f,   kTop,  32.0f);  // ghost, a of 2
    cell(15, kSnare, 0.70f,   kTop,  36.0f);  // ghost, a of 4

    // ---- Closed hat ---------------------------------------------------
    cell(0,  kClosedHat, kAlways, 0.82f, 88.0f);
    cell(4,  kClosedHat, kAlways, 0.82f, 80.0f);
    cell(8,  kClosedHat, kAlways, 0.82f, 84.0f);
    cell(12, kClosedHat, kAlways, 0.82f, 80.0f);
    cell(2,  kClosedHat, 0.20f,   0.82f, 62.0f);
    cell(6,  kClosedHat, 0.20f,   0.82f, 62.0f);
    cell(10, kClosedHat, 0.20f,   0.82f, 62.0f);
    cell(14, kClosedHat, 0.20f,   0.45f, 62.0f);   // until the open hat takes over
    cell(3,  kClosedHat, 0.62f,   0.82f, 44.0f);
    cell(11, kClosedHat, 0.62f,   0.82f, 44.0f);

    // ---- Open hat -----------------------------------------------------
    cell(14, kOpenHat, 0.45f, 0.82f, 78.0f);

    // ---- Ride ---------------------------------------------------------
    for (int s = 0; s < kStepsPerBar; s += 2) {
        cell(s, kRide, 0.82f, kTop, (s % kStepsPerBeat) == 0 ? 96.0f : 74.0f);
    }

    // -----------------------------------------------------------------------
    // Fill tables.
    // -----------------------------------------------------------------------
    w.fillVelocity.assign(kNumFillVariants, Matrix{});
    for (Matrix& m : w.fillVelocity) {
        m.resize(kStepsPerBar, kNumDrums);
    }

    {   // Pickup: the groove continues on beat 4, plus two snares into the 1.
        Matrix& f = w.fillVelocity[kFillPickup];
        f.at(12, kKick)      = 96.0f;   f.at(12, kSnare) = 104.0f;
        f.at(12, kClosedHat) = 78.0f;
        f.at(14, kClosedHat) = 60.0f;   f.at(14, kSnare) = 70.0f;
        f.at(15, kSnare)     = 88.0f;
    }
    {   // Snare build across beat 4.
        Matrix& f = w.fillVelocity[kFillSnare];
        f.at(8,  kKick)  = 95.0f;  f.at(8,  kSnare) = 84.0f;
        f.at(10, kSnare) = 80.0f;  f.at(11, kSnare) = 86.0f;
        f.at(12, kKick)  = 90.0f;  f.at(12, kSnare) = 88.0f;
        f.at(13, kSnare) = 94.0f;
        f.at(14, kSnare) = 100.0f;
        f.at(15, kSnare) = 108.0f;
    }
    {   // Snare into toms.
        Matrix& f = w.fillVelocity[kFillSnareToms];
        f.at(8,  kSnare) = 90.0f;  f.at(9,  kSnare) = 58.0f;
        f.at(10, kKick)  = 85.0f;  f.at(10, kSnare) = 90.0f;
        f.at(12, kSnare) = 96.0f;  f.at(13, kSnare) = 84.0f;
        f.at(14, kKick)  = 95.0f;  f.at(14, kTomHi) = 100.0f;
        f.at(15, kTomLo) = 110.0f;
    }
    {   // Down the toms.
        Matrix& f = w.fillVelocity[kFillToms];
        f.at(8,  kKick)   = 92.0f; f.at(8,  kTomHi)  = 90.0f;
        f.at(9,  kTomHi)  = 74.0f;
        f.at(10, kTomMid) = 94.0f; f.at(11, kTomMid) = 80.0f;
        f.at(12, kKick)   = 90.0f; f.at(12, kSnare)  = 100.0f;
        f.at(13, kTomHi)  = 95.0f;
        f.at(14, kTomMid) = 100.0f;
        f.at(15, kTomLo)  = 112.0f;
    }

    // -----------------------------------------------------------------------
    // Feel, dynamics and phrasing. The stock groove is straight rock: no
    // swing. Style files (weights/styles/*.json) override this.
    // -----------------------------------------------------------------------
    w.swing8             = 0.0f;
    w.swing16            = 0.0f;
    w.velLevelGain       = 40.0f;
    w.velHumanize        = 6.0f;
    w.layOutLevel        = 0.0f;

    // A drummer settles on a pattern and lets the dynamics move inside it.
    // Intensity wanders from bar to bar, so reading it live makes the kick and
    // hats flicker. The pattern is re-chosen only on a 4-bar line, and only for
    // a real change of intensity; velocity still follows every step.
    w.grooveHoldBars     = 4;
    w.grooveHoldChange   = 0.12f;
    w.fillMinLevel       = 0.15f;
    w.fillSnareLevel     = 0.40f;
    w.fillTomLevel       = 0.62f;
    w.fillLevelLong      = 0.72f;
    w.fillStartShort     = 12;
    w.fillStartLong      = 8;
    w.fillMinBarsPlayed  = 4;
    w.crashLevel         = 0.35f;
    w.crashVelocity      = 112.0f;
    w.noteLengthSeconds  = 0.06f;
    w.lateToleranceSteps = 0.5f;
}

// ===========================================================================
// Learned layer usage
// ===========================================================================
/**
 * The built-in defaults have no learned layer (beatNetFile is empty), so the
 * stock drummer is 100% hand-written. When weights/drummer.json names a
 * beatnet.json, its activations join the hand-written evidence:
 *
 *   onsetMix 0.75     the onset signal driving tempo and phase is 3/4 learned
 *                     beat activation, 1/4 hand-written flux: the learned
 *                     layer leads, the hand-written ears keep it honest.
 *   onsetGain 10      a confident beat (p = 1) counts like a strong onset.
 *   downbeatWeight 2  a learned downbeat outweighs the hand-written cues.
 *   layOutBelow 0     laying out from the "drummer would play" output is off
 *                     until it has been measured on real songs.
 */
void defaultsLearned(Weights& w) {
    w.beatNetFile.clear();
    w.beatOnsetMix       = 0.75f;
    w.beatOnsetGain      = 10.0f;
    w.beatDownbeatWeight = 2.0f;
    w.beatLayOutBelow    = 0.0f;
    w.beatLayInAbove     = 0.0f;
    w.beatDrumsDecay     = decayForSeconds(1.0);
    w.beatNet            = BeatNetWeights{};
}

// ===========================================================================
// Realization: spec -> tables for one sample rate
// ===========================================================================

bool fail(std::string* error, const char* message) {
    if (error) {
        *error = message;
    }
    return false;
}

/// Validate every spec field the forward pass relies on. A hand-edited file
/// must never be able to crash the audio thread.
bool validateSpec(const Weights& w, std::string* error) {
    if (!(w.windowSeconds >= 0.005f && w.windowSeconds <= 0.2f)) return fail(error, "windowSeconds must be in [0.005, 0.2]");
    if (!(w.melMinHz > 0.0f && w.melMaxHz > w.melMinHz))         return fail(error, "need 0 < melMinHz < melMaxHz");
    if (!(w.lowBandMinHz > 0.0f && w.lowBandMaxHz > w.lowBandMinHz)) return fail(error, "need 0 < lowBandMinHz < lowBandMaxHz");
    if (!(w.logGain > 0.0f && w.chromaLogGain > 0.0f))           return fail(error, "log gains must be positive");
    if (w.hpssTimeFrames < 1 || w.hpssTimeFrames > 64)            return fail(error, "hpssTimeFrames must be in [1, 64]");
    if (w.hpssFreqBins < 1 || w.hpssFreqBins > 129)               return fail(error, "hpssFreqBins must be in [1, 129]");
    if (!(w.percussiveMix >= 0.0f && w.percussiveMix <= 1.0f))    return fail(error, "percussiveMix must be in [0, 1]");
    if (!(w.leadership >= 0.0f && w.leadership <= 1.0f))          return fail(error, "leadership must be in [0, 1]");
    if (static_cast<int>(w.fluxReadout.size()) != kMelBands)      return fail(error, "fluxReadout must have 40 values");
    if (w.fluxMaxRadius < 0 || w.fluxMaxRadius > 8)               return fail(error, "fluxMaxRadius must be in [0, 8]");
    if (!(w.chromaWindowSeconds >= 0.02f && w.chromaWindowSeconds <= 0.4f)) return fail(error, "chromaWindowSeconds must be in [0.02, 0.4]");
    if (!(w.chromaMinHz > 0.0f && w.chromaMaxHz > w.chromaMinHz)) return fail(error, "need 0 < chromaMinHz < chromaMaxHz");
    if (!(w.chromaSigma > 0.0f))                                  return fail(error, "chromaSigma must be positive");
    if (w.harmonicSpanFrames < 2 || 2 * w.harmonicSpanFrames > kChromaHist) return fail(error, "harmonicSpanFrames must be in [2, 32]");
    if (static_cast<int>(w.tempoPrior.size()) != kNumPeriods)     return fail(error, "tempoPrior must have 92 values");
    if (w.grooveHoldBars < 1)                                     return fail(error, "holdBars must be >= 1");
    if (w.grooveHoldChange < 0.0f)                                return fail(error, "holdChange must be >= 0");
    if (w.beatLayInAbove < w.beatLayOutBelow)                     return fail(error, "layInAbove must be >= layOutBelow");
    if (w.phaseSettleSnapFrames < 1)                              return fail(error, "settleSnapFrames must be >= 1");
    if (w.pauseBeats <= 0.0f)                                     return fail(error, "pauseBeats must be > 0");
    if (w.pauseLevel <= 0.0f || w.pauseLevel >= 1.0f)             return fail(error, "pauseLevel must be in (0, 1)");
    if (w.pauseForget < 0.0f || w.pauseForget > 1.0f)             return fail(error, "pauseForget must be in [0, 1]");
    if (w.templateBeats < 1 || w.templateBeats > 8)               return fail(error, "templateBeats must be in [1, 8]");
    if (w.accentPre < 0 || w.accentPost < 0 || w.accentPre > 20 || w.accentPost > 20) return fail(error, "accent window must be in [0, 20] frames");
    if (w.hitOnLevel.rows != kStepsPerBar || w.hitOnLevel.cols != kNumDrums ||
        w.hitOffLevel.rows != kStepsPerBar || w.hitOffLevel.cols != kNumDrums ||
        w.velocity.rows != kStepsPerBar || w.velocity.cols != kNumDrums) return fail(error, "groove tables must be 16 steps x 9 drums");
    if (static_cast<int>(w.fillVelocity.size()) != kNumFillVariants) return fail(error, "need 4 fill tables");
    for (const Matrix& m : w.fillVelocity) {
        if (m.rows != kStepsPerBar || m.cols != kNumDrums)         return fail(error, "fill tables must be 16 steps x 9 drums");
    }
    if (w.fillStartShort < 0 || w.fillStartShort >= kStepsPerBar ||
        w.fillStartLong < 0 || w.fillStartLong >= kStepsPerBar)   return fail(error, "fill starts must be steps 0..15");
    if (!(w.beatOnsetMix >= 0.0f && w.beatOnsetMix <= 1.0f))      return fail(error, "beatNet.onsetMix must be in [0, 1]");
    if (!(w.swing8 >= 0.0f && w.swing8 <= 1.0f && w.swing16 >= 0.0f && w.swing16 <= 1.0f))
                                                                  return fail(error, "swing8 / swing16 must be in [0, 1]");
    if (!w.beatNetFile.empty() && !w.beatNet.valid())             return fail(error, "beatNet file named but its weights are missing or malformed");
    return true;
}

void realizeFrontEnd(Weights& w, double sr) {
    w.sampleRate    = sr;
    w.hopSamples    = std::max(1, static_cast<int>(std::lround(sr / kFrameRate)));
    w.windowSamples = nearestPow2(w.windowSeconds * sr);

    const int    N        = w.windowSamples;
    const double binHz    = sr / N;
    const double melTopHz = std::min(static_cast<double>(w.melMaxHz), 0.95 * 0.5 * sr);
    const double melBotHz = std::min(static_cast<double>(w.melMinHz), 0.5 * melTopHz);

    // Keep every bin up to one past the top mel edge, never past Nyquist.
    const int topBin = static_cast<int>(std::floor(melTopHz / binHz)) + 1;
    w.numBins = std::clamp(topBin + 1, 2, N / 2 + 1);

    writeDftWindow(w.dftWindow, N);

    // ---- Mel filters ----------------------------------------------------
    const double melLo = hzToMel(melBotHz);
    const double melHi = hzToMel(melTopHz);
    std::vector<double> edgesHz(static_cast<size_t>(kMelBands + 2));
    for (int i = 0; i < kMelBands + 2; ++i) {
        edgesHz[static_cast<size_t>(i)] = melToHz(melLo + (melHi - melLo) * i / (kMelBands + 1));
    }

    w.mel.resize(kMelBands, w.numBins);
    for (int b = 0; b < kMelBands; ++b) {
        const double lo  = edgesHz[static_cast<size_t>(b)];
        const double ctr = edgesHz[static_cast<size_t>(b + 1)];
        const double hi  = edgesHz[static_cast<size_t>(b + 2)];
        double rowSum = 0.0;
        for (int k = 0; k < w.numBins; ++k) {
            const double f = k * binHz;
            double weight = 0.0;
            if (f > lo && f <= ctr && ctr > lo) {
                weight = (f - lo) / (ctr - lo);
            } else if (f > ctr && f < hi && hi > ctr) {
                weight = (hi - f) / (hi - ctr);
            }
            w.mel.at(b, k) = static_cast<float>(weight);
            rowSum += weight;
        }
        if (rowSum > 0.0) {
            for (int k = 0; k < w.numBins; ++k) {
                w.mel.at(b, k) = static_cast<float>(w.mel.at(b, k) / rowSum);
            }
        } else {
            // Band narrower than a bin: read the nearest bin instead of nothing.
            const int nearest = std::clamp(static_cast<int>(std::lround(ctr / binHz)), 0, w.numBins - 1);
            w.mel.at(b, nearest) = 1.0f;
        }
    }

    // ---- Sub-bass band ----------------------------------------------------
    w.lowBand.assign(static_cast<size_t>(w.numBins), 0.0f);
    int lowCount = 0;
    for (int k = 1; k < w.numBins; ++k) {
        const double f = k * binHz;
        if (f >= w.lowBandMinHz && f <= w.lowBandMaxHz) {
            w.lowBand[static_cast<size_t>(k)] = 1.0f;
            ++lowCount;
        }
    }
    if (lowCount > 0) {
        for (float& v : w.lowBand) {
            v /= static_cast<float>(lowCount);
        }
    } else {
        const double centre  = 0.5 * (w.lowBandMinHz + w.lowBandMaxHz);
        const int    nearest = std::clamp(static_cast<int>(std::lround(centre / binHz)), 1, w.numBins - 1);
        w.lowBand[static_cast<size_t>(nearest)] = 1.0f;
    }
}

bool realizeChroma(Weights& w, double sr, std::string* error) {
    const int    Nc    = nearestPow2(w.chromaWindowSeconds * sr);
    const double binHz = sr / Nc;
    const double topHz = std::min(static_cast<double>(w.chromaMaxHz), 0.95 * 0.5 * sr);

    const int kLo = std::max(1, static_cast<int>(std::ceil(w.chromaMinHz / binHz)));
    const int kHi = std::min(Nc / 2, static_cast<int>(std::floor(topHz / binHz)));
    if (kHi < kLo) {
        return fail(error, "chroma range holds no DFT bins at this sample rate");
    }

    w.chromaWindowSamples = Nc;
    w.chromaBinLo         = kLo;
    w.numChromaBins       = kHi - kLo + 1;
    writeDftWindow(w.chromaWindow, Nc);

    // -----------------------------------------------------------------------
    // Fold every bin onto the 12 pitch classes. Each bin hands out a total
    // weight of 1, spread by a Gaussian around its own pitch class.
    // -----------------------------------------------------------------------
    w.chroma.resize(kChromaClasses, w.numChromaBins);
    const double semitoneRatio = std::pow(2.0, 1.0 / 12.0) - 1.0;
    for (int i = 0; i < w.numChromaBins; ++i) {
        const double f    = (kLo + i) * binHz;
        const double midi = 69.0 + 12.0 * std::log2(f / 440.0);
        double pc = std::fmod(midi, 12.0);
        if (pc < 0.0) {
            pc += 12.0;
        }

        // A bin wider than a semitone cannot say which note it holds.
        const double binsPerSemitone = (f * semitoneRatio) / binHz;
        const double sigma = std::max(static_cast<double>(w.chromaSigma), 0.5 / std::max(binsPerSemitone, 1e-6));

        double weights[kChromaClasses];
        double total = 0.0;
        for (int c = 0; c < kChromaClasses; ++c) {
            double d = pc - c;
            d -= 12.0 * std::floor(d / 12.0 + 0.5);   // circular distance in [-6, 6)
            weights[c] = std::exp(-0.5 * (d / sigma) * (d / sigma));
            total += weights[c];
        }
        for (int c = 0; c < kChromaClasses; ++c) {
            w.chroma.at(c, i) = static_cast<float>(total > 0.0 ? weights[c] / total : 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Where in time does a detected change belong? The chroma window is
    // centred (Nc - N)/2 samples before the onset window, and the
    // before/after boundary sits D - 0.5 chroma frames back.
    // -----------------------------------------------------------------------
    const double offset = w.harmonicSpanFrames - 0.5 +
                          (Nc - w.windowSamples) / (2.0 * std::max(1, w.hopSamples));
    w.harmonicOffsetFrames = std::max(1, static_cast<int>(std::lround(offset)));

    // The template reads chord changes from beats >= 1 period back; the
    // newest change must be older than the shortest period.
    if (w.harmonicOffsetFrames >= kMinPeriodFrames - 1) {
        return fail(error, "harmonicSpanFrames too long: chord changes would arrive later than one beat");
    }
    return true;
}

} // namespace

// ===========================================================================
// Public entry points
// ===========================================================================

void programDefaults(Weights& w) {
    Weights fresh;
    defaultsFrontEnd(fresh);
    defaultsTransient(fresh);
    defaultsOnset(fresh);
    defaultsChord(fresh);
    defaultsTempo(fresh);
    defaultsPhase(fresh);
    defaultsDownbeat(fresh);
    defaultsIntensity(fresh);
    defaultsGroove(fresh);
    defaultsLearned(fresh);
    w = std::move(fresh);
}

bool realize(Weights& w, double sampleRate, std::string* error) {
    if (!(sampleRate >= 8000.0 && sampleRate <= 768000.0)) {
        return fail(error, "sample rate must be in [8 kHz, 768 kHz]");
    }
    if (!validateSpec(w, error)) {
        return false;
    }
    realizeFrontEnd(w, sampleRate);
    if (!realizeChroma(w, sampleRate, error)) {
        w.dftWindow.clear();   // Leave no half-realized network behind.
        return false;
    }
    return true;
}

bool programWeights(Weights& w, double sampleRate) {
    Weights fresh;
    programDefaults(fresh);
    if (!realize(fresh, sampleRate)) {
        return false;
    }
    w = std::move(fresh);
    return true;
}

} // namespace drumalong
