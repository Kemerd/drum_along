#pragma once
/**
 * @file WeightProgrammer.h
 * @brief Writes every weight of the drummer network by hand. No training.
 *
 * Two steps, so the weights file can sit in between:
 *
 *   programDefaults()   the hand-reasoned SPEC (every musical decision).
 *   (optional)          loadWeightsFile() overlays an edited spec.
 *   realize()           derives the front-end tables for a sample rate.
 */

#include "Weights.h"

#include <string>

namespace drumalong {

/**
 * @brief Write the hand-reasoned default SPEC into @p w.
 *
 * Sample-rate independent. Clears any realized tables, so call realize()
 * afterwards. Allocates: never call on the audio thread.
 */
void programDefaults(Weights& w);

/**
 * @brief Derive the DFT, mel, sub-bass and chroma tables from the SPEC.
 *
 * @param w          Weights whose SPEC is filled in.
 * @param sampleRate Host rate in Hz, [8 kHz, 768 kHz].
 * @param error      Optional: receives a message when the spec is unusable.
 * @return false if the rate or the spec is out of range; @p w is then
 *         left with no realized tables.
 */
bool realize(Weights& w, double sampleRate, std::string* error = nullptr);

/// programDefaults() + realize(). Convenience for code that wants the stock drummer.
bool programWeights(Weights& w, double sampleRate);

} // namespace drumalong
