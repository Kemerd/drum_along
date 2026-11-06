#pragma once
/**
 * @file WeightsIO.h
 * @brief Read and write the editable weights file (JSON).
 *
 * The file holds the SPEC part of Weights: every musical decision, grouped
 * by layer, with the 16-step groove and fill patterns laid out per drum so
 * they can be edited by hand. The big front-end tables are never stored;
 * realize() derives them from the spec for whatever rate the host runs at.
 *
 * Loading is an overlay: keys missing from the file keep the values already
 * in the Weights (normally the hand-reasoned defaults), so old files keep
 * working when new weights are added.
 */

#include "Weights.h"

#include <string>

namespace drumalong {

/// Current file format version.
constexpr int kWeightsFileVersion = 1;

/**
 * @brief Write the SPEC of @p w to @p path as pretty JSON.
 * @return false (with @p error set) if the file cannot be written.
 */
bool saveWeightsFile(const std::string& path, const Weights& w, std::string* error = nullptr);

/**
 * @brief Overlay the values in @p path onto the SPEC of @p w.
 *
 * Does not realize: call realize() afterwards, which also validates ranges.
 * @return false (with @p error set) on a missing file, bad JSON, a wrong
 *         format tag, or a value of the wrong type or array length. @p w is
 *         untouched on failure.
 */
bool loadWeightsFile(const std::string& path, Weights& w, std::string* error = nullptr);

/**
 * @brief Read a trained beat network (weights/beatnet.json, written by
 *        scripts/train/train_beatnet.py).
 *
 * Every tensor is shape-checked against the architecture in BeatNet.h.
 * @return false (with @p error set) on any mismatch; @p net is untouched.
 */
bool loadBeatNetFile(const std::string& path, BeatNetWeights& net, std::string* error = nullptr);

} // namespace drumalong
