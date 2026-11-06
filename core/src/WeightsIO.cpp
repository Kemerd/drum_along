/**
 * @file WeightsIO.cpp
 * @brief JSON serialisation of the weight SPEC.
 *
 * Every scalar weight is listed once in a registry (group, key, member
 * pointer), so saving and loading can never drift apart. Arrays and the
 * per-drum pattern tables are handled explicitly below the registry.
 */

#include "drumalong/WeightsIO.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace drumalong {

using nlohmann::json;

namespace {

// ===========================================================================
// Scalar registry
// ===========================================================================

struct FloatField {
    const char*     group;
    const char*     key;
    float Weights::* member;
};

struct IntField {
    const char*   group;
    const char*   key;
    int Weights::* member;
};

const FloatField kFloatFields[] = {
    // front end
    {"frontEnd", "windowSeconds",  &Weights::windowSeconds},
    {"frontEnd", "melMinHz",       &Weights::melMinHz},
    {"frontEnd", "melMaxHz",       &Weights::melMaxHz},
    {"frontEnd", "lowBandMinHz",   &Weights::lowBandMinHz},
    {"frontEnd", "lowBandMaxHz",   &Weights::lowBandMaxHz},
    {"frontEnd", "logGain",        &Weights::logGain},
    // transient channel
    {"transient", "percussiveMix", &Weights::percussiveMix},
    // onset
    {"onset", "fluxNormDecay",     &Weights::fluxNormDecay},
    {"onset", "fluxNormEps",       &Weights::fluxNormEps},
    {"onset", "onsetBias",         &Weights::onsetBias},
    // chord channel
    {"chord", "chromaWindowSeconds", &Weights::chromaWindowSeconds},
    {"chord", "chromaMinHz",         &Weights::chromaMinHz},
    {"chord", "chromaMaxHz",         &Weights::chromaMaxHz},
    {"chord", "chromaSigma",         &Weights::chromaSigma},
    {"chord", "chromaLogGain",       &Weights::chromaLogGain},
    {"chord", "harmonicNormDecay",   &Weights::harmonicNormDecay},
    {"chord", "harmonicNormEps",     &Weights::harmonicNormEps},
    {"chord", "harmonicBias",        &Weights::harmonicBias},
    // tempo
    {"tempo", "acDecay",           &Weights::acDecay},
    {"tempo", "harmonicDouble",    &Weights::harmonicDouble},
    {"tempo", "harmonicHalf",      &Weights::harmonicHalf},
    {"tempo", "harmWeight",        &Weights::tempoHarmWeight},
    {"tempo", "harmAcDecay",       &Weights::harmAcDecay},
    {"tempo", "follow",            &Weights::tempoFollow},
    {"tempo", "tolerance",         &Weights::tempoTolerance},
    {"tempo", "switchMargin",      &Weights::tempoSwitchMargin},
    {"tempo", "settleMargin",      &Weights::tempoSettleMargin},
    {"tempo", "pauseBeats",        &Weights::pauseBeats},
    {"tempo", "pauseLevel",        &Weights::pauseLevel},
    {"tempo", "pauseForget",       &Weights::pauseForget},
    {"tempo", "lockOn",            &Weights::lockOn},
    {"tempo", "lockOff",           &Weights::lockOff},
    // phase
    {"phase", "templateLowWeight",   &Weights::templateLowWeight},
    {"phase", "templateHarmWeight",  &Weights::templateHarmWeight},
    {"phase", "templateDecay",       &Weights::templateDecay},
    {"phase", "templateMinContrast", &Weights::templateMinContrast},
    {"phase", "pllPhaseGain",        &Weights::pllPhaseGain},
    {"phase", "snapError",           &Weights::phaseSnapError},
    {"phase", "snapMargin",          &Weights::phaseSnapMargin},
    {"phase", "leadership",          &Weights::leadership},
    {"phase", "timingOffsetMs",      &Weights::timingOffsetMs},
    // downbeat
    {"downbeat", "accentLowWeight",  &Weights::accentLowWeight},
    {"downbeat", "accentAllWeight",  &Weights::accentAllWeight},
    {"downbeat", "accentHarmWeight", &Weights::accentHarmWeight},
    {"downbeat", "decay",            &Weights::downbeatDecay},
    {"downbeat", "switchMargin",     &Weights::downbeatSwitchMargin},
    // intensity
    {"intensity", "loudFastDecay",    &Weights::loudFastDecay},
    {"intensity", "densFastDecay",    &Weights::densFastDecay},
    {"intensity", "levelSmoothDecay", &Weights::levelSmoothDecay},
    {"intensity", "loudMinSpan",      &Weights::loudMinSpan},
    {"intensity", "densMinSpan",      &Weights::densMinSpan},
    {"intensity", "loudLeak",         &Weights::loudLeak},
    {"intensity", "densLeak",         &Weights::densLeak},
    {"intensity", "loudMix",          &Weights::loudMix},
    {"intensity", "sectionJump",      &Weights::sectionJump},
    {"intensity", "silenceDecay",     &Weights::silenceDecay},
    {"intensity", "silenceLevel",     &Weights::silenceLevel},
    // groove
    {"groove", "swing8",              &Weights::swing8},
    {"groove", "swing16",             &Weights::swing16},
    {"groove", "velLevelGain",        &Weights::velLevelGain},
    {"groove", "velHumanize",         &Weights::velHumanize},
    {"groove", "layOutLevel",         &Weights::layOutLevel},
    {"groove", "holdChange",          &Weights::grooveHoldChange},
    {"groove", "noteLengthSeconds",   &Weights::noteLengthSeconds},
    {"groove", "lateToleranceSteps",  &Weights::lateToleranceSteps},
    // fills
    {"fills", "minLevel",      &Weights::fillMinLevel},
    {"fills", "snareLevel",    &Weights::fillSnareLevel},
    {"fills", "tomLevel",      &Weights::fillTomLevel},
    {"fills", "longLevel",     &Weights::fillLevelLong},
    {"fills", "crashLevel",    &Weights::crashLevel},
    {"fills", "crashVelocity", &Weights::crashVelocity},
    // learned layer usage
    {"beatNet", "onsetMix",       &Weights::beatOnsetMix},
    {"beatNet", "onsetGain",      &Weights::beatOnsetGain},
    {"beatNet", "downbeatWeight", &Weights::beatDownbeatWeight},
    {"beatNet", "layOutBelow",    &Weights::beatLayOutBelow},
    {"beatNet", "layInAbove",     &Weights::beatLayInAbove},
    {"beatNet", "drumsDecay",     &Weights::beatDrumsDecay},
};

const IntField kIntFields[] = {
    {"transient", "hpssTimeFrames",     &Weights::hpssTimeFrames},
    {"transient", "hpssFreqBins",       &Weights::hpssFreqBins},
    {"onset",     "fluxMaxRadius",      &Weights::fluxMaxRadius},
    {"chord",     "harmonicSpanFrames", &Weights::harmonicSpanFrames},
    {"tempo",     "switchFrames",       &Weights::tempoSwitchFrames},
    {"tempo",     "settleFrames",       &Weights::tempoSettleFrames},
    {"tempo",     "settleSwitchFrames", &Weights::tempoSettleSwitchFrames},
    {"tempo",     "lockMinFrames",      &Weights::lockMinFrames},
    {"tempo",     "minSoundFrames",     &Weights::minSoundFrames},
    {"phase",     "templateBeats",      &Weights::templateBeats},
    {"phase",     "snapFrames",         &Weights::phaseSnapFrames},
    {"phase",     "settleSnapFrames",   &Weights::phaseSettleSnapFrames},
    {"groove",    "holdBars",           &Weights::grooveHoldBars},
    {"downbeat",  "accentPre",          &Weights::accentPre},
    {"downbeat",  "accentPost",         &Weights::accentPost},
    {"downbeat",  "switchBeats",        &Weights::downbeatSwitchBeats},
    {"downbeat",  "minUpdates",         &Weights::downbeatMinUpdates},
    {"intensity", "sectionLookbackFrames", &Weights::sectionLookbackFrames},
    {"intensity", "sectionCooldownFrames", &Weights::sectionCooldownFrames},
    {"intensity", "silenceFrames",      &Weights::silenceFrames},
    {"intensity", "resetSilenceFrames", &Weights::resetSilenceFrames},
    {"fills",     "startShort",         &Weights::fillStartShort},
    {"fills",     "startLong",          &Weights::fillStartLong},
    {"fills",     "minBarsPlayed",      &Weights::fillMinBarsPlayed},
};

// ===========================================================================
// Helpers
// ===========================================================================

/**
 * A float as the shortest decimal that survives a float round trip
 * (7 significant digits), so the file shows 0.023 instead of
 * 0.023000000044703484 and still reloads to the exact same weight.
 */
double tidy(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.7g", static_cast<double>(v));
    return std::strtod(buf, nullptr);
}

/// A float vector as tidy JSON numbers.
json tidyArray(const std::vector<float>& v) {
    json a = json::array();
    for (float x : v) {
        a.push_back(tidy(x));
    }
    return a;
}

/// One drum's 16-step column of a pattern matrix, as a JSON array.
json column(const Matrix& m, int drum) {
    json a = json::array();
    for (int s = 0; s < m.rows; ++s) {
        a.push_back(tidy(m.at(s, drum)));
    }
    return a;
}

/// Read a JSON array of exactly @p n numbers into @p out.
bool readArray(const json& j, size_t n, std::vector<float>& out, const std::string& where, std::string& err) {
    if (!j.is_array() || j.size() != n) {
        err = where + ": expected an array of " + std::to_string(n) + " numbers";
        return false;
    }
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (!j[i].is_number()) {
            err = where + "[" + std::to_string(i) + "]: not a number";
            return false;
        }
        out[i] = j[i].get<float>();
        if (!std::isfinite(out[i])) {
            err = where + "[" + std::to_string(i) + "]: not finite";
            return false;
        }
    }
    return true;
}

/// Read one drum column (16 numbers) into a pattern matrix.
bool readColumn(const json& j, Matrix& m, int drum, const std::string& where, std::string& err) {
    std::vector<float> v;
    if (!readArray(j, static_cast<size_t>(kStepsPerBar), v, where, err)) {
        return false;
    }
    for (int s = 0; s < kStepsPerBar; ++s) {
        m.at(s, drum) = v[static_cast<size_t>(s)];
    }
    return true;
}

/**
 * Pretty printer that keeps arrays of plain numbers on a single line, so a
 * 16-step drum pattern reads like a sequencer row instead of 16 lines.
 * Templated so it prints ordered_json (keys in signal-flow order).
 */
template <class J>
void writePretty(std::ostringstream& os, const J& j, int indent) {
    const std::string pad(static_cast<size_t>(indent), ' ');
    const std::string padIn(static_cast<size_t>(indent + 2), ' ');

    if (j.is_object()) {
        if (j.empty()) {
            os << "{}";
            return;
        }
        os << "{\n";
        size_t i = 0;
        for (auto it = j.begin(); it != j.end(); ++it, ++i) {
            os << padIn << J(it.key()).dump() << ": ";
            writePretty(os, it.value(), indent + 2);
            os << (i + 1 < j.size() ? ",\n" : "\n");
        }
        os << pad << "}";
    } else if (j.is_array()) {
        bool flat = true;
        for (const auto& e : j) {
            flat = flat && (e.is_number() || e.is_string() || e.is_boolean());
        }
        if (flat) {
            os << "[";
            for (size_t i = 0; i < j.size(); ++i) {
                os << (i ? ", " : "") << j[i].dump();
            }
            os << "]";
        } else {
            os << "[\n";
            for (size_t i = 0; i < j.size(); ++i) {
                os << padIn;
                writePretty(os, j[i], indent + 2);
                os << (i + 1 < j.size() ? ",\n" : "\n");
            }
            os << pad << "]";
        }
    } else {
        os << j.dump();
    }
}

} // namespace

// ===========================================================================
// Save
// ===========================================================================

bool saveWeightsFile(const std::string& path, const Weights& w, std::string* error) {
    // nlohmann::ordered_json keeps keys in insertion order, so the file reads
    // top to bottom in the same order as the signal flows.
    nlohmann::ordered_json j;
    j["format"]  = "drum_along.weights";
    j["version"] = kWeightsFileVersion;
    j["about"]   = "Hand-written weights of the drum_along drummer. Edit any value and restart: "
                   "no recompiling. Groove: each drum has 16 sixteenth-note steps; a step plays when "
                   "on <= intensity < off (on = 2 means never); velocity is the hit at intensity 0.5. "
                   "Fill patterns hold velocities, 0 = silent. Decays are per-frame (100 frames/s).";

    for (const FloatField& f : kFloatFields) {
        j[f.group][f.key] = tidy(w.*(f.member));
    }
    for (const IntField& f : kIntFields) {
        j[f.group][f.key] = w.*(f.member);
    }

    // ---- Arrays --------------------------------------------------------------
    j["onset"]["fluxReadout"]  = tidyArray(w.fluxReadout);
    j["onset"]["smoothKernel"] = {tidy(w.smoothKernel[0]), tidy(w.smoothKernel[1]), tidy(w.smoothKernel[2])};
    j["tempo"]["priorFirstPeriodFrames"] = kMinPeriodFrames;
    j["tempo"]["prior"] = tidyArray(w.tempoPrior);

    // ---- Learned layer: only the file name; its numbers live in that file ----
    j["beatNet"]["file"] = w.beatNetFile;

    // ---- Groove patterns, one block per drum ----------------------------------
    for (int d = 0; d < kNumDrums; ++d) {
        auto& drum = j["groove"]["drums"][kDrumNames[d]];
        drum["note"]     = kDrumNotes[d];
        drum["on"]       = column(w.hitOnLevel, d);
        drum["off"]      = column(w.hitOffLevel, d);
        drum["velocity"] = column(w.velocity, d);
    }

    // ---- Fill patterns: only drums that play in the fill -----------------------
    for (int v = 0; v < kNumFillVariants && v < static_cast<int>(w.fillVelocity.size()); ++v) {
        auto& fill = j["fills"]["patterns"][kFillNames[v]];
        fill = nlohmann::ordered_json::object();
        for (int d = 0; d < kNumDrums; ++d) {
            const json col = column(w.fillVelocity[static_cast<size_t>(v)], d);
            bool any = false;
            for (const json& x : col) {
                any = any || x.get<float>() != 0.0f;
            }
            if (any) {
                fill[kDrumNames[d]] = col;
            }
        }
    }

    // ---- Write -------------------------------------------------------------------
    std::ostringstream os;
    writePretty(os, j, 0);
    os << "\n";

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        if (error) *error = "cannot open " + path + " for writing";
        return false;
    }
    const std::string text = os.str();
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!f) {
        if (error) *error = "write failed for " + path;
        return false;
    }
    return true;
}

// ===========================================================================
// Load
// ===========================================================================

/**
 * @brief Path of a file named from inside a weights file.
 * @param dir  Folder of the file doing the naming, with its trailing slash (may be empty).
 * @param name The name as written: relative to @p dir, or absolute.
 * @return @p name unchanged when it is absolute, otherwise @p dir followed by @p name.
 */
static std::string resolveNamedFile(const std::string& dir, const std::string& name) {
    return std::filesystem::path(name).is_absolute() ? name : dir + name;
}

namespace {
bool loadWeightsFileImpl(const std::string& path, Weights& w, std::string* error, int depth);
}

bool loadWeightsFile(const std::string& path, Weights& w, std::string* error) {
    return loadWeightsFileImpl(path, w, error, 0);
}

namespace {

bool loadWeightsFileImpl(const std::string& path, Weights& w, std::string* error, int depth) {
    std::string err;
    auto failWith = [&](const std::string& message) {
        if (error && error->empty()) *error = path + ": " + message;
        return false;
    };
    if (depth > 4) {
        return failWith("\"extends\" chain deeper than 4 files (a file extending itself?)");
    }
    const size_t slash = path.find_last_of("/\\");
    const std::string dir = (slash == std::string::npos) ? std::string() : path.substr(0, slash + 1);

    // ---- Parse ---------------------------------------------------------------------
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return failWith("cannot open file");
    }
    json j;
    try {
        j = json::parse(f, nullptr, true, /*ignore_comments=*/true);
    } catch (const std::exception& e) {
        return failWith(std::string("invalid JSON: ") + e.what());
    }
    if (!j.is_object() || j.value("format", std::string()) != "drum_along.weights") {
        return failWith("not a drum_along weights file (format tag missing)");
    }
    if (j.value("version", 0) > kWeightsFileVersion) {
        return failWith("file is from a newer version of drum_along");
    }

    // Work on a copy so a half-read file never leaks into the caller.
    Weights out = w;

    // ---- "extends": load the parent file first, then overlay this one --------
    // Style files use this to change only the groove and inherit everything else.
    if (j.contains("extends")) {
        if (!j["extends"].is_string()) {
            return failWith("extends: expected a file name");
        }
        if (!loadWeightsFileImpl(resolveNamedFile(dir, j["extends"].get<std::string>()), out, error, depth + 1)) {
            return false;
        }
    }

    // ---- Scalars -------------------------------------------------------------------
    for (const FloatField& fld : kFloatFields) {
        if (!j.contains(fld.group) || !j[fld.group].contains(fld.key)) {
            continue;   // Missing keys keep their current value.
        }
        const json& v = j[fld.group][fld.key];
        if (!v.is_number() || !std::isfinite(v.get<double>())) {
            return failWith(std::string(fld.group) + "." + fld.key + ": expected a finite number");
        }
        out.*(fld.member) = v.get<float>();
    }
    for (const IntField& fld : kIntFields) {
        if (!j.contains(fld.group) || !j[fld.group].contains(fld.key)) {
            continue;
        }
        const json& v = j[fld.group][fld.key];
        if (!v.is_number_integer()) {
            return failWith(std::string(fld.group) + "." + fld.key + ": expected an integer");
        }
        out.*(fld.member) = v.get<int>();
    }

    // ---- Arrays --------------------------------------------------------------------
    if (j.contains("onset") && j["onset"].contains("fluxReadout")) {
        if (!readArray(j["onset"]["fluxReadout"], kMelBands, out.fluxReadout, "onset.fluxReadout", err)) return failWith(err);
    }
    if (j.contains("onset") && j["onset"].contains("smoothKernel")) {
        std::vector<float> k;
        if (!readArray(j["onset"]["smoothKernel"], 3, k, "onset.smoothKernel", err)) return failWith(err);
        for (int i = 0; i < 3; ++i) out.smoothKernel[i] = k[static_cast<size_t>(i)];
    }
    if (j.contains("tempo") && j["tempo"].contains("prior")) {
        if (!readArray(j["tempo"]["prior"], kNumPeriods, out.tempoPrior, "tempo.prior", err)) return failWith(err);
    }

    // ---- Groove patterns -------------------------------------------------------------
    if (j.contains("groove") && j["groove"].contains("drums")) {
        const json& drums = j["groove"]["drums"];
        for (int d = 0; d < kNumDrums; ++d) {
            if (!drums.contains(kDrumNames[d])) continue;
            const json& dj = drums[kDrumNames[d]];
            const std::string where = std::string("groove.drums.") + kDrumNames[d];
            if (dj.contains("on")       && !readColumn(dj["on"],       out.hitOnLevel,  d, where + ".on", err))       return failWith(err);
            if (dj.contains("off")      && !readColumn(dj["off"],      out.hitOffLevel, d, where + ".off", err))      return failWith(err);
            if (dj.contains("velocity") && !readColumn(dj["velocity"], out.velocity,    d, where + ".velocity", err)) return failWith(err);
        }
    }

    // ---- Fill patterns: a listed fill replaces its whole table ----------------------
    if (j.contains("fills") && j["fills"].contains("patterns")) {
        const json& pats = j["fills"]["patterns"];
        for (int v = 0; v < kNumFillVariants; ++v) {
            if (!pats.contains(kFillNames[v])) continue;
            Matrix table;
            table.resize(kStepsPerBar, kNumDrums);
            const json& pj = pats[kFillNames[v]];
            for (int d = 0; d < kNumDrums; ++d) {
                if (!pj.contains(kDrumNames[d])) continue;
                const std::string where = std::string("fills.patterns.") + kFillNames[v] + "." + kDrumNames[d];
                if (!readColumn(pj[kDrumNames[d]], table, d, where, err)) return failWith(err);
            }
            out.fillVelocity[static_cast<size_t>(v)] = std::move(table);
        }
    }

    // ---- Learned layer: load the named file, relative to this one ----------------
    if (j.contains("beatNet") && j["beatNet"].contains("file")) {
        if (!j["beatNet"]["file"].is_string()) {
            return failWith("beatNet.file: expected a string");
        }
        out.beatNetFile = j["beatNet"]["file"].get<std::string>();
    }
    // Only (re)load the network when this file names one; otherwise keep what
    // a parent file loaded. The name resolves against the file that wrote it.
    if (j.contains("beatNet") && j["beatNet"].contains("file")) {
        out.beatNet = BeatNetWeights{};
        if (!out.beatNetFile.empty()) {
            std::string netError;
            if (!loadBeatNetFile(resolveNamedFile(dir, out.beatNetFile), out.beatNet, &netError)) {
                return failWith(netError);
            }
        }
    }

    w = std::move(out);
    return true;
}

} // namespace

// ===========================================================================
// Learned layer file
// ===========================================================================

bool loadBeatNetFile(const std::string& path, BeatNetWeights& net, std::string* error) {
    std::string err;
    auto failWith = [&](const std::string& message) {
        if (error) *error = path + ": " + message;
        return false;
    };

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return failWith("cannot open file");
    }
    json j;
    try {
        j = json::parse(f);
    } catch (const std::exception& e) {
        return failWith(std::string("invalid JSON: ") + e.what());
    }
    if (!j.is_object() || j.value("format", std::string()) != "drum_along.beatnet") {
        return failWith("not a drum_along beatnet file");
    }
    if (j.value("numFeatures", 0) != kBeatFeatures || j.value("numOutputs", 0) != kNumBeatNetOutputs) {
        return failWith("trained for a different feature set or output count");
    }

    BeatNetWeights out;
    out.channels = j.value("channels", 0);
    if (out.channels <= 0 || out.channels > 256) {
        return failWith("channels must be in [1, 256]");
    }
    const size_t C = static_cast<size_t>(out.channels);

    if (!j.contains("dilations") || !j["dilations"].is_array()) {
        return failWith("dilations missing");
    }
    for (const json& d : j["dilations"]) {
        if (!d.is_number_integer()) return failWith("dilations must be integers");
        out.dilations.push_back(d.get<int>());
    }
    const size_t B = out.dilations.size();

    auto need = [&](const char* key, size_t n, std::vector<float>& dst) {
        if (!j.contains(key)) {
            err = std::string(key) + " missing";
            return false;
        }
        return readArray(j[key], n, dst, key, err);
    };
    if (!need("featMean", kBeatFeatures, out.featMean) || !need("featStd", kBeatFeatures, out.featStd) ||
        !need("inW", C * kBeatFeatures, out.inW) || !need("inB", C, out.inB) ||
        !need("headW", C * kNumBeatNetOutputs, out.headW) || !need("headB", kNumBeatNetOutputs, out.headB)) {
        return failWith(err);
    }

    if (!j.contains("blocks") || !j["blocks"].is_array() || j["blocks"].size() != B) {
        return failWith("blocks must list one entry per dilation");
    }
    out.blockW.resize(B);
    out.blockB.resize(B);
    for (size_t b = 0; b < B; ++b) {
        const json& bj = j["blocks"][b];
        const std::string where = "blocks[" + std::to_string(b) + "]";
        if (!bj.contains("w") || !readArray(bj["w"], C * C * 3, out.blockW[b], where + ".w", err)) return failWith(err.empty() ? where + ".w missing" : err);
        if (!bj.contains("b") || !readArray(bj["b"], C, out.blockB[b], where + ".b", err))         return failWith(err.empty() ? where + ".b missing" : err);
    }

    if (!out.valid()) {
        return failWith("weights failed the shape check");
    }
    net = std::move(out);
    return true;
}

} // namespace drumalong
