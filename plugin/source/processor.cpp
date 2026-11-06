/**
 * @file processor.cpp
 * @brief VST3 processor implementation.
 */

#include "processor.h"
#include "pluginids.h"

#include "base/source/fstreamer.h"
#include "pluginterfaces/base/ibstream.h"
#include "pluginterfaces/vst/ivstevents.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"

#include <algorithm>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace drumalong::vst {

namespace {

/**
 * @brief Folder that holds the bundled weights:
 *        drum_along.vst3/Contents/Resources/weights/
 *
 * The binary lives in drum_along.vst3/Contents/x86_64-win/, so the resources
 * are one level up. Returns an empty string if the path cannot be found; the
 * engine then falls back to its built-in hand-written weights.
 */
std::string bundledWeightsDir() {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&bundledWeightsDir), &module)) {
        return {};
    }
    char path[MAX_PATH * 4] = {};
    const DWORD n = GetModuleFileNameA(module, path, static_cast<DWORD>(sizeof(path)));
    if (n == 0 || n >= sizeof(path)) {
        return {};
    }
    std::string p(path, n);
    const size_t binDir = p.find_last_of("\\/");
    if (binDir == std::string::npos) {
        return {};
    }
    const size_t contents = p.find_last_of("\\/", binDir - 1);
    if (contents == std::string::npos) {
        return {};
    }
    return p.substr(0, contents + 1) + "Resources\\weights\\";
#else
    return {};
#endif
}

bool fileExists(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f) {
        std::fclose(f);
        return true;
    }
    return false;
}

} // namespace

// ===========================================================================
// Setup
// ===========================================================================

DrumAlongProcessor::DrumAlongProcessor(bool instrument) : instrument_(instrument) {
    setControllerClass(kControllerUID);
}

/// Adds the audio buses for this variant with the given input arrangement.
static void addBuses(AudioEffect& fx, bool instrument, SpeakerArrangement in, SpeakerArrangement out) {
    if (instrument) {
        // The band arrives on a sidechain; the audio output stays silent.
        fx.addAudioInput(STR16("Band (Sidechain)"), in, BusTypes::kAux);
        fx.addAudioOutput(STR16("Out"), out);
    } else {
        fx.addAudioInput(STR16("Band In"), in);
        fx.addAudioOutput(STR16("Band Out"), out);
    }
}

tresult PLUGIN_API DrumAlongProcessor::initialize(FUnknown* context) {
    const tresult result = AudioEffect::initialize(context);
    if (result != kResultOk) {
        return result;
    }
    addBuses(*this, instrument_, SpeakerArr::kStereo, SpeakerArr::kStereo);
    addEventOutput(STR16("Drums MIDI Out"), 16);
    return kResultOk;
}

tresult PLUGIN_API DrumAlongProcessor::setBusArrangements(SpeakerArrangement* inputs, int32 numIns,
                                                          SpeakerArrangement* outputs, int32 numOuts) {
    if (numIns != 1 || numOuts != 1 || !inputs || !outputs) {
        return kResultFalse;
    }
    const bool inOk  = inputs[0] == SpeakerArr::kMono || inputs[0] == SpeakerArr::kStereo;
    const bool outOk = outputs[0] == SpeakerArr::kMono || outputs[0] == SpeakerArr::kStereo;
    // The effect is a pass-through, so its output must match its input; the
    // instrument's output is silent and may differ from its sidechain.
    if (!inOk || !outOk || (!instrument_ && inputs[0] != outputs[0])) {
        return kResultFalse;
    }
    removeAudioBusses();
    addBuses(*this, instrument_, inputs[0], outputs[0]);
    return kResultTrue;
}

tresult PLUGIN_API DrumAlongProcessor::canProcessSampleSize(int32 symbolicSampleSize) {
    return symbolicSampleSize == kSample32 ? kResultTrue : kResultFalse;
}

tresult PLUGIN_API DrumAlongProcessor::setupProcessing(ProcessSetup& setup) {
    sampleRate_ = setup.sampleRate;
    maxBlock_   = std::max<int32>(1, setup.maxSamplesPerBlock);
    return AudioEffect::setupProcessing(setup);
}

/**
 * Build the engine for the current rate. Runs off the audio thread.
 * Weights: the bundled weights/drummer.json (+ beatnet.json it names) when
 * present, otherwise the built-in hand-written defaults. Styles are loaded
 * from weights/styles/ when present.
 */
void DrumAlongProcessor::prepareEngine() {
    auto engine = std::make_unique<DrumAlongEngine>();
    const std::string dir = bundledWeightsDir();
    const std::string drummer = dir + "drummer.json";

    std::string error;
    bool ok = fileExists(drummer) && engine->prepare(sampleRate_, maxBlock_, drummer, &error);
    if (!ok) {
        ok = engine->prepare(sampleRate_, maxBlock_);   // built-in defaults
    }
    if (!ok) {
        engine_.reset();
        return;
    }
    for (int s = 1; s < kNumStyles; ++s) {
        const std::string style = dir + "styles\\" + kStyleFiles[s];
        if (!fileExists(style) || engine->addStyle(style, &error) < 0) {
            break;   // Missing styles simply are not offered beyond this point.
        }
    }

    engine_ = std::move(engine);
    midi_.assign(4096, MidiEvent{});
    applyParameter(kParamPlay, playNorm_);
    applyParameter(kParamStyle, styleNorm_);
    applyParameter(kParamOffset, offsetNorm_);
}

tresult PLUGIN_API DrumAlongProcessor::setActive(TBool state) {
    if (state) {
        prepareEngine();
    } else if (engine_) {
        engine_->reset();
    }
    return AudioEffect::setActive(state);
}

// ===========================================================================
// Parameters
// ===========================================================================

void DrumAlongProcessor::applyParameter(ParamID id, double value) {
    value = std::clamp(value, 0.0, 1.0);
    switch (id) {
        case kParamPlay:
            playNorm_ = value;
            if (engine_) engine_->setPlaying(value > 0.5);
            break;
        case kParamStyle:
            styleNorm_ = value;
            if (engine_) engine_->setStyle(std::min(styleFromNorm(value), engine_->numStyles() - 1));
            break;
        case kParamOffset:
            offsetNorm_ = value;
            if (engine_) engine_->setOutputOffsetMs(static_cast<float>(offsetFromNorm(value)));
            break;
        case kParamChannel:
            channelNorm_ = value;
            break;
        default:
            break;
    }
}

// ===========================================================================
// Audio thread
// ===========================================================================

tresult PLUGIN_API DrumAlongProcessor::process(ProcessData& data) {
    // ---- 1. Parameter changes: the last value of each queue in this block ----
    if (IParameterChanges* changes = data.inputParameterChanges) {
        const int32 count = changes->getParameterCount();
        for (int32 i = 0; i < count; ++i) {
            IParamValueQueue* queue = changes->getParameterData(i);
            if (!queue || queue->getPointCount() <= 0) {
                continue;
            }
            int32      offset = 0;
            ParamValue value  = 0.0;
            if (queue->getPoint(queue->getPointCount() - 1, offset, value) == kResultTrue) {
                applyParameter(queue->getParameterId(), value);
            }
        }
    }

    if (data.numSamples <= 0 || data.numInputs < 1 || data.numOutputs < 1) {
        return kResultOk;   // parameter flush
    }

    AudioBusBuffers& in  = data.inputs[0];
    AudioBusBuffers& out = data.outputs[0];
    const int32 n   = data.numSamples;
    const int32 chs = std::min(in.numChannels, out.numChannels);

    // ---- 2. Audio out: the band (effect) or silence (instrument) -------------------
    if (instrument_ && out.channelBuffers32) {
        for (int32 c = 0; c < out.numChannels; ++c) {
            if (out.channelBuffers32[c]) {
                std::memset(out.channelBuffers32[c], 0, sizeof(float) * static_cast<size_t>(n));
            }
        }
        out.silenceFlags = (out.numChannels >= 64) ? ~uint64(0) : ((uint64(1) << out.numChannels) - 1);
    } else if (in.channelBuffers32 && out.channelBuffers32) {
        for (int32 c = 0; c < chs; ++c) {
            if (in.channelBuffers32[c] && out.channelBuffers32[c] && in.channelBuffers32[c] != out.channelBuffers32[c]) {
                std::memcpy(out.channelBuffers32[c], in.channelBuffers32[c], sizeof(float) * static_cast<size_t>(n));
            }
        }
        for (int32 c = chs; c < out.numChannels; ++c) {
            if (out.channelBuffers32[c]) {
                std::memset(out.channelBuffers32[c], 0, sizeof(float) * static_cast<size_t>(n));
            }
        }
        out.silenceFlags = in.silenceFlags;
    }

    // ---- 3. Listen and play ------------------------------------------------------
    if (!engine_ || !in.channelBuffers32) {
        return kResultOk;
    }
    const int count = engine_->process(in.channelBuffers32, in.numChannels, n,
                                       midi_.data(), static_cast<int>(midi_.size()));

    // ---- 4. Drum notes out -----------------------------------------------------------
    if (IEventList* events = data.outputEvents) {
        const int16 channel = static_cast<int16>(channelFromNorm(channelNorm_));
        for (int i = 0; i < count; ++i) {
            const MidiEvent& m = midi_[static_cast<size_t>(i)];
            Event e{};
            e.busIndex     = 0;
            e.sampleOffset = std::clamp<int32>(m.sampleOffset, 0, n - 1);
            e.flags        = Event::kIsLive;
            if (m.noteOn) {
                e.type             = Event::kNoteOnEvent;
                e.noteOn.channel   = channel;
                e.noteOn.pitch     = m.note;
                e.noteOn.velocity  = m.velocity / 127.0f;
                e.noteOn.noteId    = -1;
            } else {
                e.type             = Event::kNoteOffEvent;
                e.noteOff.channel  = channel;
                e.noteOff.pitch    = m.note;
                e.noteOff.velocity = 0.0f;
                e.noteOff.noteId   = -1;
            }
            events->addEvent(e);
        }
    }
    return kResultOk;
}

// ===========================================================================
// State: the four parameters, normalised, little endian
// ===========================================================================

tresult PLUGIN_API DrumAlongProcessor::setState(IBStream* state) {
    if (!state) {
        return kResultFalse;
    }
    IBStreamer s(state, kLittleEndian);
    double v[kNumParams] = {playNorm_, styleNorm_, offsetNorm_, channelNorm_};
    for (double& x : v) {
        if (!s.readDouble(x)) {
            return kResultFalse;
        }
    }
    applyParameter(kParamPlay, v[0]);
    applyParameter(kParamStyle, v[1]);
    applyParameter(kParamOffset, v[2]);
    applyParameter(kParamChannel, v[3]);
    return kResultOk;
}

tresult PLUGIN_API DrumAlongProcessor::getState(IBStream* state) {
    if (!state) {
        return kResultFalse;
    }
    IBStreamer s(state, kLittleEndian);
    s.writeDouble(playNorm_);
    s.writeDouble(styleNorm_);
    s.writeDouble(offsetNorm_);
    s.writeDouble(channelNorm_);
    return kResultOk;
}

} // namespace drumalong::vst
