/**
 * @file controller.cpp
 * @brief VST3 edit controller implementation.
 */

#include "controller.h"
#include "pluginids.h"

#include "base/source/fstreamer.h"
#include "pluginterfaces/base/ibstream.h"
#include "public.sdk/source/vst/vstparameters.h"

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace drumalong::vst {

tresult PLUGIN_API DrumAlongController::initialize(FUnknown* context) {
    const tresult result = EditController::initialize(context);
    if (result != kResultOk) {
        return result;
    }

    // Play: on / off toggle. Off mutes the drums but keeps listening.
    parameters.addParameter(STR16("Play"), nullptr, 1, 1.0, ParameterInfo::kCanAutomate, kParamPlay);

    // Style: which groove the drummer plays. Switching never loses the beat.
    auto* style = new StringListParameter(STR16("Style"), kParamStyle, nullptr,
                                          ParameterInfo::kCanAutomate | ParameterInfo::kIsList);
    style->appendString(STR16("Rock"));
    style->appendString(STR16("Alt Rock"));
    style->appendString(STR16("Pop"));
    style->appendString(STR16("Jazz"));
    parameters.addParameter(style);

    // Timing offset: compensates the drum instrument's own latency.
    parameters.addParameter(new RangeParameter(STR16("Timing Offset"), kParamOffset, STR16("ms"),
                                               -50.0, 50.0, 0.0, 0, ParameterInfo::kCanAutomate));

    // MIDI channel for the drum notes (GM drums live on channel 10).
    parameters.addParameter(new RangeParameter(STR16("MIDI Channel"), kParamChannel, nullptr,
                                               1.0, 16.0, 10.0, 15, ParameterInfo::kCanAutomate));
    return kResultOk;
}

tresult PLUGIN_API DrumAlongController::setComponentState(IBStream* state) {
    // Mirror the processor's saved state into the displayed parameters.
    if (!state) {
        return kResultFalse;
    }
    IBStreamer s(state, kLittleEndian);
    double v[kNumParams] = {};
    for (double& x : v) {
        if (!s.readDouble(x)) {
            return kResultFalse;
        }
    }
    setParamNormalized(kParamPlay, v[0]);
    setParamNormalized(kParamStyle, v[1]);
    setParamNormalized(kParamOffset, v[2]);
    setParamNormalized(kParamChannel, v[3]);
    return kResultOk;
}

} // namespace drumalong::vst
