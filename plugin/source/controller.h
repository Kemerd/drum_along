#pragma once
/**
 * @file controller.h
 * @brief VST3 edit controller: the parameters the host shows and automates.
 *
 * No custom editor yet: hosts draw their generic parameter panel.
 */

#include "public.sdk/source/vst/vsteditcontroller.h"

namespace drumalong::vst {

class DrumAlongController : public Steinberg::Vst::EditController {
public:
    static Steinberg::FUnknown* createInstance(void*) {
        return static_cast<Steinberg::Vst::IEditController*>(new DrumAlongController);
    }

    Steinberg::tresult PLUGIN_API initialize(Steinberg::FUnknown* context) override;
    Steinberg::tresult PLUGIN_API setComponentState(Steinberg::IBStream* state) override;
    Steinberg::IPlugView* PLUGIN_API createView(Steinberg::FIDString) override { return nullptr; }
};

} // namespace drumalong::vst
