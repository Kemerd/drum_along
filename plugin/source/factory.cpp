/**
 * @file factory.cpp
 * @brief Registers the processor and controller classes with the host.
 *
 * Category "Fx|Generator": an effect that generates events. Hosts that
 * support event outputs on effects (Reaper, Bitwig, Cubase, Studio One)
 * route its MIDI to any drum instrument.
 */

#include "controller.h"
#include "pluginids.h"
#include "processor.h"
#include "version.h"

#include "public.sdk/source/main/pluginfactory.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"

using namespace Steinberg;
using namespace Steinberg::Vst;
using namespace drumalong::vst;

BEGIN_FACTORY_DEF(stringCompanyName, stringCompanyWeb, stringCompanyEmail)

DEF_CLASS2(INLINE_UID_FROM_FUID(kProcessorUID), PClassInfo::kManyInstances, kVstAudioEffectClass,
           stringPluginName " (FX)", Vst::kDistributable, "Fx|Generator", FULL_VERSION_STR, kVstVersionString,
           DrumAlongProcessor::createInstance)

// Instrument variant: sits on a MIDI track and hears the band via sidechain.
// This is the class to use in Ableton Live.
DEF_CLASS2(INLINE_UID_FROM_FUID(kInstrumentUID), PClassInfo::kManyInstances, kVstAudioEffectClass,
           stringPluginName " (Instrument)", Vst::kDistributable, "Instrument|Generator", FULL_VERSION_STR,
           kVstVersionString, DrumAlongProcessor::createInstrumentInstance)

DEF_CLASS2(INLINE_UID_FROM_FUID(kControllerUID), PClassInfo::kManyInstances, kVstComponentControllerClass,
           stringPluginName "Controller", 0, "", FULL_VERSION_STR, kVstVersionString,
           DrumAlongController::createInstance)

END_FACTORY
