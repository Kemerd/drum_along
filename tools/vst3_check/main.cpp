/**
 * @file main.cpp
 * @brief Minimal VST3 host that plays a WAV through the real drum_along.vst3
 *        and records every MIDI event the plug-in emits.
 *
 * Usage:
 *   vst3_check <drum_along.vst3> <input.wav> <out_events.csv> [--block N]
 *
 * It loads the bundle exactly as a DAW does (module -> factory -> component),
 * negotiates buses, activates, and calls process() block by block with real
 * audio. The CSV has the same format as wav2mid's *_events.csv, so the two
 * can be compared event for event.
 */

#include "../common/WavIO.h"

#include "public.sdk/source/vst/hosting/eventlist.h"
#include "public.sdk/source/vst/hosting/hostclasses.h"
#include "public.sdk/source/vst/hosting/module.h"
#include "public.sdk/source/vst/hosting/plugprovider.h"
#include "public.sdk/source/vst/hosting/processdata.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivstcomponent.h"
#include "pluginterfaces/vst/ivstevents.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace Steinberg;
using namespace Steinberg::Vst;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: vst3_check <drum_along.vst3> <input.wav> <out_events.csv> [--block N]\n");
        return 2;
    }
    const std::string pluginPath = argv[1];
    const std::string wavPath    = argv[2];
    const std::string outPath    = argv[3];
    int block = 512;
    std::string wantClass = "FX";
    for (int i = 4; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--block") {
            block = std::clamp(std::atoi(argv[i + 1]), 16, 8192);
        } else if (std::string(argv[i]) == "--class") {
            wantClass = argv[i + 1];   // "FX" or "Instrument"
        }
    }

    // ---- Audio -------------------------------------------------------------------
    drumalong::tools::WavData wav;
    std::string err;
    if (!drumalong::tools::readWav(wavPath, wav, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    const int channels = std::min(wav.channels, 2);

    // ---- Load the plug-in like a host ----------------------------------------------
    HostApplication host;
    PluginContextFactory::instance().setPluginContext(&host);

    auto module = VST3::Hosting::Module::create(pluginPath, err);
    if (!module) {
        std::fprintf(stderr, "error: cannot load %s: %s\n", pluginPath.c_str(), err.c_str());
        return 1;
    }
    auto factory = module->getFactory();
    IPtr<PlugProvider> provider;
    for (auto& info : factory.classInfos()) {
        if (info.category() == kVstAudioEffectClass && info.name().find(wantClass) != std::string::npos) {
            provider = owned(new PlugProvider(factory, info, true));
            std::printf("plug-in    %s (%s)\n", info.name().c_str(), info.subCategoriesString().c_str());
            break;
        }
    }
    // getComponent() creates and initialises the plug-in on first use and
    // returns an added reference, hence owned().
    IPtr<IComponent> component = provider ? owned(provider->getComponent()) : nullptr;
    if (!component) {
        std::fprintf(stderr, "error: could not create the audio effect in %s\n", pluginPath.c_str());
        return 1;
    }
    FUnknownPtr<IAudioProcessor> processor(component);
    if (!processor) {
        std::fprintf(stderr, "error: component is not an IAudioProcessor\n");
        return 1;
    }

    // ---- Buses, setup, activation ------------------------------------------------------
    SpeakerArrangement arr = (channels == 1) ? SpeakerArr::kMono : SpeakerArr::kStereo;
    if (processor->setBusArrangements(&arr, 1, &arr, 1) != kResultTrue) {
        std::fprintf(stderr, "error: plug-in refused the bus arrangement\n");
        return 1;
    }
    component->activateBus(kAudio, kInput, 0, true);
    component->activateBus(kAudio, kOutput, 0, true);
    component->activateBus(kEvent, kOutput, 0, true);

    ProcessSetup setup{kOffline, kSample32, block, wav.sampleRate};
    processor->setupProcessing(setup);
    component->setActive(true);
    processor->setProcessing(true);

    HostProcessData data;
    data.prepare(*component, block, kSample32);
    EventList outEvents(4096);
    data.outputEvents = &outEvents;

    // ---- Stream the song block by block ---------------------------------------------------
    struct Rec { double t; int note; int vel; int on; };
    std::vector<Rec> recs;
    for (size_t start = 0; start < wav.frames; start += static_cast<size_t>(block)) {
        const int n = static_cast<int>(std::min<size_t>(block, wav.frames - start));
        data.numSamples = n;
        for (int c = 0; c < data.inputs[0].numChannels; ++c) {
            const auto& src = wav.data[static_cast<size_t>(std::min(c, wav.channels - 1))];
            std::copy(src.begin() + static_cast<std::ptrdiff_t>(start),
                      src.begin() + static_cast<std::ptrdiff_t>(start + n), data.inputs[0].channelBuffers32[c]);
        }
        outEvents.clear();
        if (processor->process(data) != kResultOk) {
            std::fprintf(stderr, "error: process() failed at sample %zu\n", start);
            return 1;
        }
        for (int32 i = 0; i < outEvents.getEventCount(); ++i) {
            Event e{};
            if (outEvents.getEvent(i, e) != kResultOk) {
                continue;
            }
            const double t = (static_cast<double>(start) + e.sampleOffset) / wav.sampleRate;
            if (e.type == Event::kNoteOnEvent) {
                recs.push_back({t, e.noteOn.pitch, static_cast<int>(std::lround(e.noteOn.velocity * 127.0f)), 1});
            } else if (e.type == Event::kNoteOffEvent) {
                recs.push_back({t, e.noteOff.pitch, 0, 0});
            }
        }
    }

    processor->setProcessing(false);
    component->setActive(false);
    data.unprepare();

    // ---- Write the same CSV format as wav2mid ----------------------------------------------
    FILE* f = std::fopen(outPath.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "error: cannot write %s\n", outPath.c_str());
        return 1;
    }
    std::fprintf(f, "time_s,note,velocity,on\n");
    int ons = 0;
    for (const Rec& r : recs) {
        std::fprintf(f, "%.5f,%d,%d,%d\n", r.t, r.note, r.vel, r.on);
        ons += r.on;
    }
    std::fclose(f);
    std::printf("events     %zu (%d note-ons) -> %s\n", recs.size(), ons, outPath.c_str());
    return 0;
}
