#pragma once
/**
 * @file processor.h
 * @brief VST3 audio processor: band audio in, band audio out (pass-through),
 *        drum MIDI out.
 *
 * Real-time rules: everything that allocates (loading weights, preparing the
 * engine, adding styles) happens in setActive(), which hosts call off the
 * audio thread. process() only reads parameters, copies audio, runs the
 * engine and writes events.
 */

#include "drumalong/DrumAlongEngine.h"

#include "public.sdk/source/vst/vstaudioeffect.h"

#include <memory>
#include <vector>

namespace drumalong::vst {

class DrumAlongProcessor : public Steinberg::Vst::AudioEffect {
public:
    /**
     * @param instrument false: effect with main audio in -> audio out (pass-through).
     *                   true:  instrument with a sidechain (aux) audio in and a
     *                          silent audio out, for hosts such as Ableton Live.
     */
    explicit DrumAlongProcessor(bool instrument = false);

    static Steinberg::FUnknown* createInstance(void*) {
        return static_cast<Steinberg::Vst::IAudioProcessor*>(new DrumAlongProcessor(false));
    }
    static Steinberg::FUnknown* createInstrumentInstance(void*) {
        return static_cast<Steinberg::Vst::IAudioProcessor*>(new DrumAlongProcessor(true));
    }

    // ---- IComponent / IAudioProcessor ---------------------------------------
    Steinberg::tresult PLUGIN_API initialize(Steinberg::FUnknown* context) override;
    Steinberg::tresult PLUGIN_API setBusArrangements(Steinberg::Vst::SpeakerArrangement* inputs, Steinberg::int32 numIns,
                                                     Steinberg::Vst::SpeakerArrangement* outputs, Steinberg::int32 numOuts) override;
    Steinberg::tresult PLUGIN_API canProcessSampleSize(Steinberg::int32 symbolicSampleSize) override;
    Steinberg::tresult PLUGIN_API setupProcessing(Steinberg::Vst::ProcessSetup& setup) override;
    Steinberg::tresult PLUGIN_API setActive(Steinberg::TBool state) override;
    Steinberg::tresult PLUGIN_API process(Steinberg::Vst::ProcessData& data) override;
    Steinberg::uint32  PLUGIN_API getLatencySamples() override { return 0; }   // hits are predicted, not delayed
    Steinberg::tresult PLUGIN_API setState(Steinberg::IBStream* state) override;
    Steinberg::tresult PLUGIN_API getState(Steinberg::IBStream* state) override;

private:
    void applyParameter(Steinberg::Vst::ParamID id, double value);
    void prepareEngine();

    const bool instrument_;                     ///< Sidechain-input instrument variant.
    std::unique_ptr<DrumAlongEngine> engine_;   ///< Built in setActive(true).
    std::vector<MidiEvent>           midi_;     ///< Output scratch, sized in setActive.
    double sampleRate_   = 44100.0;
    int    maxBlock_     = 1024;

    // Parameter values (normalised, as the host sends them).
    double playNorm_    = 1.0;
    double styleNorm_   = 0.0;
    double offsetNorm_  = 0.5;    // 0 ms
    double channelNorm_ = 9.0 / 15.0;   // MIDI channel 10
};

} // namespace drumalong::vst
