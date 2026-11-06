#pragma once
/**
 * @file pluginids.h
 * @brief Class IDs and parameter IDs of the drum_along VST3 plug-in.
 *
 * The class IDs are permanent: hosts store them in projects, so they must
 * never change once a build has been shared.
 */

#include "pluginterfaces/base/funknown.h"
#include "pluginterfaces/vst/vsttypes.h"

namespace drumalong::vst {

/// "drum_along (FX)": on the band's audio track, audio passes through.
static const Steinberg::FUID kProcessorUID(0x533599E4, 0x54514708, 0xAD157F24, 0x1C1DD339);
/// "drum_along (Instrument)": on a MIDI track, hears the band via a sidechain
/// input. This is the one Ableton Live needs, since Live only routes plug-in
/// MIDI onward from plug-ins in an instrument slot.
static const Steinberg::FUID kInstrumentUID(0x45DDCD0C, 0x8E324439, 0x8F497A73, 0xB5E8A33A);
/// Shared by both processors.
static const Steinberg::FUID kControllerUID(0xAD130A6D, 0x50C8428B, 0xA391709B, 0xD5B36FA6);

/// Parameter IDs (stable: hosts store automation against them).
enum ParamId : Steinberg::Vst::ParamID {
    kParamPlay    = 0,   ///< On / off. Off keeps listening, so it re-enters in time.
    kParamStyle   = 1,   ///< Rock, Alt Rock, Pop, Jazz.
    kParamOffset  = 2,   ///< Timing offset in ms, -50 .. +50 (negative = earlier).
    kParamChannel = 3,   ///< MIDI channel 1 .. 16 for the drum notes (default 10).
    kNumParams
};

/// Style list shown to the user, in parameter order. Index 0 is the base groove.
static constexpr int kNumStyles = 4;
static constexpr const char* kStyleFiles[kNumStyles] = {nullptr, "alt_rock.json", "pop.json", "jazz.json"};

/// Normalised <-> real conversions shared by processor and controller.
inline double offsetFromNorm(double v)  { return v * 100.0 - 50.0; }
inline double offsetToNorm(double ms)   { return (ms + 50.0) / 100.0; }
inline int    styleFromNorm(double v)   { return static_cast<int>(v * (kNumStyles - 1) + 0.5); }
inline int    channelFromNorm(double v) { return static_cast<int>(v * 15.0 + 0.5); }   // 0-based
inline double channelToNorm(int ch0)    { return ch0 / 15.0; }

} // namespace drumalong::vst
