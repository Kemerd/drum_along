#pragma once
/**
 * @file MidiEvent.h
 * @brief Minimal MIDI event type shared between the engine, the offline tool
 *        and (later) the VST3 wrapper.
 */

#include <cstdint>

namespace drumalong {

/**
 * @brief One MIDI note event, timestamped in samples.
 *
 * sampleOffset is relative to the start of the audio block the event was
 * emitted in, which is exactly what a VST3 host wants. The offline tool
 * converts it to absolute time itself.
 */
struct MidiEvent {
    int32_t sampleOffset = 0;   ///< Position within the current block.
    uint8_t note         = 0;   ///< GM percussion note number.
    uint8_t velocity     = 0;   ///< 1..127 for note-on; ignored for note-off.
    bool    noteOn       = true;
};

} // namespace drumalong
