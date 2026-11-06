/**
 * @file main.cpp
 * @brief Offline test harness: stream a WAV through the engine, write MIDI.
 *
 * Usage:
 *   wav2mid <input.wav> <output_prefix> [--offset-ms X] [--block N]
 *
 * Writes:
 *   <prefix>.mid         Drum track, GM channel 10.
 *   <prefix>_events.csv  Every note event with its absolute time.
 *   <prefix>_frames.csv  Per-frame network state (tempo, phase, level...).
 *   <prefix>_steps.csv   Every planned sixteenth and what the groove did.
 *
 * The file is fed to the engine in host-sized blocks, exactly like a DAW
 * would, so this exercises the same code path as the plugin.
 */

#include "drumalong/DrumAlongEngine.h"
#include "drumalong/WeightProgrammer.h"
#include "drumalong/WeightsIO.h"
#include "../common/WavIO.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace drumalong;
using drumalong::tools::WavData;
using drumalong::tools::readWav;

namespace {

// ===========================================================================
// Standard MIDI File writing (format 0, one track, GM drums on channel 10)
// ===========================================================================

struct TimedEvent {
    double  seconds = 0.0;
    uint8_t note    = 0;
    uint8_t vel     = 0;
    bool    on      = true;
};

void writeVarLen(std::vector<uint8_t>& v, uint32_t value) {
    uint8_t buf[5];
    int n = 0;
    buf[n++] = static_cast<uint8_t>(value & 0x7F);
    while ((value >>= 7) != 0) {
        buf[n++] = static_cast<uint8_t>((value & 0x7F) | 0x80);
    }
    while (n > 0) {
        v.push_back(buf[--n]);
    }
}

/**
 * @brief Write events as an SMF at a nominal 120 BPM, 960 PPQ.
 *
 * The drummer has no fixed tempo, so the file tempo is just a time base:
 * 1 tick = 1/1920 s. A DAW will place every hit at its true time.
 */
bool writeMidi(const std::string& path, std::vector<TimedEvent> events) {
    constexpr uint16_t kPpq          = 960;
    constexpr uint32_t kUsPerQuarter = 500000;   // 120 BPM
    constexpr double   kTicksPerSec  = kPpq * 1e6 / kUsPerQuarter;

    // Note-offs first at equal times, so a retrigger never cuts itself off.
    std::stable_sort(events.begin(), events.end(), [](const TimedEvent& a, const TimedEvent& b) {
        if (a.seconds != b.seconds) {
            return a.seconds < b.seconds;
        }
        return !a.on && b.on;
    });

    std::vector<uint8_t> track;
    // Tempo meta event.
    track.insert(track.end(), {0x00, 0xFF, 0x51, 0x03,
                               static_cast<uint8_t>((kUsPerQuarter >> 16) & 0xFF),
                               static_cast<uint8_t>((kUsPerQuarter >> 8) & 0xFF),
                               static_cast<uint8_t>(kUsPerQuarter & 0xFF)});

    uint32_t lastTick = 0;
    for (const TimedEvent& e : events) {
        const double   ticksD = std::max(0.0, e.seconds * kTicksPerSec);
        const uint32_t tick   = static_cast<uint32_t>(std::llround(ticksD));
        writeVarLen(track, tick >= lastTick ? tick - lastTick : 0);
        lastTick = std::max(lastTick, tick);
        track.push_back(static_cast<uint8_t>((e.on ? 0x90 : 0x80) | 9));   // channel 10
        track.push_back(e.note & 0x7F);
        track.push_back(e.on ? static_cast<uint8_t>(std::clamp<int>(e.vel, 1, 127)) : 64);
    }
    track.insert(track.end(), {0x00, 0xFF, 0x2F, 0x00});   // end of track

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    auto be16 = [&f](uint16_t v) { const char b[2] = {char(v >> 8), char(v & 0xFF)}; f.write(b, 2); };
    auto be32 = [&f](uint32_t v) {
        const char b[4] = {char(v >> 24), char((v >> 16) & 0xFF), char((v >> 8) & 0xFF), char(v & 0xFF)};
        f.write(b, 4);
    };
    f.write("MThd", 4); be32(6); be16(0); be16(1); be16(kPpq);
    f.write("MTrk", 4); be32(static_cast<uint32_t>(track.size()));
    f.write(reinterpret_cast<const char*>(track.data()), static_cast<std::streamsize>(track.size()));
    return static_cast<bool>(f);
}

// ===========================================================================
// Debug observer: logs every frame and step to CSV
// ===========================================================================

class CsvObserver : public IEngineObserver {
public:
    CsvObserver(FILE* frames, FILE* steps, double sampleRate)
        : frames_(frames), steps_(steps), sr_(sampleRate) {
        if (frames_) {
            std::fprintf(frames_, "frame,time_s,flux,onset,low_onset,harmonic,loudness,periodicity,tempo_bpm,"
                                  "beat_pos,phase_err,downbeat_slot,level,locked,silent,section,"
                                  "beat_act,downbeat_act,drums_active\n");
        }
        if (steps_) {
            std::fprintf(steps_, "step,time_s,step_in_bar,phrase_bar,playing,fill,crash,level,tempo_bpm,hits\n");
        }
    }

    void onFrame(const FrameState& s) override {
        // Training export: one row of raw features per frame.
        if (features_ && s.features && s.numFeatures > 0) {
            std::fwrite(s.features, sizeof(float), static_cast<size_t>(s.numFeatures), features_);
        }
        if (s.locked && !s.silent) {
            lockedTempos.push_back(s.tempoBpm);
            if (firstLockSeconds < 0.0) {
                firstLockSeconds = s.centerSample / sr_;
            }
        }
        if (!frames_) {
            return;
        }
        std::fprintf(frames_, "%lld,%.4f,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.4f,%.4f,%d,%.4f,%d,%d,%d,%.4f,%.4f,%.4f\n",
                     static_cast<long long>(s.frameIndex), s.centerSample / sr_, s.flux, s.onset,
                     s.lowOnset, s.harmonic, s.loudness, s.periodicity, s.tempoBpm, s.beatPos, s.phaseError,
                     s.downbeatSlot, s.level, s.locked ? 1 : 0, s.silent ? 1 : 0, s.sectionChange ? 1 : 0,
                     s.beatAct, s.downbeatAct, s.drumsActive);
    }

    void onStep(const StepInfo& st) override {
        if (st.playing && firstPlaySeconds < 0.0) {
            firstPlaySeconds = st.timeSample / sr_;
        }
        if (st.fill && st.stepInBar == 12) {
            ++fills;
        }
        if (st.crash) {
            ++crashes;
        }
        if (!steps_) {
            return;
        }
        std::fprintf(steps_, "%lld,%.5f,%d,%d,%d,%d,%d,%.4f,%.3f,%d\n",
                     static_cast<long long>(st.globalStep), st.timeSample / sr_, st.stepInBar,
                     st.phraseBar, st.playing ? 1 : 0, st.fill ? 1 : 0, st.crash ? 1 : 0,
                     st.level, st.tempoBpm, st.hits);
    }

    std::vector<float> lockedTempos;
    double firstLockSeconds = -1.0;
    double firstPlaySeconds = -1.0;
    int    fills   = 0;
    int    crashes = 0;
    FILE*  features_ = nullptr;   ///< Optional training export (owned by main).

private:
    FILE*  frames_ = nullptr;
    FILE*  steps_  = nullptr;
    double sr_     = 1.0;
};

void usage() {
    std::fprintf(stderr,
                 "usage: wav2mid <input.wav> <output_prefix> [--weights file.json] [--features out.daf]\n"
                 "               [--offset-ms X] [--block N]\n"
                 "       wav2mid --dump-weights <file.json>   (write the hand-written defaults)\n");
}

} // namespace

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
    // -----------------------------------------------------------------------
    // --dump-weights: write the default weights file and stop.
    // -----------------------------------------------------------------------
    if (argc == 3 && std::string(argv[1]) == "--dump-weights") {
        Weights defaults;
        programDefaults(defaults);
        std::string error;
        if (!saveWeightsFile(argv[2], defaults, &error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        std::printf("wrote      %s\n", argv[2]);
        return 0;
    }

    if (argc < 3) {
        usage();
        return 2;
    }
    const std::string inPath = argv[1];
    const std::string prefix = argv[2];
    std::string weightsPath;
    std::string featuresPath;
    float offsetMs  = 0.0f;
    int   blockSize = 512;

    // -----------------------------------------------------------------------
    // Optional flags.
    // -----------------------------------------------------------------------
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--offset-ms" && i + 1 < argc) {
            offsetMs = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--block" && i + 1 < argc) {
            blockSize = std::clamp(std::atoi(argv[++i]), 16, 8192);
        } else if (a == "--weights" && i + 1 < argc) {
            weightsPath = argv[++i];
        } else if (a == "--features" && i + 1 < argc) {
            featuresPath = argv[++i];
        } else {
            usage();
            return 2;
        }
    }

    // -----------------------------------------------------------------------
    // Load audio.
    // -----------------------------------------------------------------------
    WavData wav;
    std::string err;
    if (!readWav(inPath, wav, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    std::printf("input      %s\n", inPath.c_str());
    std::printf("format     %.0f Hz, %d ch, %.2f s\n", wav.sampleRate, wav.channels,
                wav.frames / wav.sampleRate);

    // -----------------------------------------------------------------------
    // Engine.
    // -----------------------------------------------------------------------
    static DrumAlongEngine engine;   // Static: it holds a few MB of weights.
    std::string prepError;
    if (!engine.prepare(wav.sampleRate, blockSize, weightsPath, &prepError)) {
        std::fprintf(stderr, "error: engine.prepare failed: %s\n", prepError.c_str());
        return 1;
    }
    engine.setOutputOffsetMs(offsetMs);

    // Musical weights (the editable spec) versus derived front-end tables.
    const Weights& w = engine.weights();
    size_t musical = w.fluxReadout.size() + w.tempoPrior.size() + 3 +
                     w.hitOnLevel.data.size() + w.hitOffLevel.data.size() + w.velocity.data.size();
    for (const Matrix& m : w.fillVelocity) {
        musical += m.data.size();
    }
    const size_t tables = w.dftWindow.size() + w.mel.data.size() + w.lowBand.size() +
                          w.chromaWindow.size() + w.chroma.data.size();
    std::printf("weights    %s\n", weightsPath.empty() ? "built-in defaults" : weightsPath.c_str());
    std::printf("network    onset window %d, chord window %d, hop %d; %zu musical weights + %zu derived table values\n",
                w.windowSamples, w.chromaWindowSamples, w.hopSamples, musical, tables);

    FILE* framesCsv = std::fopen((prefix + "_frames.csv").c_str(), "w");
    FILE* stepsCsv  = std::fopen((prefix + "_steps.csv").c_str(), "w");
    CsvObserver observer(framesCsv, stepsCsv, wav.sampleRate);
    engine.setObserver(&observer);

    // -----------------------------------------------------------------------
    // Optional training export. Header (little endian):
    //   "DAF1", int32 numFeatures, int32 hopSamples, int32 onsetWindowSamples,
    //   float64 sampleRate, then numFrames x numFeatures float32.
    // Frame t is centred at ((t + 1) * hop - window / 2) / sampleRate seconds.
    // -----------------------------------------------------------------------
    FILE* featuresFile = nullptr;
    if (!featuresPath.empty()) {
        featuresFile = std::fopen(featuresPath.c_str(), "wb");
        if (!featuresFile) {
            std::fprintf(stderr, "error: cannot open %s\n", featuresPath.c_str());
            return 1;
        }
        const int32_t hdr[3] = {kBeatFeatures, w.hopSamples, w.windowSamples};
        const double  sr     = w.sampleRate;
        std::fwrite("DAF1", 1, 4, featuresFile);
        std::fwrite(hdr, sizeof(int32_t), 3, featuresFile);
        std::fwrite(&sr, sizeof(double), 1, featuresFile);
        observer.features_ = featuresFile;
    }

    // -----------------------------------------------------------------------
    // Stream in host-sized blocks.
    // -----------------------------------------------------------------------
    std::vector<const float*> chPtrs(static_cast<size_t>(wav.channels));
    std::vector<MidiEvent>    midiOut(4096);
    std::vector<TimedEvent>   events;
    events.reserve(1 << 16);

    const auto t0 = std::chrono::steady_clock::now();
    for (size_t start = 0; start < wav.frames; start += static_cast<size_t>(blockSize)) {
        const int n = static_cast<int>(std::min<size_t>(blockSize, wav.frames - start));
        for (int c = 0; c < wav.channels; ++c) {
            chPtrs[static_cast<size_t>(c)] = wav.data[static_cast<size_t>(c)].data() + start;
        }
        const int count = engine.process(chPtrs.data(), wav.channels, n,
                                         midiOut.data(), static_cast<int>(midiOut.size()));
        for (int i = 0; i < count; ++i) {
            const MidiEvent& m = midiOut[static_cast<size_t>(i)];
            events.push_back({(static_cast<double>(start) + m.sampleOffset) / wav.sampleRate,
                              m.note, m.velocity, m.noteOn});
        }
    }
    const auto   t1      = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    if (framesCsv)    std::fclose(framesCsv);
    if (stepsCsv)     std::fclose(stepsCsv);
    if (featuresFile) std::fclose(featuresFile);

    // -----------------------------------------------------------------------
    // Write MIDI and an event CSV.
    // -----------------------------------------------------------------------
    if (!writeMidi(prefix + ".mid", events)) {
        std::fprintf(stderr, "error: could not write %s.mid\n", prefix.c_str());
        return 1;
    }
    if (FILE* ev = std::fopen((prefix + "_events.csv").c_str(), "w")) {
        std::fprintf(ev, "time_s,note,velocity,on\n");
        for (const TimedEvent& e : events) {
            std::fprintf(ev, "%.5f,%d,%d,%d\n", e.seconds, e.note, e.vel, e.on ? 1 : 0);
        }
        std::fclose(ev);
    }

    // -----------------------------------------------------------------------
    // Summary.
    // -----------------------------------------------------------------------
    const double audioSeconds = wav.frames / wav.sampleRate;
    const double numFrames    = audioSeconds * kFrameRate;
    std::printf("cpu        %.3f s for %.2f s of audio (%.0fx realtime, %.1f us per frame)\n",
                elapsed, audioSeconds, audioSeconds / std::max(elapsed, 1e-9),
                1e6 * elapsed / std::max(numFrames, 1.0));

    if (observer.firstLockSeconds >= 0.0) {
        std::vector<float> tempos = observer.lockedTempos;
        std::nth_element(tempos.begin(), tempos.begin() + tempos.size() / 2, tempos.end());
        std::printf("lock       first at %.2f s, median tempo %.2f BPM\n",
                    observer.firstLockSeconds, tempos[tempos.size() / 2]);
    } else {
        std::printf("lock       never locked\n");
    }
    if (observer.firstPlaySeconds >= 0.0) {
        std::printf("drummer    starts at %.2f s\n", observer.firstPlaySeconds);
    }
    const EngineStats& st = engine.stats();
    std::printf("notes      %lld note-ons, %d fills, %d crashes\n",
                static_cast<long long>(st.notesOn), observer.fills, observer.crashes);
    std::printf("timing     %lld late steps, %lld skipped steps, %lld dropped events\n",
                static_cast<long long>(st.lateEvents), static_cast<long long>(st.skippedSteps),
                static_cast<long long>(st.droppedEvents));
    std::printf("wrote      %s.mid, %s_events.csv, %s_frames.csv, %s_steps.csv\n",
                prefix.c_str(), prefix.c_str(), prefix.c_str(), prefix.c_str());
    return 0;
}
