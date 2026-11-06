#pragma once
/**
 * @file WavIO.h
 * @brief Minimal, defensive WAV reader shared by the offline tools
 *        (wav2mid and vst3_check), so both read audio identically.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace drumalong::tools {

// ===========================================================================
// WAV reading (PCM 16/24/32, float 32/64, WAVE_FORMAT_EXTENSIBLE)
// ===========================================================================

struct WavData {
    double sampleRate = 0.0;
    int    channels   = 0;
    size_t frames     = 0;
    std::vector<std::vector<float>> data;   ///< [channel][frame]
};

inline uint16_t readU16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
inline uint32_t readU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

/**
 * @brief Parse a RIFF/WAVE file into deinterleaved float channels.
 * @return false with a message in @p err on any malformed or unsupported input.
 */
inline bool readWav(const std::string& path, WavData& out, std::string& err) {
    // -----------------------------------------------------------------------
    // Slurp the file.
    // -----------------------------------------------------------------------
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    const std::streamoff size = f.tellg();
    if (size < 44) {
        err = "file too small to be a WAV";
        return false;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!f) {
        err = "read failed";
        return false;
    }

    if (std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        err = "not a RIFF/WAVE file";
        return false;
    }

    // -----------------------------------------------------------------------
    // Walk the chunks looking for fmt and data.
    // -----------------------------------------------------------------------
    uint16_t format = 0, channels = 0, bits = 0, blockAlign = 0;
    uint32_t rate = 0;
    const uint8_t* dataPtr = nullptr;
    size_t dataSize = 0;

    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const uint8_t* ck   = bytes.data() + pos;
        const uint32_t ckSz = readU32(ck + 4);
        const size_t   body = pos + 8;
        const size_t   avail = bytes.size() - body;
        const size_t   len  = std::min<size_t>(ckSz, avail);

        if (std::memcmp(ck, "fmt ", 4) == 0 && len >= 16) {
            format     = readU16(bytes.data() + body);
            channels   = readU16(bytes.data() + body + 2);
            rate       = readU32(bytes.data() + body + 4);
            blockAlign = readU16(bytes.data() + body + 12);
            bits       = readU16(bytes.data() + body + 14);
            // Extensible: the real format code is the first 2 bytes of the GUID.
            if (format == 0xFFFE && len >= 26) {
                format = readU16(bytes.data() + body + 24);
            }
        } else if (std::memcmp(ck, "data", 4) == 0) {
            dataPtr  = bytes.data() + body;
            dataSize = len;
        }

        // Chunks are padded to even sizes.
        pos = body + static_cast<size_t>(ckSz) + (ckSz & 1u);
    }

    // -----------------------------------------------------------------------
    // Validate.
    // -----------------------------------------------------------------------
    if (!dataPtr || channels == 0 || rate == 0 || blockAlign == 0) {
        err = "missing fmt or data chunk";
        return false;
    }
    const bool isPcm   = (format == 1) && (bits == 16 || bits == 24 || bits == 32);
    const bool isFloat = (format == 3) && (bits == 32 || bits == 64);
    if (!isPcm && !isFloat) {
        err = "unsupported WAV format " + std::to_string(format) + " / " + std::to_string(bits) + " bits";
        return false;
    }
    const size_t bytesPerSample = bits / 8u;
    if (blockAlign != channels * bytesPerSample) {
        err = "inconsistent block align";
        return false;
    }

    // -----------------------------------------------------------------------
    // Decode to float.
    // -----------------------------------------------------------------------
    out.sampleRate = rate;
    out.channels   = channels;
    out.frames     = dataSize / blockAlign;
    out.data.assign(channels, std::vector<float>(out.frames));

    for (size_t i = 0; i < out.frames; ++i) {
        const uint8_t* frame = dataPtr + i * blockAlign;
        for (int c = 0; c < channels; ++c) {
            const uint8_t* p = frame + c * bytesPerSample;
            float v = 0.0f;
            if (isPcm && bits == 16) {
                v = static_cast<int16_t>(readU16(p)) / 32768.0f;
            } else if (isPcm && bits == 24) {
                int32_t s = p[0] | (p[1] << 8) | (p[2] << 16);
                if (s & 0x800000) {
                    s |= ~0xFFFFFF;   // sign-extend
                }
                v = s / 8388608.0f;
            } else if (isPcm && bits == 32) {
                v = static_cast<int32_t>(readU32(p)) / 2147483648.0f;
            } else if (isFloat && bits == 32) {
                std::memcpy(&v, p, 4);
            } else {
                double d = 0.0;
                std::memcpy(&d, p, 8);
                v = static_cast<float>(d);
            }
            out.data[static_cast<size_t>(c)][i] = v;
        }
    }
    return true;
}


} // namespace drumalong::tools
