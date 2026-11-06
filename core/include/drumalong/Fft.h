#pragma once
/**
 * @file Fft.h
 * @brief Radix-2 complex FFT used to evaluate the network's DFT layers.
 *
 * The onset and chord front ends are mathematically a dense linear layer
 * (windowed cosine and sine rows). Evaluating that layer as a matrix product
 * costs N x 2K multiply-adds per frame; the FFT computes the *same* linear map
 * in O(N log N). Nothing about the network changes, only how fast its first
 * layer is evaluated.
 *
 * Real-time contract: prepare() allocates; forward() never allocates.
 */

#include <cmath>
#include <vector>

namespace drumalong {

class Fft {
public:
    /**
     * @brief Precompute twiddles and the bit-reversal permutation.
     * @param n Transform length; must be a power of two >= 2.
     * @return false for an invalid length.
     */
    bool prepare(int n) {
        if (n < 2 || (n & (n - 1)) != 0) {
            n_ = 0;
            return false;
        }
        n_ = n;
        re_.assign(static_cast<size_t>(n), 0.0f);
        im_.assign(static_cast<size_t>(n), 0.0f);

        // Twiddles W^k = exp(-2*pi*i*k/n) for k < n/2, computed in double.
        cosT_.resize(static_cast<size_t>(n / 2));
        sinT_.resize(static_cast<size_t>(n / 2));
        for (int k = 0; k < n / 2; ++k) {
            const double a = -2.0 * 3.14159265358979323846 * k / n;
            cosT_[static_cast<size_t>(k)] = static_cast<float>(std::cos(a));
            sinT_[static_cast<size_t>(k)] = static_cast<float>(std::sin(a));
        }

        // Bit-reversal table.
        int bits = 0;
        while ((1 << bits) < n) {
            ++bits;
        }
        rev_.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) {
                r |= ((i >> b) & 1) << (bits - 1 - b);
            }
            rev_[static_cast<size_t>(i)] = r;
        }
        return true;
    }

    int size() const { return n_; }

    /**
     * @brief Transform window[n] * x[n] and write bins [kLo, kLo + count) as
     *        interleaved (re, im) into out[2 * count].
     *
     * Matches the dense layer exactly: out[2i] = sum w x cos, out[2i+1] =
     * -sum w x sin, for bin k = kLo + i.
     */
    void forward(const float* x, const float* window, int kLo, int count, float* out) {
        if (n_ == 0 || !x || !window || !out || kLo < 0 || count <= 0 || kLo + count > n_ / 2 + 1) {
            return;
        }
        const int n = n_;

        // Windowed input in bit-reversed order (imaginary part zero).
        for (int i = 0; i < n; ++i) {
            const int r = rev_[static_cast<size_t>(i)];
            re_[static_cast<size_t>(r)] = x[i] * window[i];
            im_[static_cast<size_t>(r)] = 0.0f;
        }

        // Iterative Cooley-Tukey butterflies.
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len >> 1;
            const int step = n / len;
            for (int start = 0; start < n; start += len) {
                for (int j = 0; j < half; ++j) {
                    const float wr = cosT_[static_cast<size_t>(j * step)];
                    const float wi = sinT_[static_cast<size_t>(j * step)];
                    const size_t a = static_cast<size_t>(start + j);
                    const size_t b = a + static_cast<size_t>(half);
                    const float tr = re_[b] * wr - im_[b] * wi;
                    const float ti = re_[b] * wi + im_[b] * wr;
                    re_[b] = re_[a] - tr;
                    im_[b] = im_[a] - ti;
                    re_[a] += tr;
                    im_[a] += ti;
                }
            }
        }

        for (int i = 0; i < count; ++i) {
            out[2 * i]     = re_[static_cast<size_t>(kLo + i)];
            out[2 * i + 1] = im_[static_cast<size_t>(kLo + i)];
        }
    }

private:
    int n_ = 0;
    std::vector<float> re_, im_, cosT_, sinT_;
    std::vector<int>   rev_;
};

} // namespace drumalong
