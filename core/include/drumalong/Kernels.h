#pragma once
/**
 * @file Kernels.h
 * @brief The only dense-math primitives the network uses.
 *
 * Every matrix product in the drummer goes through these two functions, so
 * this header is the single seam where a SIMD-intrinsics, oneDNN or GPU
 * backend would plug in. At the current network size the plain loops below,
 * auto-vectorised with AVX2, cost a few tens of microseconds per 10 ms frame,
 * so there is no reason to leave the CPU yet.
 */

#include "Weights.h"

namespace drumalong::kernels {

/**
 * @brief Dot product with four independent accumulators.
 *
 * Splitting the sum breaks the loop-carried dependency on a single register,
 * which lets the compiler keep several FMA units busy at once.
 *
 * @param a First vector (n floats).
 * @param b Second vector (n floats).
 * @param n Length. Non-positive lengths return 0.
 */
inline float dot(const float* __restrict a, const float* __restrict b, int n) noexcept {
    if (!a || !b || n <= 0) {
        return 0.0f;
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int i = 0;

    // Main body: 4 lanes per iteration, vectorised further by the compiler.
    for (; i + 4 <= n; i += 4) {
        acc0 += a[i]     * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
    }

    // Tail.
    for (; i < n; ++i) {
        acc0 += a[i] * b[i];
    }

    return (acc0 + acc1) + (acc2 + acc3);
}

/**
 * @brief y = M * x.
 *
 * @param m Matrix (rows x cols).
 * @param x Input vector, at least m.cols floats.
 * @param y Output vector, at least m.rows floats.
 */
inline void matVec(const Matrix& m, const float* x, float* y) noexcept {
    if (!x || !y || m.empty()) {
        return;
    }
    for (int r = 0; r < m.rows; ++r) {
        y[r] = dot(m.row(r), x, m.cols);
    }
}

} // namespace drumalong::kernels
