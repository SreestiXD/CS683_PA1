/*
// conv_simd.cpp  STAGE 4: SIMD with 128-bit SSE

#include <immintrin.h>
#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {

        // W is a multiple of 8, so processing 4 at a time is safe
        for (int ox = 0; ox < W; ox += 4) {

            __m128 acc = _mm_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {

                    const int input_base =
                        (oy + ky) * in_stride + ox + kx;

                    const __m128 input =
                        _mm_loadu_ps(&in[input_base]);

                    const __m128 weight =
                        _mm_set1_ps(ker[ky * K + kx]);

                    acc = _mm_add_ps(
                        acc,
                        _mm_mul_ps(input, weight)
                    );
                }
            }

            _mm_storeu_ps(&out[oy * W + ox], acc);
        }
    }
}
*/

// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {

        for (int ox = 0; ox < W; ox += 8) {

            // Eight output accumulators
            __m256 acc = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {

                    const int input_base =
                        (oy + ky) * in_stride + ox + kx;

                    // Load 8 adjacent input values
                    __m256 input_vec =
                        _mm256_loadu_ps(&in[input_base]);

                    // Broadcast one kernel value to all 8 lanes
                    __m256 weight_vec =
                        _mm256_set1_ps(ker[ky * K + kx]);

                    // acc += input_vec * weight_vec
                    acc = _mm256_fmadd_ps(input_vec, weight_vec, acc);
                }
            }

            // Store 8 output values
            _mm256_storeu_ps(&out[oy * W + ox], acc);
        }
    }
}

