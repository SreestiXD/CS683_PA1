//*
#include <immintrin.h>
#include "convolution.h"

#define TILE_SIZE 16

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy0 = 0; oy0 < H; oy0 += TILE_SIZE) {
        for (int ox0 = 0; ox0 < W; ox0 += TILE_SIZE) {

            const int oy_end = (oy0 + TILE_SIZE < H)
                             ? oy0 + TILE_SIZE : H;
            const int ox_end = (ox0 + TILE_SIZE < W)
                             ? ox0 + TILE_SIZE : W;

            for (int oy = oy0; oy < oy_end; ++oy) {

                int ox = ox0;

                // Main 32-pixel AVX2 loop
                for (; ox + 32 <= ox_end; ox += 16) {

                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                   
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {

                            const float weight = ker[ky * K + kx];
                            const __m256 w = _mm256_set1_ps(weight);

                            const int base =
                                (oy + ky) * in_stride + ox + kx;

                            acc0 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[base]),
                                w, acc0);

                            acc1 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[base + 8]),
                                w, acc1);

                        }
                    }

                    _mm256_storeu_ps(&out[oy * W + ox],      acc0);
                    _mm256_storeu_ps(&out[oy * W + ox + 8],  acc1);
                }
            }
        }
    }
}
//*/
/*
#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {

        for (int ox = 0; ox < W; ox += 16) {

            // Eight output accumulators
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {

                    const int input_base = (oy + ky) * in_stride + ox + kx;

                    // Load 8 adjacent input values
                    __m256 input_vec1 = _mm256_loadu_ps(&in[input_base]);
                    __m256 input_vec2 = _mm256_loadu_ps(&in[input_base + 8]);
                    
                    // Broadcast one kernel value to all 8 lanes
                    __m256 weight_vec = _mm256_set1_ps(ker[ky * K + kx]);

                    // acc += input_vec * weight_vec
                    acc0 = _mm256_fmadd_ps(input_vec1, weight_vec, acc0);
                    acc1 = _mm256_fmadd_ps(input_vec2, weight_vec, acc1);
                    
                }
            }

            // Store 8 output values
            _mm256_storeu_ps(&out[oy * W + ox], acc0);
            _mm256_storeu_ps(&out[oy * W + ox + 8], acc1);
            
        }
    }
}
*/
/*
#include <immintrin.h>
#include "convolution.h"

#define TILE_SIZE 32

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy0 = 0; oy0 < H; oy0 += TILE_SIZE) {
        for (int ox0 = 0; ox0 < W; ox0 += TILE_SIZE) {

            const int oy_end = (oy0 + TILE_SIZE < H)
                             ? oy0 + TILE_SIZE : H;
            const int ox_end = (ox0 + TILE_SIZE < W)
                             ? ox0 + TILE_SIZE : W;

            for (int oy = oy0; oy < oy_end; ++oy) {

                int ox = ox0;

                // Main 32-pixel AVX2 loop
                for (; ox + 32 <= ox_end; ox += 32) {

                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {

                            const float weight = ker[ky * K + kx];
                            const __m256 w = _mm256_set1_ps(weight);

                            const int base =
                                (oy + ky) * in_stride + ox + kx;

                            acc0 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[base]),
                                w, acc0);

                            acc1 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[base + 8]),
                                w, acc1);

                            acc2 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[base + 16]),
                                w, acc2);

                            acc3 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[base + 24]),
                                w, acc3);
                        }
                    }

                    _mm256_storeu_ps(&out[oy * W + ox],      acc0);
                    _mm256_storeu_ps(&out[oy * W + ox + 8],  acc1);
                    _mm256_storeu_ps(&out[oy * W + ox + 16], acc2);
                    _mm256_storeu_ps(&out[oy * W + ox + 24], acc3);
                }
            }
        }
    }
}
//*/
