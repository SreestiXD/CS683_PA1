///*
#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

//   for (int i = 0; i < H*W ; i++)  out[i] = 0.0f;

    for (int oy = 0; oy < H; ++oy) {

        int ox = 0;

        // Main 32-pixel AVX2 loop
        for (; ox + 32 <= W; ox += 32) {

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {

                    const float weight = ker[ky * K + kx];
                    
                    const __m256 w = _mm256_set1_ps(weight);

                    const int base = (oy + ky) * in_stride + ox + kx;

                    acc0 = _mm256_fmadd_ps( _mm256_loadu_ps(&in[base]), w, acc0);

                    acc1 = _mm256_fmadd_ps( _mm256_loadu_ps(&in[base + 8]), w, acc1);

                    acc2 = _mm256_fmadd_ps( _mm256_loadu_ps(&in[base + 16]), w, acc2);

                    acc3 = _mm256_fmadd_ps( _mm256_loadu_ps(&in[base + 24]), w, acc3);
                }
            }

            _mm256_storeu_ps(&out[oy * W + ox],      acc0);
            _mm256_storeu_ps(&out[oy * W + ox + 8],  acc1);
            _mm256_storeu_ps(&out[oy * W + ox + 16], acc2);
            _mm256_storeu_ps(&out[oy * W + ox + 24], acc3);
        }
    }
}

//*/
/* mait - has tile + simd + unroll

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

                            const int base = (oy + ky) * in_stride + ox + kx;

                            acc0 = _mm256_fmadd_ps( _mm256_loadu_ps(&in[base]), w, acc0);

                    	    acc1 = _mm256_fmadd_ps( _mm256_loadu_ps(&in[base + 8]), w, acc1);

                            acc2 = _mm256_fmadd_ps( _mm256_loadu_ps(&in[base + 16]), w, acc2);

                            acc3 = _mm256_fmadd_ps( _mm256_loadu_ps(&in[base + 24]), w, acc3);
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
*/

