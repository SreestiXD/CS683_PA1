//*
// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

// Add the eight lanes of an AVX2 register
static inline float horizontal_sum(__m256 v) {
    alignas(32) float partial[8];
    _mm256_store_ps(partial, v);

    float sum = 0.0f;
    for (int lane = 0; lane < 8; ++lane) {
        sum += partial[lane];
    }

    return sum;
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;

        int j = 0;

        // Register-tiling: compute four C elements together
        for (; j + 1 < N; j += 1) {
            const float* b0 = B + static_cast<long>(j + 0) * ldb;
           
            __m256 vacc0 = _mm256_setzero_ps();
            
            int p = 0;

            for (; p + 7 < K; p += 8) {
                // Load A once and reuse it for four outputs
                __m256 va = _mm256_loadu_ps(a + p);

                __m256 vb0 = _mm256_loadu_ps(b0 + p);
                
                vacc0 = _mm256_fmadd_ps(va, vb0, vacc0);
                
            }

            float acc0 = horizontal_sum(vacc0);

            C[static_cast<long>(i) * ldc + j + 0] = acc0;
        }
    }
}
//*/

////////////////////////////////////////////////

/* 128 bit SIMD vector

// Experimental 128-bit SSE/FMA SGEMM.
// Used only for the Task 2 SIMD-width comparison.

#include <immintrin.h>

#include "matmul.h"

static inline float horizontal_sum_128(__m128 v) {
    alignas(16) float partial[4];
    _mm_store_ps(partial, v);

    return partial[0] + partial[1] + partial[2] + partial[3];
}

// Uses the same function name expected by the supplied benchmark.
// This file is compiled INSTEAD OF src/matmul_simd.cpp.
void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;

        int j = 0;

        // Compute four output elements simultaneously.
        for (; j < N; j += 1) {
            const float* b0 = B + static_cast<long>(j + 0) * ldb;

            __m128 vacc0 = _mm_setzero_ps();

            int p = 0;

            // SSE processes four single-precision floats per iteration.
            for (; p + 3 < K; p += 4) {
                __m128 va = _mm_loadu_ps(a + p);

                __m128 vb0 = _mm_loadu_ps(b0 + p);

                vacc0 = _mm_fmadd_ps(va, vb0, vacc0);
            }

            float acc0 = horizontal_sum_128(vacc0);

            for (; p < K; ++p) {
                const float av = a[p];

                acc0 += av * b0[p];
            }

            C[static_cast<long>(i) * ldc + j + 0] = acc0;
        }

    }
}
*/
