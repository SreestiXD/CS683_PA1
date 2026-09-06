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
        for (; j + 3 < N; j += 4) {
            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            __m256 vacc0 = _mm256_setzero_ps();
            __m256 vacc1 = _mm256_setzero_ps();
            __m256 vacc2 = _mm256_setzero_ps();
            __m256 vacc3 = _mm256_setzero_ps();

            int p = 0;

            for (; p + 7 < K; p += 8) {
                // Load A once and reuse it for four outputs
                __m256 va = _mm256_loadu_ps(a + p);

                __m256 vb0 = _mm256_loadu_ps(b0 + p);
                __m256 vb1 = _mm256_loadu_ps(b1 + p);
                __m256 vb2 = _mm256_loadu_ps(b2 + p);
                __m256 vb3 = _mm256_loadu_ps(b3 + p);

                vacc0 = _mm256_fmadd_ps(va, vb0, vacc0);
                vacc1 = _mm256_fmadd_ps(va, vb1, vacc1);
                vacc2 = _mm256_fmadd_ps(va, vb2, vacc2);
                vacc3 = _mm256_fmadd_ps(va, vb3, vacc3);
            }

            float acc0 = horizontal_sum(vacc0);
            float acc1 = horizontal_sum(vacc1);
            float acc2 = horizontal_sum(vacc2);
            float acc3 = horizontal_sum(vacc3);

            // Remaining elements
            for (; p < K; ++p) {
                const float av = a[p];

                acc0 += av * b0[p];
                acc1 += av * b1[p];
                acc2 += av * b2[p];
                acc3 += av * b3[p];
            }

            C[static_cast<long>(i) * ldc + j + 0] = acc0;
            C[static_cast<long>(i) * ldc + j + 1] = acc1;
            C[static_cast<long>(i) * ldc + j + 2] = acc2;
            C[static_cast<long>(i) * ldc + j + 3] = acc3;
        }

        // Handle the remaining columns when N is not divisible by 4
        for (; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;

            __m256 vacc = _mm256_setzero_ps();

            int p = 0;

            for (; p + 7 < K; p += 8) {
                __m256 va = _mm256_loadu_ps(a + p);
                __m256 vb = _mm256_loadu_ps(b + p);

                vacc = _mm256_fmadd_ps(va, vb, vacc);
            }

            float acc = horizontal_sum(vacc);

            for (; p < K; ++p) {
                acc += a[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
