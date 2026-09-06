// matmul_optimized.cpp — 3x4 register-tiled experiment

#include <algorithm>
#include <immintrin.h>
#include "matmul.h"

#ifndef M_BLOCK
#define M_BLOCK 36
#endif

#ifndef N_BLOCK
#define N_BLOCK 16
#endif

#ifndef PREFETCH_DISTANCE
#define PREFETCH_DISTANCE 64
#endif

#ifndef PREFETCH_HINT
#define PREFETCH_HINT _MM_HINT_T0
#endif

static constexpr int FLOATS_PER_CACHE_LINE = 16;

static inline float horizontal_sum(__m256 v) {
    alignas(32) float partial[8];
    _mm256_store_ps(partial, v);

    float sum = 0.0f;
    for (int lane = 0; lane < 8; ++lane)
        sum += partial[lane];

    return sum;
}


// Fallback for remaining individual rows.
static inline void compute_one_row(
    const float* a, const float* B, float* c,
    int j_begin, int j_end, int K, int ldb) {

    int j = j_begin;

    for (; j + 3 < j_end; j += 4) {
        const float* b0 = B + static_cast<long>(j + 0) * ldb;
        const float* b1 = B + static_cast<long>(j + 1) * ldb;
        const float* b2 = B + static_cast<long>(j + 2) * ldb;
        const float* b3 = B + static_cast<long>(j + 3) * ldb;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int p = 0;

        for (; p + 7 < K; p += 8) {
            __m256 va = _mm256_loadu_ps(a + p);

            acc0 = _mm256_fmadd_ps(
                va, _mm256_loadu_ps(b0 + p), acc0);
            acc1 = _mm256_fmadd_ps(
                va, _mm256_loadu_ps(b1 + p), acc1);
            acc2 = _mm256_fmadd_ps(
                va, _mm256_loadu_ps(b2 + p), acc2);
            acc3 = _mm256_fmadd_ps(
                va, _mm256_loadu_ps(b3 + p), acc3);
        }

        float sum0 = horizontal_sum(acc0);
        float sum1 = horizontal_sum(acc1);
        float sum2 = horizontal_sum(acc2);
        float sum3 = horizontal_sum(acc3);

        for (; p < K; ++p) {
            const float av = a[p];

            sum0 += av * b0[p];
            sum1 += av * b1[p];
            sum2 += av * b2[p];
            sum3 += av * b3[p];
        }

        c[j + 0] = sum0;
        c[j + 1] = sum1;
        c[j + 2] = sum2;
        c[j + 3] = sum3;
    }

    for (; j < j_end; ++j) {
        const float* b = B + static_cast<long>(j) * ldb;

        __m256 vacc = _mm256_setzero_ps();
        int p = 0;

        for (; p + 7 < K; p += 8) {
            vacc = _mm256_fmadd_ps(
                _mm256_loadu_ps(a + p),
                _mm256_loadu_ps(b + p),
                vacc);
        }

        float sum = horizontal_sum(vacc);

        for (; p < K; ++p)
            sum += a[p] * b[p];

        c[j] = sum;
    }
}


void matmul_optimized(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc) {

    for (int ii = 0; ii < M; ii += M_BLOCK) {
        const int i_end = std::min(ii + M_BLOCK, M);

        for (int jj = 0; jj < N; jj += N_BLOCK) {
            const int j_end = std::min(jj + N_BLOCK, N);

            int i = ii;

            // Three rows of A are processed together.
            for (; i + 2 < i_end; i += 3) {
                const float* a0 =
                    A + static_cast<long>(i + 0) * lda;
                const float* a1 =
                    A + static_cast<long>(i + 1) * lda;
                const float* a2 =
                    A + static_cast<long>(i + 2) * lda;

                float* c0 = C + static_cast<long>(i + 0) * ldc;
                float* c1 = C + static_cast<long>(i + 1) * ldc;
                float* c2 = C + static_cast<long>(i + 2) * ldc;

                int j = jj;

                // 3 x 4 register tile: 12 outputs.
                for (; j + 3 < j_end; j += 4) {
                    const float* b0 =
                        B + static_cast<long>(j + 0) * ldb;
                    const float* b1 =
                        B + static_cast<long>(j + 1) * ldb;
                    const float* b2 =
                        B + static_cast<long>(j + 2) * ldb;
                    const float* b3 =
                        B + static_cast<long>(j + 3) * ldb;

                    __m256 acc00 = _mm256_setzero_ps();
                    __m256 acc01 = _mm256_setzero_ps();
                    __m256 acc02 = _mm256_setzero_ps();
                    __m256 acc03 = _mm256_setzero_ps();

                    __m256 acc10 = _mm256_setzero_ps();
                    __m256 acc11 = _mm256_setzero_ps();
                    __m256 acc12 = _mm256_setzero_ps();
                    __m256 acc13 = _mm256_setzero_ps();

                    __m256 acc20 = _mm256_setzero_ps();
                    __m256 acc21 = _mm256_setzero_ps();
                    __m256 acc22 = _mm256_setzero_ps();
                    __m256 acc23 = _mm256_setzero_ps();

                    int p = 0;

                    for (; p + 7 < K; p += 8) {
                        if ((p % FLOATS_PER_CACHE_LINE == 0) &&
                            (p + PREFETCH_DISTANCE < K)) {

                            _mm_prefetch(
                                reinterpret_cast<const char*>(
                                    b0 + p + PREFETCH_DISTANCE),
                                PREFETCH_HINT);
                            _mm_prefetch(
                                reinterpret_cast<const char*>(
                                    b1 + p + PREFETCH_DISTANCE),
                                PREFETCH_HINT);
                            _mm_prefetch(
                                reinterpret_cast<const char*>(
                                    b2 + p + PREFETCH_DISTANCE),
                                PREFETCH_HINT);
                            _mm_prefetch(
                                reinterpret_cast<const char*>(
                                    b3 + p + PREFETCH_DISTANCE),
                                PREFETCH_HINT);
                        }

                        const __m256 va0 = _mm256_loadu_ps(a0 + p);
                        const __m256 va1 = _mm256_loadu_ps(a1 + p);
                        const __m256 va2 = _mm256_loadu_ps(a2 + p);

                        // One temporary B register is reused.
                        __m256 vb = _mm256_loadu_ps(b0 + p);
                        acc00 = _mm256_fmadd_ps(va0, vb, acc00);
                        acc10 = _mm256_fmadd_ps(va1, vb, acc10);
                        acc20 = _mm256_fmadd_ps(va2, vb, acc20);

                        vb = _mm256_loadu_ps(b1 + p);
                        acc01 = _mm256_fmadd_ps(va0, vb, acc01);
                        acc11 = _mm256_fmadd_ps(va1, vb, acc11);
                        acc21 = _mm256_fmadd_ps(va2, vb, acc21);

                        vb = _mm256_loadu_ps(b2 + p);
                        acc02 = _mm256_fmadd_ps(va0, vb, acc02);
                        acc12 = _mm256_fmadd_ps(va1, vb, acc12);
                        acc22 = _mm256_fmadd_ps(va2, vb, acc22);

                        vb = _mm256_loadu_ps(b3 + p);
                        acc03 = _mm256_fmadd_ps(va0, vb, acc03);
                        acc13 = _mm256_fmadd_ps(va1, vb, acc13);
                        acc23 = _mm256_fmadd_ps(va2, vb, acc23);
                    }

                    float sum00 = horizontal_sum(acc00);
                    float sum01 = horizontal_sum(acc01);
                    float sum02 = horizontal_sum(acc02);
                    float sum03 = horizontal_sum(acc03);

                    float sum10 = horizontal_sum(acc10);
                    float sum11 = horizontal_sum(acc11);
                    float sum12 = horizontal_sum(acc12);
                    float sum13 = horizontal_sum(acc13);

                    float sum20 = horizontal_sum(acc20);
                    float sum21 = horizontal_sum(acc21);
                    float sum22 = horizontal_sum(acc22);
                    float sum23 = horizontal_sum(acc23);

                    for (; p < K; ++p) {
                        const float av0 = a0[p];
                        const float av1 = a1[p];
                        const float av2 = a2[p];

                        sum00 += av0 * b0[p];
                        sum01 += av0 * b1[p];
                        sum02 += av0 * b2[p];
                        sum03 += av0 * b3[p];

                        sum10 += av1 * b0[p];
                        sum11 += av1 * b1[p];
                        sum12 += av1 * b2[p];
                        sum13 += av1 * b3[p];

                        sum20 += av2 * b0[p];
                        sum21 += av2 * b1[p];
                        sum22 += av2 * b2[p];
                        sum23 += av2 * b3[p];
                    }

                    c0[j + 0] = sum00;
                    c0[j + 1] = sum01;
                    c0[j + 2] = sum02;
                    c0[j + 3] = sum03;

                    c1[j + 0] = sum10;
                    c1[j + 1] = sum11;
                    c1[j + 2] = sum12;
                    c1[j + 3] = sum13;

                    c2[j + 0] = sum20;
                    c2[j + 1] = sum21;
                    c2[j + 2] = sum22;
                    c2[j + 3] = sum23;
                }

                // Remaining columns in the N block.
                for (; j < j_end; ++j) {
                    const float* b =
                        B + static_cast<long>(j) * ldb;

                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();

                    int p = 0;

                    for (; p + 7 < K; p += 8) {
                        const __m256 vb =
                            _mm256_loadu_ps(b + p);

                        acc0 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(a0 + p), vb, acc0);
                        acc1 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(a1 + p), vb, acc1);
                        acc2 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(a2 + p), vb, acc2);
                    }

                    float sum0 = horizontal_sum(acc0);
                    float sum1 = horizontal_sum(acc1);
                    float sum2 = horizontal_sum(acc2);

                    for (; p < K; ++p) {
                        sum0 += a0[p] * b[p];
                        sum1 += a1[p] * b[p];
                        sum2 += a2[p] * b[p];
                    }

                    c0[j] = sum0;
                    c1[j] = sum1;
                    c2[j] = sum2;
                }
            }

            // One or two leftover rows.
            for (; i < i_end; ++i) {
                const float* a =
                    A + static_cast<long>(i) * lda;
                float* c =
                    C + static_cast<long>(i) * ldc;

                compute_one_row(a, B, c, jj, j_end, K, ldb);
            }
        }
    }
}
