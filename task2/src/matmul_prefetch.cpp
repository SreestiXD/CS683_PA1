// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING
// matmul_prefetch.cpp
// Intermediate version: cache-blocked, register-tiled AVX2 SGEMM.
// Software prefetching will be added only after this version is correct.

#include <algorithm>
#include <immintrin.h>

#include "matmul.h"

// Tuned Parameters
#ifndef M_BLOCK
#define M_BLOCK 8
#endif

#ifndef N_BLOCK
#define N_BLOCK 32
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
    for (int lane = 0; lane < 8; ++lane) {
        sum += partial[lane];
    }

    return sum;
}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     int lda, int ldb, int ldc) {
    // ii selects a block of rows from A and C.
    for (int ii = 0; ii < M; ii += M_BLOCK) {
        const int i_end = std::min(ii + M_BLOCK, M);

        // jj selects a block of rows from B / columns from C.
        for (int jj = 0; jj < N; jj += N_BLOCK) {
            const int j_end = std::min(jj + N_BLOCK, N);

            // Work only inside the current M_BLOCK x N_BLOCK output tile.
            for (int i = ii; i < i_end; ++i) {
                const float* a =
                    A + static_cast<long>(i) * lda;

                int j = jj;

                // Existing 1x4 register-tiled AVX2 micro-kernel.
                for (; j + 3 < j_end; j += 4) {
                    const float* b0 =
                        B + static_cast<long>(j + 0) * ldb;
                    const float* b1 =
                        B + static_cast<long>(j + 1) * ldb;
                    const float* b2 =
                        B + static_cast<long>(j + 2) * ldb;
                    const float* b3 =
                        B + static_cast<long>(j + 3) * ldb;

                    __m256 vacc0 = _mm256_setzero_ps();
                    __m256 vacc1 = _mm256_setzero_ps();
                    __m256 vacc2 = _mm256_setzero_ps();
                    __m256 vacc3 = _mm256_setzero_ps();

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
                        __m256 va = _mm256_loadu_ps(a + p);

                        __m256 vb0 = _mm256_loadu_ps(b0 + p);
                        __m256 vb1 = _mm256_loadu_ps(b1 + p);
                        __m256 vb2 = _mm256_loadu_ps(b2 + p);
                        __m256 vb3 = _mm256_loadu_ps(b3 + p);

                        vacc0 =
                            _mm256_fmadd_ps(va, vb0, vacc0);
                        vacc1 =
                            _mm256_fmadd_ps(va, vb1, vacc1);
                        vacc2 =
                            _mm256_fmadd_ps(va, vb2, vacc2);
                        vacc3 =
                            _mm256_fmadd_ps(va, vb3, vacc3);
                    }

                    float acc0 = horizontal_sum(vacc0);
                    float acc1 = horizontal_sum(vacc1);
                    float acc2 = horizontal_sum(vacc2);
                    float acc3 = horizontal_sum(vacc3);

                    // Remaining K elements.
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

                // Remaining columns within the current N block.
                for (; j < j_end; ++j) {
                    const float* b =
                        B + static_cast<long>(j) * ldb;

                    __m256 vacc = _mm256_setzero_ps();
                    int p = 0;

                    for (; p + 7 < K; p += 8) {
                    	if ((p % FLOATS_PER_CACHE_LINE == 0) &&
			    (p + PREFETCH_DISTANCE < K)) {
			    _mm_prefetch(
				reinterpret_cast<const char*>(
				    b + p + PREFETCH_DISTANCE),
				PREFETCH_HINT);
			}
                        __m256 va = _mm256_loadu_ps(a + p);
                        __m256 vb = _mm256_loadu_ps(b + p);

                        vacc =
                            _mm256_fmadd_ps(va, vb, vacc);
                    }

                    float acc = horizontal_sum(vacc);

                    for (; p < K; ++p) {
                        acc += a[p] * b[p];
                    }

                    C[static_cast<long>(i) * ldc + j] = acc;
                }
            }
        }
    }
}
