
#include <immintrin.h>

#include "matmul.h"

// These can be overridden from the compilation command.
#ifndef PREFETCH_DISTANCE
#define PREFETCH_DISTANCE 16
#endif

#ifndef PREFETCH_HINT
#define PREFETCH_HINT _MM_HINT_T0
#endif

static constexpr int FLOATS_PER_CACHE_LINE = 16;

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;

        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;

            float acc = 0.0f;

            for (int p = 0; p < K; ++p) {
                // Issue one prefetch request per future cache line.
                if ((p % FLOATS_PER_CACHE_LINE == 0) &&
                    (p + PREFETCH_DISTANCE < K)) {
                    _mm_prefetch(
                        reinterpret_cast<const char*>(
                            a + p + PREFETCH_DISTANCE),
                        PREFETCH_HINT);

                    _mm_prefetch(
                        reinterpret_cast<const char*>(
                            b + p + PREFETCH_DISTANCE),
                        PREFETCH_HINT);
                }

                acc += a[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
