// matmul_optimized.cpp
// 3x4 register tile
// AVX2 + FMA
// Software prefetching
// 4x SIMD unrolling in K loop

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


// ============================================================
// In-register horizontal reduction
// ============================================================
static inline float horizontal_sum(__m256 v) {

    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);

    __m128 sum = _mm_add_ps(lo, hi);

    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}


// ============================================================
// Fallback for one remaining row
// ============================================================
static inline void compute_one_row(
    const float* a,
    const float* B,
    float* c,
    int j_begin,
    int j_end,
    int K,
    int ldb)
{
    int j = j_begin;


    // --------------------------------------------------------
    // Four output columns together
    // --------------------------------------------------------
    for (; j + 3 < j_end; j += 4) {

        const float* b0 =
            B + static_cast<long>(j + 0) * ldb;

        const float* b1 =
            B + static_cast<long>(j + 1) * ldb;

        const float* b2 =
            B + static_cast<long>(j + 2) * ldb;

        const float* b3 =
            B + static_cast<long>(j + 3) * ldb;


        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();


        int p = 0;


        // ====================================================
        // 4x SIMD UNROLL
        // 4 x 8 floats = 32 floats per loop iteration
        // ====================================================
        for (; p + 31 < K; p += 32) {

            // --------------------------
            // Chunk 1 : p ... p+7
            // --------------------------
            __m256 va =
                _mm256_loadu_ps(a + p);

            acc0 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b0 + p),
                acc0);

            acc1 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b1 + p),
                acc1);

            acc2 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b2 + p),
                acc2);

            acc3 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b3 + p),
                acc3);


            // --------------------------
            // Chunk 2 : p+8 ... p+15
            // --------------------------
            va =
                _mm256_loadu_ps(a + p + 8);

            acc0 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b0 + p + 8),
                acc0);

            acc1 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b1 + p + 8),
                acc1);

            acc2 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b2 + p + 8),
                acc2);

            acc3 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b3 + p + 8),
                acc3);


            // --------------------------
            // Chunk 3 : p+16 ... p+23
            // --------------------------
            va =
                _mm256_loadu_ps(a + p + 16);

            acc0 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b0 + p + 16),
                acc0);

            acc1 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b1 + p + 16),
                acc1);

            acc2 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b2 + p + 16),
                acc2);

            acc3 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b3 + p + 16),
                acc3);


            // --------------------------
            // Chunk 4 : p+24 ... p+31
            // --------------------------
            va =
                _mm256_loadu_ps(a + p + 24);

            acc0 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b0 + p + 24),
                acc0);

            acc1 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b1 + p + 24),
                acc1);

            acc2 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b2 + p + 24),
                acc2);

            acc3 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b3 + p + 24),
                acc3);
        }


        // ----------------------------------------------------
        // Remaining complete SIMD vectors
        // ----------------------------------------------------
        for (; p + 7 < K; p += 8) {

            const __m256 va =
                _mm256_loadu_ps(a + p);

            acc0 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b0 + p),
                acc0);

            acc1 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b1 + p),
                acc1);

            acc2 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b2 + p),
                acc2);

            acc3 = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b3 + p),
                acc3);
        }


        float sum0 = horizontal_sum(acc0);
        float sum1 = horizontal_sum(acc1);
        float sum2 = horizontal_sum(acc2);
        float sum3 = horizontal_sum(acc3);


        // ----------------------------------------------------
        // Scalar remainder
        // ----------------------------------------------------
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


    // --------------------------------------------------------
    // Remaining individual columns
    // --------------------------------------------------------
    for (; j < j_end; ++j) {

        const float* b =
            B + static_cast<long>(j) * ldb;


        __m256 acc = _mm256_setzero_ps();


        int p = 0;


        // 4x SIMD unroll
        for (; p + 31 < K; p += 32) {

            __m256 va =
                _mm256_loadu_ps(a + p);

            acc = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b + p),
                acc);


            va =
                _mm256_loadu_ps(a + p + 8);

            acc = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b + p + 8),
                acc);


            va =
                _mm256_loadu_ps(a + p + 16);

            acc = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b + p + 16),
                acc);


            va =
                _mm256_loadu_ps(a + p + 24);

            acc = _mm256_fmadd_ps(
                va,
                _mm256_loadu_ps(b + p + 24),
                acc);
        }


        // Remaining SIMD vector(s)
        for (; p + 7 < K; p += 8) {

            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(a + p),
                _mm256_loadu_ps(b + p),
                acc);
        }


        float sum = horizontal_sum(acc);


        // Scalar remainder
        for (; p < K; ++p) {

            sum += a[p] * b[p];
        }


        c[j] = sum;
    }
}



// ============================================================
// Main optimized matrix multiplication
// ============================================================
void matmul_optimized(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc)
{

    // --------------------------------------------------------
    // M blocking
    // --------------------------------------------------------
    for (int ii = 0; ii < M; ii += M_BLOCK) {

        const int i_end =
            std::min(ii + M_BLOCK, M);


        // ----------------------------------------------------
        // N blocking
        // ----------------------------------------------------
        for (int jj = 0; jj < N; jj += N_BLOCK) {

            const int j_end =
                std::min(jj + N_BLOCK, N);


            int i = ii;


            // =================================================
            // Main 3 x 4 register tile
            //
            // Three rows of A
            // Four columns of B
            //
            // Computes 12 outputs simultaneously
            // =================================================
            for (; i + 2 < i_end; i += 3) {

                const float* a0 =
                    A + static_cast<long>(i + 0) * lda;

                const float* a1 =
                    A + static_cast<long>(i + 1) * lda;

                const float* a2 =
                    A + static_cast<long>(i + 2) * lda;


                float* c0 =
                    C + static_cast<long>(i + 0) * ldc;

                float* c1 =
                    C + static_cast<long>(i + 1) * ldc;

                float* c2 =
                    C + static_cast<long>(i + 2) * ldc;


                int j = jj;


                // =============================================
                // Four columns of B per register tile
                // =============================================
                for (; j + 3 < j_end; j += 4) {

                    const float* b0 =
                        B + static_cast<long>(j + 0) * ldb;

                    const float* b1 =
                        B + static_cast<long>(j + 1) * ldb;

                    const float* b2 =
                        B + static_cast<long>(j + 2) * ldb;

                    const float* b3 =
                        B + static_cast<long>(j + 3) * ldb;


                    // -----------------------------------------
                    // 12 accumulators
                    // -----------------------------------------
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


                    // =================================================
                    // 4x SIMD UNROLL
                    //
                    // Per iteration:
                    //
                    // p     ... p+7
                    // p+8   ... p+15
                    // p+16  ... p+23
                    // p+24  ... p+31
                    //
                    // 32 floats / iteration
                    // =================================================
                    for (; p + 31 < K; p += 32) {

                        // ---------------------------------------------
                        // Software prefetch
                        // ---------------------------------------------
                        if (p + PREFETCH_DISTANCE < K) {

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


                        // =============================================
                        // CHUNK 1
                        // p ... p+7
                        // =============================================

                        __m256 va0 =
                            _mm256_loadu_ps(a0 + p);

                        __m256 va1 =
                            _mm256_loadu_ps(a1 + p);

                        __m256 va2 =
                            _mm256_loadu_ps(a2 + p);


                        __m256 vb =
                            _mm256_loadu_ps(b0 + p);


                        acc00 =
                            _mm256_fmadd_ps(va0, vb, acc00);

                        acc10 =
                            _mm256_fmadd_ps(va1, vb, acc10);

                        acc20 =
                            _mm256_fmadd_ps(va2, vb, acc20);


                        vb =
                            _mm256_loadu_ps(b1 + p);


                        acc01 =
                            _mm256_fmadd_ps(va0, vb, acc01);

                        acc11 =
                            _mm256_fmadd_ps(va1, vb, acc11);

                        acc21 =
                            _mm256_fmadd_ps(va2, vb, acc21);


                        vb =
                            _mm256_loadu_ps(b2 + p);


                        acc02 =
                            _mm256_fmadd_ps(va0, vb, acc02);

                        acc12 =
                            _mm256_fmadd_ps(va1, vb, acc12);

                        acc22 =
                            _mm256_fmadd_ps(va2, vb, acc22);


                        vb =
                            _mm256_loadu_ps(b3 + p);


                        acc03 =
                            _mm256_fmadd_ps(va0, vb, acc03);

                        acc13 =
                            _mm256_fmadd_ps(va1, vb, acc13);

                        acc23 =
                            _mm256_fmadd_ps(va2, vb, acc23);



                        // =============================================
                        // CHUNK 2
                        // p+8 ... p+15
                        // =============================================

                        va0 =
                            _mm256_loadu_ps(a0 + p + 8);

                        va1 =
                            _mm256_loadu_ps(a1 + p + 8);

                        va2 =
                            _mm256_loadu_ps(a2 + p + 8);


                        vb =
                            _mm256_loadu_ps(b0 + p + 8);


                        acc00 =
                            _mm256_fmadd_ps(va0, vb, acc00);

                        acc10 =
                            _mm256_fmadd_ps(va1, vb, acc10);

                        acc20 =
                            _mm256_fmadd_ps(va2, vb, acc20);


                        vb =
                            _mm256_loadu_ps(b1 + p + 8);


                        acc01 =
                            _mm256_fmadd_ps(va0, vb, acc01);

                        acc11 =
                            _mm256_fmadd_ps(va1, vb, acc11);

                        acc21 =
                            _mm256_fmadd_ps(va2, vb, acc21);


                        vb =
                            _mm256_loadu_ps(b2 + p + 8);


                        acc02 =
                            _mm256_fmadd_ps(va0, vb, acc02);

                        acc12 =
                            _mm256_fmadd_ps(va1, vb, acc12);

                        acc22 =
                            _mm256_fmadd_ps(va2, vb, acc22);


                        vb =
                            _mm256_loadu_ps(b3 + p + 8);


                        acc03 =
                            _mm256_fmadd_ps(va0, vb, acc03);

                        acc13 =
                            _mm256_fmadd_ps(va1, vb, acc13);

                        acc23 =
                            _mm256_fmadd_ps(va2, vb, acc23);



                        // =============================================
                        // CHUNK 3
                        // p+16 ... p+23
                        // =============================================

                        va0 =
                            _mm256_loadu_ps(a0 + p + 16);

                        va1 =
                            _mm256_loadu_ps(a1 + p + 16);

                        va2 =
                            _mm256_loadu_ps(a2 + p + 16);


                        vb =
                            _mm256_loadu_ps(b0 + p + 16);


                        acc00 =
                            _mm256_fmadd_ps(va0, vb, acc00);

                        acc10 =
                            _mm256_fmadd_ps(va1, vb, acc10);

                        acc20 =
                            _mm256_fmadd_ps(va2, vb, acc20);


                        vb =
                            _mm256_loadu_ps(b1 + p + 16);


                        acc01 =
                            _mm256_fmadd_ps(va0, vb, acc01);

                        acc11 =
                            _mm256_fmadd_ps(va1, vb, acc11);

                        acc21 =
                            _mm256_fmadd_ps(va2, vb, acc21);


                        vb =
                            _mm256_loadu_ps(b2 + p + 16);


                        acc02 =
                            _mm256_fmadd_ps(va0, vb, acc02);

                        acc12 =
                            _mm256_fmadd_ps(va1, vb, acc12);

                        acc22 =
                            _mm256_fmadd_ps(va2, vb, acc22);


                        vb =
                            _mm256_loadu_ps(b3 + p + 16);


                        acc03 =
                            _mm256_fmadd_ps(va0, vb, acc03);

                        acc13 =
                            _mm256_fmadd_ps(va1, vb, acc13);

                        acc23 =
                            _mm256_fmadd_ps(va2, vb, acc23);



                        // =============================================
                        // CHUNK 4
                        // p+24 ... p+31
                        // =============================================

                        va0 =
                            _mm256_loadu_ps(a0 + p + 24);

                        va1 =
                            _mm256_loadu_ps(a1 + p + 24);

                        va2 =
                            _mm256_loadu_ps(a2 + p + 24);


                        vb =
                            _mm256_loadu_ps(b0 + p + 24);


                        acc00 =
                            _mm256_fmadd_ps(va0, vb, acc00);

                        acc10 =
                            _mm256_fmadd_ps(va1, vb, acc10);

                        acc20 =
                            _mm256_fmadd_ps(va2, vb, acc20);


                        vb =
                            _mm256_loadu_ps(b1 + p + 24);


                        acc01 =
                            _mm256_fmadd_ps(va0, vb, acc01);

                        acc11 =
                            _mm256_fmadd_ps(va1, vb, acc11);

                        acc21 =
                            _mm256_fmadd_ps(va2, vb, acc21);


                        vb =
                            _mm256_loadu_ps(b2 + p + 24);


                        acc02 =
                            _mm256_fmadd_ps(va0, vb, acc02);

                        acc12 =
                            _mm256_fmadd_ps(va1, vb, acc12);

                        acc22 =
                            _mm256_fmadd_ps(va2, vb, acc22);


                        vb =
                            _mm256_loadu_ps(b3 + p + 24);


                        acc03 =
                            _mm256_fmadd_ps(va0, vb, acc03);

                        acc13 =
                            _mm256_fmadd_ps(va1, vb, acc13);

                        acc23 =
                            _mm256_fmadd_ps(va2, vb, acc23);
                    }



                    // =================================================
                    // Remaining full SIMD vectors
                    // =================================================
                    for (; p + 7 < K; p += 8) {

                        const __m256 va0 =
                            _mm256_loadu_ps(a0 + p);

                        const __m256 va1 =
                            _mm256_loadu_ps(a1 + p);

                        const __m256 va2 =
                            _mm256_loadu_ps(a2 + p);


                        __m256 vb =
                            _mm256_loadu_ps(b0 + p);


                        acc00 =
                            _mm256_fmadd_ps(va0, vb, acc00);

                        acc10 =
                            _mm256_fmadd_ps(va1, vb, acc10);

                        acc20 =
                            _mm256_fmadd_ps(va2, vb, acc20);


                        vb =
                            _mm256_loadu_ps(b1 + p);


                        acc01 =
                            _mm256_fmadd_ps(va0, vb, acc01);

                        acc11 =
                            _mm256_fmadd_ps(va1, vb, acc11);

                        acc21 =
                            _mm256_fmadd_ps(va2, vb, acc21);


                        vb =
                            _mm256_loadu_ps(b2 + p);


                        acc02 =
                            _mm256_fmadd_ps(va0, vb, acc02);

                        acc12 =
                            _mm256_fmadd_ps(va1, vb, acc12);

                        acc22 =
                            _mm256_fmadd_ps(va2, vb, acc22);


                        vb =
                            _mm256_loadu_ps(b3 + p);


                        acc03 =
                            _mm256_fmadd_ps(va0, vb, acc03);

                        acc13 =
                            _mm256_fmadd_ps(va1, vb, acc13);

                        acc23 =
                            _mm256_fmadd_ps(va2, vb, acc23);
                    }



                    // =================================================
                    // Horizontal reductions
                    // =================================================

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



                    // =================================================
                    // Scalar K remainder
                    // =================================================

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



                    // =================================================
                    // Store 12 outputs
                    // =================================================

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



                // =============================================
                // Remaining columns in N block
                // =============================================
                for (; j < j_end; ++j) {

                    const float* b =
                        B + static_cast<long>(j) * ldb;


                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();


                    int p = 0;


                    // -----------------------------------------
                    // 4x SIMD unroll
                    // -----------------------------------------
                    for (; p + 31 < K; p += 32) {

                        __m256 vb =
                            _mm256_loadu_ps(b + p);


                        acc0 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a0 + p),
                                vb,
                                acc0);

                        acc1 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a1 + p),
                                vb,
                                acc1);

                        acc2 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a2 + p),
                                vb,
                                acc2);



                        vb =
                            _mm256_loadu_ps(b + p + 8);


                        acc0 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a0 + p + 8),
                                vb,
                                acc0);

                        acc1 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a1 + p + 8),
                                vb,
                                acc1);

                        acc2 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a2 + p + 8),
                                vb,
                                acc2);



                        vb =
                            _mm256_loadu_ps(b + p + 16);


                        acc0 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a0 + p + 16),
                                vb,
                                acc0);

                        acc1 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a1 + p + 16),
                                vb,
                                acc1);

                        acc2 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a2 + p + 16),
                                vb,
                                acc2);



                        vb =
                            _mm256_loadu_ps(b + p + 24);


                        acc0 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a0 + p + 24),
                                vb,
                                acc0);

                        acc1 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a1 + p + 24),
                                vb,
                                acc1);

                        acc2 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a2 + p + 24),
                                vb,
                                acc2);
                    }


                    // -----------------------------------------
                    // Remaining complete SIMD vectors
                    // -----------------------------------------
                    for (; p + 7 < K; p += 8) {

                        const __m256 vb =
                            _mm256_loadu_ps(b + p);


                        acc0 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a0 + p),
                                vb,
                                acc0);

                        acc1 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a1 + p),
                                vb,
                                acc1);

                        acc2 =
                            _mm256_fmadd_ps(
                                _mm256_loadu_ps(a2 + p),
                                vb,
                                acc2);
                    }


                    float sum0 = horizontal_sum(acc0);
                    float sum1 = horizontal_sum(acc1);
                    float sum2 = horizontal_sum(acc2);


                    // -----------------------------------------
                    // Scalar remainder
                    // -----------------------------------------
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



            // =================================================
            // One or two remaining rows
            // =================================================
            for (; i < i_end; ++i) {

                const float* a =
                    A + static_cast<long>(i) * lda;

                float* c =
                    C + static_cast<long>(i) * ldc;


                compute_one_row(
                    a,
                    B,
                    c,
                    jj,
                    j_end,
                    K,
                    ldb);
            }
        }
    }
}
