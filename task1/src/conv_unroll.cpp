// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox+=8) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const int input_base = (oy + ky) * in_stride + ox + kx;
                    acc0  += in[input_base + 0] * ker[ky * K + kx];
                    acc1  += in[input_base + 1] * ker[ky * K + kx];
                    acc2  += in[input_base + 2] * ker[ky * K + kx];
                    acc3  += in[input_base + 3] * ker[ky * K + kx];
                    acc4  += in[input_base + 4] * ker[ky * K + kx];
                    acc5  += in[input_base + 5] * ker[ky * K + kx];
                    acc6  += in[input_base + 6] * ker[ky * K + kx];
                    acc7  += in[input_base + 7] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox + 0] = acc0;
            out[oy * W + ox + 1] = acc1;
            out[oy * W + ox + 2] = acc2;
            out[oy * W + ox + 3] = acc3;
            out[oy * W + ox + 4] = acc4;
            out[oy * W + ox + 5] = acc5;
            out[oy * W + ox + 6] = acc6;
            out[oy * W + ox + 7] = acc7;
        }
    }
    // TODO(student): replace this placeholder with your unrolled implementation.

}
