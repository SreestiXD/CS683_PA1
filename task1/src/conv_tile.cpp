
#include "convolution.h"

#define TILE_SIZE 128

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy0 = 0; oy0 < H; oy0 += TILE_SIZE) {
        for (int ox0 = 0; ox0 < W; ox0 += TILE_SIZE) {

            const int oy_end = (oy0 + TILE_SIZE < H) ? oy0 + TILE_SIZE : H;
            const int ox_end = (ox0 + TILE_SIZE < W) ? ox0 + TILE_SIZE : W;

            for (int oy = oy0; oy < oy_end; ++oy) {

                for (int ox = ox0; ox < ox_end; ox += 1) {

                    float acc0 = 0.0f;
                    /*
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;
                    float acc4 = 0.0f;
                    float acc5 = 0.0f;
                    float acc6 = 0.0f;
                    float acc7 = 0.0f;
                    */

                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {

                            const int input_base =
                                (oy + ky) * in_stride + ox + kx;

                            const float weight = ker[ky * K + kx];

                            acc0 += in[input_base + 0] * weight;
                            /*
                            acc1 += in[input_base + 1] * weight;
                            acc2 += in[input_base + 2] * weight;
                            acc3 += in[input_base + 3] * weight;
                            acc4 += in[input_base + 4] * weight;
                            acc5 += in[input_base + 5] * weight;
                            acc6 += in[input_base + 6] * weight;
                            acc7 += in[input_base + 7] * weight;
                            */
                        }
                    }

                    out[oy * W + ox + 0] = acc0;
                    /*
                    out[oy * W + ox + 1] = acc1;
                    out[oy * W + ox + 2] = acc2;
                    out[oy * W + ox + 3] = acc3;
                    out[oy * W + ox + 4] = acc4;
                    out[oy * W + ox + 5] = acc5;
                    out[oy * W + ox + 6] = acc6;
                    out[oy * W + ox + 7] = acc7;
                    */
                }
            }
        }
    }
}
