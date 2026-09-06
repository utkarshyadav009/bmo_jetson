// Path B prototype: per-element tier mask + warp-ballot in-warp prefix for stream offsets.
// ROWS_PER_BLOCK warps per CUDA block; one warp owns one output row (same as v2).
#if defined(__CUDACC__)
#define _Float32 float
#define _Float64 double
#define _Float32x float
#define _Float64x double
#define _Float128 double
#endif

#include "bmo_proto_kernels.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

template <int ROWS_PER_BLOCK>
__global__ void fused_dequant_matvec_kernel_proto(
    const uint8_t *__restrict__ pw,
    const uint8_t *__restrict__ pm_elem,
    const __half *__restrict__ fp16_vals,
    const int32_t *__restrict__ row_c2_g,
    const int32_t *__restrict__ row_c4_g,
    const int32_t *__restrict__ row_c8_g,
    const int32_t *__restrict__ row_c16_g,
    int rows,
    int cols,
    int block_size,
    int n_2bit_bytes,
    int n_4bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    const float *__restrict__ x,
    float *__restrict__ y) {
    static_assert(ROWS_PER_BLOCK == 8 || ROWS_PER_BLOCK == 4, "proto supports 4 or 8 rows/block");

    const int tid = threadIdx.x;
    const int row_in_block = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;

    if (row >= rows) {
        return;
    }

    const int n_iters = cols >> 5;

    int32_t o2 = row_c2_g[row];
    int32_t o4 = row_c4_g[row];
    int32_t o8 = row_c8_g[row];
    int32_t o16 = row_c16_g[row];

    const uint8_t *stream8 = pw + n_2bit_bytes + n_4bit_bytes;
    const uint8_t *stream4 = pw + n_2bit_bytes;
    const uint8_t *stream2 = pw;

    const int row_base_elems = row * cols;

    float acc = 0.0f;

#pragma unroll 4
    for (int k = 0; k < n_iters; ++k) {
        const int c = (k << 5) + lane;
        const int elem_i = row_base_elems + c;
        const uint8_t mbyte = pm_elem[elem_i >> 2];
        const uint8_t tier = (mbyte >> ((elem_i & 3) * 2)) & 0x3;

        uint32_t bm[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            bm[t] = __ballot_sync(0xffffffffu, (int) (tier == t));
        }
        const uint32_t lane_mask = (1u << lane) - 1u;
        const int ti = (int) tier;
        const int rank = __popc(bm[ti] & lane_mask);

        int32_t base;
        if (tier == 0) {
            base = o16;
        } else if (tier == 1) {
            base = o8;
        } else if (tier == 2) {
            base = o4;
        } else {
            base = o2;
        }

        const int32_t off = base + rank;

        // `off` is already the per-thread stream index (v2's block base + in-block lane).
        // Do not add `lane` again — that double-counts vs fused_dequant_matvec_kernel_v2.
        float w;
        if (tier == 0) {
            w = __half2float(fp16_vals[off]);
        } else if (tier == 1) {
            const uint8_t q = stream8[off];
            w = ((float) q - zp_int8) * scale_int8;
        } else if (tier == 2) {
            const int idx = off;
            const uint8_t bb = stream4[idx >> 1];
            const uint8_t q = (idx & 1) ? ((bb >> 4) & 0xF) : (bb & 0xF);
            w = ((float) q - zp_int4) * scale_int4;
        } else {
            const int idx = off;
            const uint8_t bb = stream2[idx >> 2];
            const uint8_t q = (bb >> ((idx & 3) * 2)) & 0x3;
            w = ((float) q - zp_low) * scale_low;
        }
        acc += w * x[c];

        o2 += (int32_t) __popc(bm[3]);
        o4 += (int32_t) __popc(bm[2]);
        o8 += (int32_t) __popc(bm[1]);
        o16 += (int32_t) __popc(bm[0]);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }

    if (lane == 0) {
        y[row] = acc;
    }
}

} // namespace

void launch_fused_dequant_matvec_proto(
    const void *pw,
    const void *pm_elem,
    const void *fp16_vals,
    const int32_t *row_c2,
    const int32_t *row_c4,
    const int32_t *row_c8,
    const int32_t *row_c16,
    int rows,
    int cols,
    int block_size,
    int n_2bit_bytes,
    int n_4bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    const float *x,
    float *y,
    int rows_per_block,
    cudaStream_t stream) {
    const int threads = rows_per_block * 32;
    const int n_blocks = (rows + rows_per_block - 1) / rows_per_block;

    if (rows_per_block == 8) {
        fused_dequant_matvec_kernel_proto<8><<<n_blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t *>(pw),
            reinterpret_cast<const uint8_t *>(pm_elem),
            reinterpret_cast<const __half *>(fp16_vals),
            row_c2,
            row_c4,
            row_c8,
            row_c16,
            rows,
            cols,
            block_size,
            n_2bit_bytes,
            n_4bit_bytes,
            scale_low,
            scale_int4,
            scale_int8,
            zp_low,
            zp_int4,
            zp_int8,
            x,
            y);
    } else if (rows_per_block == 4) {
        fused_dequant_matvec_kernel_proto<4><<<n_blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t *>(pw),
            reinterpret_cast<const uint8_t *>(pm_elem),
            reinterpret_cast<const __half *>(fp16_vals),
            row_c2,
            row_c4,
            row_c8,
            row_c16,
            rows,
            cols,
            block_size,
            n_2bit_bytes,
            n_4bit_bytes,
            scale_low,
            scale_int4,
            scale_int8,
            zp_low,
            zp_int4,
            zp_int8,
            x,
            y);
    }
}
