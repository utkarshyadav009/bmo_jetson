// Fused matvec: per-row tier bases + shared-memory in-row prefix scan (matches global block_offset layout).
#if defined(__CUDACC__)
#define _Float32 float
#define _Float64 double
#define _Float32x float
#define _Float64x double
#define _Float128 double
#endif
#include "bmo.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <stdexcept>

namespace {

__device__ inline float fp16_to_fp32_device(ggml_fp16_t value) {
    __half_raw raw;
    raw.x = value;
    const __half half_value = *reinterpret_cast<const __half *>(&raw);
    return __half2float(half_value);
}

__global__ void unpack_kernel(
    const uint8_t * packed_weights,
    const uint8_t * packed_mask,
    int rows,
    int cols,
    int n_2bit_bytes,
    int n_4bit_bytes,
    int n_8bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    const int32_t * block_offset,
    const ggml_fp16_t * fp16_values,
    int block_size,
    float * out_w) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (pos >= total) {
        return;
    }

    const uint8_t * stream2 = packed_weights;
    const uint8_t * stream4 = packed_weights + n_2bit_bytes;
    const uint8_t * stream8 = packed_weights + n_2bit_bytes + n_4bit_bytes;

    const int block_idx = pos / block_size;
    const int in_block = pos - block_idx * block_size;
    const uint8_t mbyte = packed_mask[block_idx / 4];
    const uint8_t tier = (mbyte >> ((block_idx % 4) * 2)) & 0x3;
    const int off = block_offset[block_idx];

    float v = 0.0f;
    if (tier == 0) {
        v = fp16_to_fp32_device(fp16_values[off + in_block]);
    } else if (tier == 1) {
        const uint8_t q = stream8[off + in_block];
        v = ((float) q - zp_int8) * scale_int8;
    } else if (tier == 2) {
        const int idx4 = off + in_block;
        const uint8_t b = stream4[idx4 / 2];
        const uint8_t q = (idx4 % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
        v = ((float) q - zp_int4) * scale_int4;
    } else {
        const int idx2 = off + in_block;
        const uint8_t b = stream2[idx2 / 4];
        const uint8_t q = (b >> ((idx2 % 4) * 2)) & 0x3;
        v = ((float) q - zp_low) * scale_low;
    }
    out_w[pos] = v;
}

} // namespace

// V2: 8 rows per block, 1 warp per row, blockDim.x = 256 fixed.
template <int ROWS_PER_BLOCK = 8>
__global__ void fused_dequant_matvec_kernel_v2(
    const uint8_t * __restrict__ pw,
    const uint8_t * __restrict__ pm,
    const __half * __restrict__ fp16_vals,
    const int32_t * __restrict__ row_c2_g,
    const int32_t * __restrict__ row_c4_g,
    const int32_t * __restrict__ row_c8_g,
    int rows, int cols, int block_size,
    int n_2bit_bytes, int n_4bit_bytes,
    float scale_low, float scale_int4, float scale_int8,
    float zp_low, float zp_int4, float zp_int8,
    const float * __restrict__ x,
    float * __restrict__ y) {
    static_assert(ROWS_PER_BLOCK == 8, "v2 expects 8 rows/block (256 threads)");

    const int tid = threadIdx.x;
    const int row_in_block = tid >> 5; // tid / 32
    const int lane = tid & 31;         // tid % 32
    const int row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;

    if (row >= rows) return;

    const int blocks_per_row = cols / block_size; // assumes cols % 32 == 0

    extern __shared__ uint8_t smem_raw[];
    const size_t tier_bytes = (size_t) ROWS_PER_BLOCK * (size_t) blocks_per_row;
    const size_t tier_pad = (4 - (tier_bytes % 4)) % 4;
    uint8_t * tiers_base = smem_raw;
    int32_t * offs_base =
        reinterpret_cast<int32_t *>(tiers_base + tier_bytes + tier_pad);

    uint8_t * s_tier = tiers_base + (size_t) row_in_block * (size_t) blocks_per_row;
    int32_t * s_off = offs_base + (size_t) row_in_block * (size_t) blocks_per_row;

    // ---- Load tiers for this row (32 threads cooperate, stride 32) ----
    for (int b = lane; b < blocks_per_row; b += 32) {
        const int64_t b_global = (int64_t) row * blocks_per_row + b;
        const uint8_t mbyte = pm[b_global >> 2];
        s_tier[b] = (mbyte >> ((b_global & 3) * 2)) & 0x3;
    }
    __syncthreads();

    // Per-row tier bases + in-row prefix scan (matches global unpack_layer_to_f32_blockwise).
    if (lane == 0) {
        const int32_t rb2 = row_c2_g[row];
        const int32_t rb4 = row_c4_g[row];
        const int32_t rb8 = row_c8_g[row];
        const int32_t blocks_before = row * blocks_per_row;
        const int32_t rb16 = (blocks_before - rb2 / block_size - rb4 / block_size - rb8 / block_size) *
                             block_size;

        int32_t o2 = rb2;
        int32_t o4 = rb4;
        int32_t o8 = rb8;
        int32_t o16 = rb16;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t tier = s_tier[b];
            int32_t off;
            if (tier == 0) {
                off = o16;
                o16 += block_size;
            } else if (tier == 1) {
                off = o8;
                o8 += block_size;
            } else if (tier == 2) {
                off = o4;
                o4 += block_size;
            } else {
                off = o2;
                o2 += block_size;
            }
            s_off[b] = off;
        }
    }
    __syncthreads();

    // ---- Matvec: 32 threads stride through cols ----
    const uint8_t * stream8 = pw + n_2bit_bytes + n_4bit_bytes;
    const uint8_t * stream4 = pw + n_2bit_bytes;
    const uint8_t * stream2 = pw;

    float acc = 0.0f;
    const int n_iters = cols >> 5; // cols / 32

#pragma unroll 4
    for (int k = 0; k < n_iters; ++k) {
        const int c = (k << 5) + lane;
        const uint8_t tier = s_tier[k];
        const int off = s_off[k];

        float w;
        if (tier == 0) {
            w = __half2float(fp16_vals[off + lane]);
        } else if (tier == 1) {
            const uint8_t q = stream8[off + lane];
            w = ((float) q - zp_int8) * scale_int8;
        } else if (tier == 2) {
            const int idx = off + lane;
            const uint8_t bb = stream4[idx >> 1];
            const uint8_t q = (idx & 1) ? ((bb >> 4) & 0xF) : (bb & 0xF);
            w = ((float) q - zp_int4) * scale_int4;
        } else {
            const int idx = off + lane;
            const uint8_t bb = stream2[idx >> 2];
            const uint8_t q = (bb >> ((idx & 3) * 2)) & 0x3;
            w = ((float) q - zp_low) * scale_low;
        }
        acc += w * x[c];
    }

    // ---- Pure warp-shuffle reduction ----
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) y[row] = acc;
}

void launch_unpack_kernel_streamed(
    const void * packed_weights,
    const void * packed_mask,
    const void * fp16_values,
    const int32_t * block_offset,
    int32_t rows,
    int32_t cols,
    int32_t block_size,
    int32_t n_2bit_bytes,
    int32_t n_4bit_bytes,
    int32_t n_8bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    float * out_w) {
    const int threads = 256;
    const int total = rows * cols;
    const int blocks = (total + threads - 1) / threads;

    unpack_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint8_t *>(packed_weights),
        reinterpret_cast<const uint8_t *>(packed_mask),
        rows,
        cols,
        n_2bit_bytes,
        n_4bit_bytes,
        n_8bit_bytes,
        scale_low,
        scale_int4,
        scale_int8,
        zp_low,
        zp_int4,
        zp_int8,
        block_offset,
        reinterpret_cast<const ggml_fp16_t *>(fp16_values),
        block_size > 0 ? block_size : 32,
        out_w);
}

__global__ void rmsnorm_kernel(
    const float * __restrict__ x,
    const float * __restrict__ weight,
    float eps, int n_embd, float * __restrict__ y) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    float sum_sq = 0.0f;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        const float v = x[i];
        sum_sq += v * v;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    float total_sq = 0.0f;
    if (warp_id == 0) {
        total_sq = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            total_sq += __shfl_down_sync(0xff, total_sq, offset);
        }
        if (lane == 0) warp_sums[0] = total_sq;
    }
    __syncthreads();
    total_sq = warp_sums[0];

    const float scale = rsqrtf(total_sq / (float) n_embd + eps);

    for (int i = tid; i < n_embd; i += blockDim.x) {
        y[i] = x[i] * scale * weight[i];
    }
}

void launch_rmsnorm(
    const float * x_dev,
    const float * weight_dev,
    float eps,
    int n_embd,
    float * y_dev,
    void * stream) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    rmsnorm_kernel<<<1, 256, 0, s>>>(x_dev, weight_dev, eps, n_embd, y_dev);
}

__global__ void residual_add_kernel(
    const float * __restrict__ a,
    const float * __restrict__ b,
    int n,
    float * __restrict__ y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = a[idx] + b[idx];
}

void launch_residual_add(
    const float * a_dev,
    const float * b_dev,
    int n,
    float * y_dev,
    void * stream) {
    const int threads = 256;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    residual_add_kernel<<<(n + threads - 1) / threads, threads, 0, s>>>(a_dev, b_dev, n, y_dev);
}

// Interleaved RoPE: pairs are (idx_r, idx_i) = (2i, 2i+1) within each head.
// Tensor layout follows ggml's [head_dim, n_heads, n_token] (ne[0..2]).
__global__ void rope_interleaved_kernel(
    const float * __restrict__ x,
    int n_heads, int head_dim, int n_token,
    int pos_base, float theta_base,
    float * __restrict__ y,
    const int * __restrict__ pos_dev = nullptr) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int tid = threadIdx.x;
    if (head >= n_heads || token >= n_token) return;

    const int half = head_dim >> 1;
    const size_t off_in_tensor = ((size_t) token * n_heads + head) * head_dim;
    const float * x_head = x + off_in_tensor;
    float * y_head = y + off_in_tensor;

    const int effective_pos = (pos_dev != nullptr) ? (*pos_dev) : pos_base;
    const int pos = effective_pos + token;

    for (int i = tid; i < half; i += blockDim.x) {
        const float exponent = (float) (2 * i) / (float) head_dim;
        const float inv_freq = __powf(theta_base, -exponent);
        const float angle = (float) pos * inv_freq;
        float cs, sn;
        __sincosf(angle, &sn, &cs);

        const int idx_r = 2 * i;
        const int idx_i = 2 * i + 1;

        const float xr = x_head[idx_r];
        const float xi = x_head[idx_i];

        y_head[idx_r] = xr * cs - xi * sn;
        y_head[idx_i] = xr * sn + xi * cs;
    }
}

void launch_rope_interleaved(
    const float * x_dev,
    int n_heads, int head_dim, int n_token,
    int pos_base, float theta_base,
    float * y_dev,
    void * stream,
    const int * pos_dev) {
    const int threads = std::min(64, head_dim / 2);
    dim3 grid((unsigned) n_heads, (unsigned) n_token);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    rope_interleaved_kernel<<<grid, threads, 0, s>>>(
        x_dev, n_heads, head_dim, n_token, pos_base, theta_base, y_dev, pos_dev);
}

// SwiGLU on a split [gate | up] vector: y[i] = silu(gate[i]) * up[i].
// Input  layout: h[0..d_ff)     = gate, h[d_ff..2*d_ff) = up.
// Output layout: y[0..d_ff)     = silu(gate) * up.
__global__ void swiglu_split_kernel(const float * __restrict__ h, int d_ff, float * __restrict__ y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_ff) return;

    const float gate = h[idx];
    const float up   = h[idx + d_ff];
    const float silu = gate / (1.0f + __expf(-gate));
    y[idx] = silu * up;
}

void launch_swiglu_split(const float * h_dev, int d_ff, float * y_dev, void * stream) {
    const int threads = 256;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    swiglu_split_kernel<<<(d_ff + threads - 1) / threads, threads, 0, s>>>(h_dev, d_ff, y_dev);
}

void launch_fused_dequant_matvec(
    const void * pw,
    const void * pm,
    const void * fp16_vals,
    const int32_t * row_c2,
    const int32_t * row_c4,
    const int32_t * row_c8,
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
    const float * x,
    float * y,
    void * stream) {
    constexpr int ROWS_PER_BLOCK = 8;
    const int blocks_per_row = cols / block_size;
    const size_t tier_region = (size_t) ROWS_PER_BLOCK * (size_t) blocks_per_row;
    const size_t tier_pad = (4 - (tier_region % 4)) % 4;
    const size_t off_region = (size_t) ROWS_PER_BLOCK * (size_t) blocks_per_row * sizeof(int32_t);
    const size_t smem_bytes = tier_region + tier_pad + off_region;

    const int n_blocks = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    fused_dequant_matvec_kernel_v2<ROWS_PER_BLOCK><<<n_blocks, 256, smem_bytes, s>>>(
        reinterpret_cast<const uint8_t *>(pw),
        reinterpret_cast<const uint8_t *>(pm),
        reinterpret_cast<const __half *>(fp16_vals),
        row_c2,
        row_c4,
        row_c8,
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

#ifndef BMO_JETSON
void launch_unpack_kernel(
    const device_packed_t * dp,
    int32_t rows,
    int32_t cols,
    int32_t n_2bit_bytes,
    int32_t n_4bit_bytes,
    int32_t n_8bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    float * out_w) {
    if (!dp || !dp->is_valid) {
        throw std::runtime_error("launch_unpack_kernel: invalid device_packed_t");
    }

    const int threads = 256;
    const int total = rows * cols;
    const int blocks = (total + threads - 1) / threads;

    unpack_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint8_t *>(dp->packed_weights),
        reinterpret_cast<const uint8_t *>(dp->packed_mask),
        rows,
        cols,
        n_2bit_bytes,
        n_4bit_bytes,
        n_8bit_bytes,
        scale_low,
        scale_int4,
        scale_int8,
        zp_low,
        zp_int4,
        zp_int8,
        reinterpret_cast<const int32_t *>(dp->block_offset),
        reinterpret_cast<const ggml_fp16_t *>(dp->fp16_values),
        dp->block_size > 0 ? dp->block_size : 32,
        out_w);
}
#endif

// ---------------------------------------------------------------------------
// Standard Moshi Q4_0 Native sm_87 GEMV dispatch
// ---------------------------------------------------------------------------
#ifndef QK4_0
#define QK4_0 32
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

struct bmo_block_q4_0 {
    half d;
    uint8_t qs[QK4_0 / 2];
};

struct bmo_block_q8_1 {
    half2 ds;
    int8_t qs[QK8_1];
};

static __device__ __forceinline__ int bmo_get_int_b2(const void * x, int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32 = (int) x16[2 * i32 + 0];
    x32 |= ((int) x16[2 * i32 + 1]) << 16;
    return x32;
}

static __device__ __forceinline__ float bmo_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

static __device__ __forceinline__ float bmo_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__global__ void bmo_quantize_row_q8_1_kernel(
    const float * __restrict__ x,
    bmo_block_q8_1 * __restrict__ y,
    int cols) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = cols / QK8_1;
    if (i >= nb) return;

    const float * src = x + i * QK8_1;
    float amax = 0.0f;
    for (int j = 0; j < QK8_1; ++j) {
        const float v = fabsf(src[j]);
        if (v > amax) amax = v;
    }

    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    float sum = 0.0f;
    for (int j = 0; j < QK8_1; ++j) {
        const float v = src[j];
        const int q0 = (int) roundf(v * id);
        const int8_t q = (int8_t) max(-128, min(127, q0));
        y[i].qs[j] = q;
        sum += v;
    }

    y[i].ds = make_half2(__float2half(d), __float2half(sum));
}

__global__ void bmo_gemv_q4_0_q8_1_kernel(
    const bmo_block_q4_0 * __restrict__ W,
    const bmo_block_q8_1 * __restrict__ x,
    float * __restrict__ y,
    int rows,
    int cols) {

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_blocks_per_row = cols / QK4_0;

    if (warp_id >= rows) return;

    const bmo_block_q4_0 * row_W = W + (size_t) warp_id * num_blocks_per_row;

    const int kbx_lane = lane_id / 2;
    const int iqs = (lane_id % 2) * 2;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    int kbx_base = 0;
    #pragma unroll 2
    for (; kbx_base + 64 <= num_blocks_per_row; kbx_base += 64) {
        const int kbx0 = kbx_base + 0  + kbx_lane;
        const int kbx1 = kbx_base + 16 + kbx_lane;
        const int kbx2 = kbx_base + 32 + kbx_lane;
        const int kbx3 = kbx_base + 48 + kbx_lane;

        const bmo_block_q4_0 * bq4_0 = &row_W[kbx0];
        const bmo_block_q4_0 * bq4_1 = &row_W[kbx1];
        const bmo_block_q4_0 * bq4_2 = &row_W[kbx2];
        const bmo_block_q4_0 * bq4_3 = &row_W[kbx3];

        const bmo_block_q8_1 * bq8_0 = &x[kbx0];
        const bmo_block_q8_1 * bq8_1 = &x[kbx1];
        const bmo_block_q8_1 * bq8_2 = &x[kbx2];
        const bmo_block_q8_1 * bq8_3 = &x[kbx3];

        const float d4_0 = __half2float(bq4_0->d);
        const float d4_1 = __half2float(bq4_1->d);
        const float d4_2 = __half2float(bq4_2->d);
        const float d4_3 = __half2float(bq4_3->d);

        const float2 ds8_0 = __half22float2(bq8_0->ds);
        const float2 ds8_1 = __half22float2(bq8_1->ds);
        const float2 ds8_2 = __half22float2(bq8_2->ds);
        const float2 ds8_3 = __half22float2(bq8_3->ds);

        const int v0_0 = bmo_get_int_b2(bq4_0->qs, iqs + 0);
        const int v0_1 = bmo_get_int_b2(bq4_0->qs, iqs + 1);
        const int v1_0 = bmo_get_int_b2(bq4_1->qs, iqs + 0);
        const int v1_1 = bmo_get_int_b2(bq4_1->qs, iqs + 1);
        const int v2_0 = bmo_get_int_b2(bq4_2->qs, iqs + 0);
        const int v2_1 = bmo_get_int_b2(bq4_2->qs, iqs + 1);
        const int v3_0 = bmo_get_int_b2(bq4_3->qs, iqs + 0);
        const int v3_1 = bmo_get_int_b2(bq4_3->qs, iqs + 1);

        const int * q8_0 = reinterpret_cast<const int *>(bq8_0->qs);
        const int * q8_1 = reinterpret_cast<const int *>(bq8_1->qs);
        const int * q8_2 = reinterpret_cast<const int *>(bq8_2->qs);
        const int * q8_3 = reinterpret_cast<const int *>(bq8_3->qs);

        int sumi0 = 0;
        sumi0 = __dp4a((v0_0 >> 0) & 0x0F0F0F0F, q8_0[iqs + 0], sumi0);
        sumi0 = __dp4a((v0_0 >> 4) & 0x0F0F0F0F, q8_0[iqs + 0 + 4], sumi0);
        sumi0 = __dp4a((v0_1 >> 0) & 0x0F0F0F0F, q8_0[iqs + 1], sumi0);
        sumi0 = __dp4a((v0_1 >> 4) & 0x0F0F0F0F, q8_0[iqs + 1 + 4], sumi0);

        int sumi1 = 0;
        sumi1 = __dp4a((v1_0 >> 0) & 0x0F0F0F0F, q8_1[iqs + 0], sumi1);
        sumi1 = __dp4a((v1_0 >> 4) & 0x0F0F0F0F, q8_1[iqs + 0 + 4], sumi1);
        sumi1 = __dp4a((v1_1 >> 0) & 0x0F0F0F0F, q8_1[iqs + 1], sumi1);
        sumi1 = __dp4a((v1_1 >> 4) & 0x0F0F0F0F, q8_1[iqs + 1 + 4], sumi1);

        int sumi2 = 0;
        sumi2 = __dp4a((v2_0 >> 0) & 0x0F0F0F0F, q8_2[iqs + 0], sumi2);
        sumi2 = __dp4a((v2_0 >> 4) & 0x0F0F0F0F, q8_2[iqs + 0 + 4], sumi2);
        sumi2 = __dp4a((v2_1 >> 0) & 0x0F0F0F0F, q8_2[iqs + 1], sumi2);
        sumi2 = __dp4a((v2_1 >> 4) & 0x0F0F0F0F, q8_2[iqs + 1 + 4], sumi2);

        int sumi3 = 0;
        sumi3 = __dp4a((v3_0 >> 0) & 0x0F0F0F0F, q8_3[iqs + 0], sumi3);
        sumi3 = __dp4a((v3_0 >> 4) & 0x0F0F0F0F, q8_3[iqs + 0 + 4], sumi3);
        sumi3 = __dp4a((v3_1 >> 0) & 0x0F0F0F0F, q8_3[iqs + 1], sumi3);
        sumi3 = __dp4a((v3_1 >> 4) & 0x0F0F0F0F, q8_3[iqs + 1 + 4], sumi3);

        acc0 += d4_0 * ((float) sumi0 * ds8_0.x - 4.0f * ds8_0.y);
        acc1 += d4_1 * ((float) sumi1 * ds8_1.x - 4.0f * ds8_1.y);
        acc2 += d4_2 * ((float) sumi2 * ds8_2.x - 4.0f * ds8_2.y);
        acc3 += d4_3 * ((float) sumi3 * ds8_3.x - 4.0f * ds8_3.y);
    }

    for (; kbx_base < num_blocks_per_row; kbx_base += 16) {
        const int kbx = kbx_base + kbx_lane;
        if (kbx < num_blocks_per_row) {
            const bmo_block_q4_0 * bq4 = &row_W[kbx];
            const bmo_block_q8_1 * bq8 = &x[kbx];

            const float d4 = __half2float(bq4->d);
            const float2 ds8 = __half22float2(bq8->ds);

            const int v0 = bmo_get_int_b2(bq4->qs, iqs + 0);
            const int v1 = bmo_get_int_b2(bq4->qs, iqs + 1);

            const int * q8 = reinterpret_cast<const int *>(bq8->qs);

            int sumi = 0;
            sumi = __dp4a((v0 >> 0) & 0x0F0F0F0F, q8[iqs + 0], sumi);
            sumi = __dp4a((v0 >> 4) & 0x0F0F0F0F, q8[iqs + 0 + 4], sumi);
            sumi = __dp4a((v1 >> 0) & 0x0F0F0F0F, q8[iqs + 1], sumi);
            sumi = __dp4a((v1 >> 4) & 0x0F0F0F0F, q8[iqs + 1 + 4], sumi);

            acc0 += d4 * ((float) sumi * ds8.x - 4.0f * ds8.y);
        }
    }

    float acc = bmo_warp_reduce_sum(acc0 + acc1 + acc2 + acc3);
    if (lane_id == 0) {
        y[warp_id] = acc;
    }
}

void launch_gemv_q4_0_q8_1(
    const void * W_dev,
    const float * x_dev,
    float * y_dev,
    int rows,
    int cols,
    void * q8_scratch,
    void * stream) {

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    const int quant_threads = 256;
    const int quant_blocks = ((cols / QK8_1) + quant_threads - 1) / quant_threads;
    bmo_quantize_row_q8_1_kernel<<<quant_blocks, quant_threads, 0, s>>>(
        x_dev, reinterpret_cast<bmo_block_q8_1 *>(q8_scratch), cols);

    const int threads_per_block = 256;
    const int warps_per_block = 8;
    const int num_blocks = (rows + warps_per_block - 1) / warps_per_block;

    bmo_gemv_q4_0_q8_1_kernel<<<num_blocks, threads_per_block, 0, s>>>(
        reinterpret_cast<const bmo_block_q4_0 *>(W_dev),
        reinterpret_cast<const bmo_block_q8_1 *>(q8_scratch),
        y_dev,
        rows,
        cols);
}

__global__ void bmo_decode_attention_kernel(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    __half * __restrict__ k_cache,
    __half * __restrict__ v_cache,
    float * __restrict__ out,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int n_ctx,
    int n_past,
    int layer,
    float scale,
    const int * __restrict__ pos_dev = nullptr
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int tid = threadIdx.x;
    if (tid >= head_dim) return;

    const int effective_past = (pos_dev != nullptr) ? (*pos_dev) : n_past;
    const int q_per_kv = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1;
    const int kv_h = (n_kv_heads == n_heads) ? h : (h / q_per_kv);

    const size_t per_layer = (size_t) head_dim * (size_t) n_ctx * (size_t) n_heads;
    const size_t per_head  = (size_t) head_dim * (size_t) n_ctx;
    const size_t cache_offset = (size_t) layer * per_layer + (size_t) h * per_head + (size_t) effective_past * (size_t) head_dim;

    // Step 1: Write current K and V to cache
    if (h < n_kv_heads) {
        k_cache[cache_offset + tid] = __float2half(k[h * head_dim + tid]);
        v_cache[cache_offset + tid] = __float2half(v[h * head_dim + tid]);
    }
    __syncthreads();

    const int kv_len = effective_past + 1;

    // Fast path: kv_len == 1
    if (kv_len == 1) {
        out[h * head_dim + tid] = v[kv_h * head_dim + tid];
        return;
    }

    __shared__ float s_scores[1024];
    __shared__ float s_warp[8];

    const float q_val = q[h * head_dim + tid];
    const size_t kv_head_base = (size_t) layer * per_layer + (size_t) kv_h * per_head;

    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = (head_dim + 31) >> 5;

    for (int t = 0; t < kv_len; ++t) {
        const float k_val = __half2float(k_cache[kv_head_base + (size_t) t * (size_t) head_dim + tid]);
        float dot = q_val * k_val;
        dot = bmo_warp_reduce_sum(dot);
        if (lane_id == 0) {
            s_warp[warp_id] = dot;
        }
        __syncthreads();

        if (tid == 0) {
            float total_dot = 0.0f;
            for (int w = 0; w < num_warps; ++w) {
                total_dot += s_warp[w];
            }
            s_scores[t] = total_dot * scale;
        }
        __syncthreads();
    }

    // Parallel softmax reduction across all 128 threads in block
    float local_max = -1e20f;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        if (s_scores[t] > local_max) local_max = s_scores[t];
    }
    local_max = bmo_warp_reduce_max(local_max);
    if (lane_id == 0) {
        s_warp[warp_id] = local_max;
    }
    __syncthreads();
    if (tid == 0) {
        float m = s_warp[0];
        for (int w = 1; w < num_warps; ++w) {
            if (s_warp[w] > m) m = s_warp[w];
        }
        s_warp[0] = m;
    }
    __syncthreads();
    const float max_s = s_warp[0];

    float local_sum = 0.0f;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        float e = expf(s_scores[t] - max_s);
        s_scores[t] = e;
        local_sum += e;
    }
    local_sum = bmo_warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        s_warp[warp_id] = local_sum;
    }
    __syncthreads();
    if (tid == 0) {
        float sum_e = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            sum_e += s_warp[w];
        }
        s_warp[0] = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;
    }
    __syncthreads();
    const float inv_sum = s_warp[0];

    for (int t = tid; t < kv_len; t += blockDim.x) {
        s_scores[t] *= inv_sum;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
        const float w = s_scores[t];
        const float v_val = __half2float(v_cache[kv_head_base + (size_t) t * (size_t) head_dim + tid]);
        acc += w * v_val;
    }
    out[h * head_dim + tid] = acc;
}

void launch_decode_attention(
    const float * q_dev,
    const float * k_dev,
    const float * v_dev,
    void * k_cache_dev,
    void * v_cache_dev,
    float * out_dev,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int n_ctx,
    int n_past,
    int layer,
    void * stream,
    const int * pos_dev) {

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    float scale = 1.0f / sqrtf((float) head_dim);

    bmo_decode_attention_kernel<<<n_heads, head_dim, 0, s>>>(
        q_dev,
        k_dev,
        v_dev,
        reinterpret_cast<__half *>(k_cache_dev),
        reinterpret_cast<__half *>(v_cache_dev),
        out_dev,
        head_dim,
        n_heads,
        n_kv_heads,
        n_ctx,
        n_past,
        layer,
        scale,
        pos_dev);
}
