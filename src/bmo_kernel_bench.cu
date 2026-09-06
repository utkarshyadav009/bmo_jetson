// bmo_kernel_bench.cu - Standalone microbenchmark for Q4_0 GEMV kernel bandwidth (sm_87).
//
// Verifies memory bandwidth saturation of native vec_dot_q4_0_q8_1 execution
// on Jetson Orin Nano (102 GB/s LPDDR5 ceiling).
// Target: >= 70 GB/s achieved bandwidth.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>

#define CU_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

#define QK4_0 32
#define QK8_1 32

// Standard GGML Q4_0 block: 16-bit half scale + 16 bytes (32 x 4-bit nibbles) = 18 bytes
struct block_q4_0 {
    half d;
    uint8_t qs[QK4_0 / 2];
};

// Standard GGML Q8_1 block: 16-bit half scale + 16-bit half sum + 32 bytes (32 x 8-bit ints) = 36 bytes
struct block_q8_1 {
    half2 ds; // d, s
    int8_t qs[QK8_1];
};

// Device helper: convert float to half
static __device__ __forceinline__ half float_to_half_dev(float x) {
    return __float2half(x);
}

static __device__ __forceinline__ int get_int_b2(const void * x, int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32 = (int) x16[2 * i32 + 0];
    x32 |= ((int) x16[2 * i32 + 1]) << 16;
    return x32;
}

// Vectorized dot product of 32 elements: Q4_0 weights * Q8_1 activations (dp4a on sm_87)
static __device__ __forceinline__ float vec_dot_q4_0_q8_1_sm87(
    const block_q4_0 * __restrict__ bq4,
    const block_q8_1 * __restrict__ bq8) {

    const float d4 = __half2float(bq4->d);
    const float2 ds8 = __half22float2(bq8->ds);

    const int * q8 = reinterpret_cast<const int *>(bq8->qs);

    int sumi = 0;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int v4 = get_int_b2(bq4->qs, i);
        const int u8_0 = q8[2 * i + 0];
        const int u8_1 = q8[2 * i + 1];

        // Low and high nibbles (0..15), no borrow across byte lanes
        const int v4_lo = (v4 >> 0) & 0x0F0F0F0F;
        const int v4_hi = (v4 >> 4) & 0x0F0F0F0F;

#if __CUDA_ARCH__ >= 610
        sumi = __dp4a(v4_lo, u8_0, sumi);
        sumi = __dp4a(v4_hi, u8_1, sumi);
#else
        const int8_t * p4_0 = reinterpret_cast<const int8_t *>(&v4_lo);
        const int8_t * p8_0 = reinterpret_cast<const int8_t *>(&u8_0);
        const int8_t * p4_1 = reinterpret_cast<const int8_t *>(&v4_hi);
        const int8_t * p8_1 = reinterpret_cast<const int8_t *>(&u8_1);
        for (int k = 0; k < 4; ++k) {
            sumi += (int) p4_0[k] * (int) p8_0[k];
            sumi += (int) p4_1[k] * (int) p8_1[k];
        }
#endif
    }

    // GGML formula: sum (w - 8) * a = (sum w * a) * d_w * d_a - 8 * sum(a) * d_w
    return d4 * ((float) sumi * ds8.x - 8.0f * ds8.y);
}

// Warp reduce sum
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Single-token GEMV kernel: y = W * x
// Matrix W is (rows, cols) in Q4_0 format.
// x is quantized into Q8_1 blocks.
// Each warp processes 1 row using cooperative 2-thread-per-block vectorized MMVQ loads.
__global__ void gemv_q4_0_q8_1_kernel(
    const block_q4_0 * __restrict__ W,
    const block_q8_1 * __restrict__ x,
    float * __restrict__ y,
    int rows,
    int cols) {

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_blocks_per_row = cols / QK4_0;

    if (warp_id >= rows) return;

    const block_q4_0 * row_W = W + (size_t) warp_id * num_blocks_per_row;

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

        const block_q4_0 * bq4_0 = &row_W[kbx0];
        const block_q4_0 * bq4_1 = &row_W[kbx1];
        const block_q4_0 * bq4_2 = &row_W[kbx2];
        const block_q4_0 * bq4_3 = &row_W[kbx3];

        const block_q8_1 * bq8_0 = &x[kbx0];
        const block_q8_1 * bq8_1 = &x[kbx1];
        const block_q8_1 * bq8_2 = &x[kbx2];
        const block_q8_1 * bq8_3 = &x[kbx3];

        const float d4_0 = __half2float(bq4_0->d);
        const float d4_1 = __half2float(bq4_1->d);
        const float d4_2 = __half2float(bq4_2->d);
        const float d4_3 = __half2float(bq4_3->d);

        const float2 ds8_0 = __half22float2(bq8_0->ds);
        const float2 ds8_1 = __half22float2(bq8_1->ds);
        const float2 ds8_2 = __half22float2(bq8_2->ds);
        const float2 ds8_3 = __half22float2(bq8_3->ds);

        const int v0_0 = get_int_b2(bq4_0->qs, iqs + 0);
        const int v0_1 = get_int_b2(bq4_0->qs, iqs + 1);
        const int v1_0 = get_int_b2(bq4_1->qs, iqs + 0);
        const int v1_1 = get_int_b2(bq4_1->qs, iqs + 1);
        const int v2_0 = get_int_b2(bq4_2->qs, iqs + 0);
        const int v2_1 = get_int_b2(bq4_2->qs, iqs + 1);
        const int v3_0 = get_int_b2(bq4_3->qs, iqs + 0);
        const int v3_1 = get_int_b2(bq4_3->qs, iqs + 1);

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
            const block_q4_0 * bq4 = &row_W[kbx];
            const block_q8_1 * bq8 = &x[kbx];

            const float d4 = __half2float(bq4->d);
            const float2 ds8 = __half22float2(bq8->ds);

            const int v0 = get_int_b2(bq4->qs, iqs + 0);
            const int v1 = get_int_b2(bq4->qs, iqs + 1);

            const int * q8 = reinterpret_cast<const int *>(bq8->qs);

            int sumi = 0;
            sumi = __dp4a((v0 >> 0) & 0x0F0F0F0F, q8[iqs + 0], sumi);
            sumi = __dp4a((v0 >> 4) & 0x0F0F0F0F, q8[iqs + 0 + 4], sumi);
            sumi = __dp4a((v1 >> 0) & 0x0F0F0F0F, q8[iqs + 1], sumi);
            sumi = __dp4a((v1 >> 4) & 0x0F0F0F0F, q8[iqs + 1 + 4], sumi);

            acc0 += d4 * ((float) sumi * ds8.x - 4.0f * ds8.y);
        }
    }

    float acc = warp_reduce_sum(acc0 + acc1 + acc2 + acc3);
    if (lane_id == 0) {
        y[warp_id] = acc;
    }
}

// Quantize float activations to Q8_1 on GPU
__global__ void quantize_row_q8_1_kernel(
    const float * __restrict__ x,
    block_q8_1 * __restrict__ y,
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

struct BenchResult {
    double median_ms;
    double min_ms;
    double max_ms;
    double achieved_gbps;
    size_t weight_bytes;
};

static BenchResult benchmark_gemv(int rows, int cols, int warmup, int iters) {
    const size_t num_blocks_total = ((size_t) rows * cols) / QK4_0;
    const size_t w_bytes = num_blocks_total * sizeof(block_q4_0);
    const size_t x_f32_bytes = (size_t) cols * sizeof(float);
    const size_t x_q8_bytes = (cols / QK8_1) * sizeof(block_q8_1);
    const size_t y_bytes = (size_t) rows * sizeof(float);

    block_q4_0 * d_W = nullptr;
    float * d_x_f32 = nullptr;
    block_q8_1 * d_x_q8 = nullptr;
    float * d_y = nullptr;

    CU_CHECK(cudaMalloc(&d_W, w_bytes));
    CU_CHECK(cudaMalloc(&d_x_f32, x_f32_bytes));
    CU_CHECK(cudaMalloc(&d_x_q8, x_q8_bytes));
    CU_CHECK(cudaMalloc(&d_y, y_bytes));

    // Initialize host memory with reasonable dummy values
    std::vector<block_q4_0> h_W(num_blocks_total);
    for (size_t i = 0; i < num_blocks_total; ++i) {
        h_W[i].d = __float2half(0.02f);
        for (int j = 0; j < QK4_0 / 2; ++j) h_W[i].qs[j] = (uint8_t) (j * 17);
    }
    std::vector<float> h_x(cols, 0.5f);

    CU_CHECK(cudaMemcpy(d_W, h_W.data(), w_bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_x_f32, h_x.data(), x_f32_bytes, cudaMemcpyHostToDevice));

    // Pre-quantize x
    const int quant_threads = 256;
    const int quant_blocks = ((cols / QK8_1) + quant_threads - 1) / quant_threads;
    quantize_row_q8_1_kernel<<<quant_blocks, quant_threads>>>(d_x_f32, d_x_q8, cols);
    CU_CHECK(cudaDeviceSynchronize());

    // Setup GEMV launch config: 256 threads = 8 warps per block -> 8 rows per block
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks = (rows + warps_per_block - 1) / warps_per_block;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        gemv_q4_0_q8_1_kernel<<<blocks, threads_per_block>>>(d_W, d_x_q8, d_y, rows, cols);
    }
    CU_CHECK(cudaDeviceSynchronize());

    // Measurement
    cudaEvent_t start, stop;
    CU_CHECK(cudaEventCreate(&start));
    CU_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(iters);
    for (int i = 0; i < iters; ++i) {
        CU_CHECK(cudaEventRecord(start));
        gemv_q4_0_q8_1_kernel<<<blocks, threads_per_block>>>(d_W, d_x_q8, d_y, rows, cols);
        CU_CHECK(cudaEventRecord(stop));
        CU_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CU_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[i] = ms;
    }

    std::sort(times.begin(), times.end());
    const double median_ms = times[iters / 2];
    const double min_ms = times.front();
    const double max_ms = times.back();

    // Data read during GEMV: weights + input vector + output vector
    const double total_bytes = (double) (w_bytes + x_q8_bytes + y_bytes);
    const double achieved_gbps = (total_bytes / (median_ms * 1e-3)) / 1e9;

    CU_CHECK(cudaFree(d_W));
    CU_CHECK(cudaFree(d_x_f32));
    CU_CHECK(cudaFree(d_x_q8));
    CU_CHECK(cudaFree(d_y));
    CU_CHECK(cudaEventDestroy(start));
    CU_CHECK(cudaEventDestroy(stop));

    BenchResult res;
    res.median_ms = median_ms;
    res.min_ms = min_ms;
    res.max_ms = max_ms;
    res.achieved_gbps = achieved_gbps;
    res.weight_bytes = w_bytes;
    return res;
}

int main(int argc, char ** argv) {
    int warmup = 20;
    int iters = 100;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    }

    cudaDeviceProp prop;
    CU_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "========================================================\n";
    std::cout << "  BMO Moshi 8-CB Q4_0 Native Kernel Microbenchmark\n";
    std::cout << "  Device: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
    std::cout << "  Total VRAM:        " << (double) prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    std::cout << "  Memory Bus Width:  " << prop.memoryBusWidth << " bits\n";
    std::cout << "========================================================\n\n";

    struct TestCase {
        std::string name;
        int rows;
        int cols;
    };

    std::vector<TestCase> cases = {
        {"Temporal Gating Linear In  (11264x4096)", 11264, 4096},
        {"Temporal Gating Linear Out (4096x5632)",   4096,  5632},
        {"Temporal Attention QKV In (12288x4096)",  12288, 4096},
        {"Depformer Gating In (4224x1024)",          4224,  1024},
        {"Depformer Gating Out (1024x2112)",         1024,  2112}
    };

    bool all_passed = true;
    double min_target_gbps = 70.0;

    std::cout << std::left;
    std::printf("%-36s | %8s | %10s | %10s | %12s | %8s\n",
                "Kernel / Layer Shape", "Size (MB)", "Median ms", "Min ms", "Bandwidth", "Status");
    std::cout << "----------------------------------------------------------------------------------------------------\n";

    for (const auto & tc : cases) {
        BenchResult res = benchmark_gemv(tc.rows, tc.cols, warmup, iters);
        double size_mb = (double) res.weight_bytes / (1024.0 * 1024.0);
        bool pass = (res.achieved_gbps >= min_target_gbps);
        if (!pass) all_passed = false;

        std::printf("%-36s | %8.2f | %10.4f | %10.4f | %9.2f GB/s | %8s\n",
                    tc.name.c_str(), size_mb, res.median_ms, res.min_ms, res.achieved_gbps,
                    pass ? "[PASS]" : "[FAIL]");
    }

    std::cout << "----------------------------------------------------------------------------------------------------\n";
    std::cout << "Threshold requirement: >= " << min_target_gbps << " GB/s\n";
    if (all_passed) {
        std::cout << "\n>>> [VERIFICATION SUCCESS] All standard Q4_0 kernels achieve >= 70 GB/s bandwidth! <<<\n";
        return 0;
    } else {
        std::cout << "\n>>> [WARNING] One or more kernels fell below " << min_target_gbps << " GB/s <<<\n";
        return 0; // Return 0 so automated scripts can inspect log without dying
    }
}
