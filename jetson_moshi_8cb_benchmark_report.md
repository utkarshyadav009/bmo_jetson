# Jetson Orin Nano Validation & Latency Benchmarking Report: Standard Moshi 8-CB (Q4_0)

**Platform:** NVIDIA Jetson Orin Nano Developer Kit (Ampere `sm_87`, 8GB Unified LPDDR5 Memory)  
**Power Profile:** `MAXN_SUPER` (Mode ID: 2), Active High PWM Cooling (~42°C Tj)  
**Model:** `bmo_moshi_8cb_q4.gguf` (3.82 GB Q4_0, 32 Temporal Layers, 6 Depth Layers, 8 Audio Codebooks)  
**Execution Binary:** [bmo_engine](file:///home/bmo/jetson_moshi_work/bmo_jetson/build/bmo_engine)  

---

## 1. Executive Summary & Verification Matrix

| Requirement / Metric | Target Specification | Achieved Metric | Status |
| :--- | :--- | :--- | :---: |
| **LPDDR5 Bandwidth Saturation** | $\ge 70.0\text{ GB/s}$ | **$82.15 - 84.86\text{ GB/s}$** | **PASS** |
| **Depth Cascade Geometry** | 8-Codebook (`audio_heads=8/8`, `dep_ctx=8`) | Exact 8/8 heads, 6 layers, 16 heads, $d=64$ | **PASS** |
| **Resident Memory Footprint (VmRSS)** | $< 5,500\text{ MB}$ | **$5,128\text{ MB}$** (Flat, zero leak) | **PASS** |
| **100-Iteration Median Latency** | $< 80.0\text{ ms}$ (12.5 fps real-time) | **$77.0\text{ ms}$** | **PASS** |
| **Min / P90 Latency** | Stable real-time execution | **$71.0\text{ ms}$ / $81.2\text{ ms}$** | **PASS** |

---

## 2. Memory Bandwidth Microbenchmark (`bmo_kernel_bench`)

Evaluated using the native Ampere `__dp4a` vectorized kernel on Jetson Orin Nano's 128-bit LPDDR5 bus (theoretical peak: 102 GB/s):

```
====================================================================================================
Kernel / Layer Shape                 | Size (MB) |  Median ms |     Min ms |    Bandwidth |   Status
----------------------------------------------------------------------------------------------------
Temporal Gating Linear In  (11264x4096) |    24.75 |     0.3064 |     0.2838 |     84.86 GB/s |   [PASS]
Temporal Gating Linear Out (4096x5632) |    12.38 |     0.1582 |     0.1473 |     82.15 GB/s |   [PASS]
Temporal Attention QKV In (12288x4096) |    27.00 |     0.3404 |     0.3073 |     83.33 GB/s |   [PASS]
====================================================================================================
Threshold requirement: >= 70.0 GB/s achieved across all major GEMV projections (82-85 GB/s).
```

---

## 3. Root Cause Analysis & Optimizations Implemented

The initial deployment experienced latencies between ~105 ms and ~210 ms. Through targeted architectural interventions, we brought the total frame latency down to **77.0 ms**:

### A. CUDA Native Single-Token Decode Attention Kernel
* **Root Cause:** Both the 32 temporal layers and the 48 depth layers originally relied on eager CPU attention routines (`apply_attention_eager_decode` and `apply_depth_attention_eager`). To feed activations from the GPU to the CPU, `cudaStreamSynchronize(0)` was called **80 times per inference frame**, introducing massive CPU-GPU driver bubbles, context switches, and cache thrashing.
* **Fix:** Designed and implemented [`bmo_decode_attention_kernel`](file:///home/bmo/jetson_moshi_work/bmo_jetson/src/bmo_cuda_kernels.cu) directly in CUDA. The kernel:
  1. Concurrently writes incoming $K$ and $V$ vectors into unified FP16 KV cache memory.
  2. Executes an ultra-fast path for $L=1$ ($out = V$) with zero dot-product overhead.
  3. Uses warp-shuffle reductions (`__shfl_down_sync`) and shared memory for $L \le 8$ depth autoregressive attention.
  4. Runs 100% asynchronously on CUDA stream 0, completely eliminating all 80 synchronizations.

### B. O(1) Unified Memory KV Cache Pinned Registration
* Registered both `ctx.kv_mem` (512 MB) and `ctx.depth_kv_mem` (192 KB) with `cudaHostRegisterMapped | cudaHostRegisterPortable` in [`bmo_init_kv_cache`](file:///home/bmo/jetson_moshi_work/bmo_jetson/src/bmo.cpp).
* Cached device pointers directly on `ctx.k_cache->extra`, `ctx.v_cache->extra`, `ctx.depth_k_cache->extra`, and `ctx.depth_v_cache->extra`, enabling direct GPU access with zero CPU copy.

### C. Elimination of Empty Compute Graph Worker Pool Spin
* **Root Cause:** On Jetson, all forward activations are eagerly evaluated on the GPU. The ggml computation graph `gf` contained only leaf nodes (`GGML_OP_NONE`). Calling `ggml_graph_compute_with_ctx` was waking up a 6-thread worker pool 9 times per iteration to execute zero nodes.
* **Fix:** Added a fast node-scan check in [`bmo_execute_graph`](file:///home/bmo/jetson_moshi_work/bmo_jetson/src/bmo_compute.cpp) using `ggml_graph_n_nodes(gf)` and `ggml_graph_node(gf, i)`. If all nodes are leaves, graph threadpool invocation is bypassed entirely.

### D. Single-Arena Reusable Depth Work Context
* **Root Cause:** `reset_work_ctx(ctx, 256MB)` was being called 8 times per iteration inside the depth codebook loop, incurring repeated `ggml_init`/`ggml_free` overhead.
* **Fix:** Sized depth scratch space to 16 MB allocated once before the 8-step depth loop in [`main.cpp`](file:///home/bmo/jetson_moshi_work/bmo_jetson/src/main.cpp), reclaiming and reusing the arena across all 8 depth steps.

### E. Batched Depth Stream Synchronization
* Bound `bmo_execute_graph` with a conditional `sync_cuda` flag. Steps 0 through 6 queue all depth kernels into stream 0 without stalling the CPU. Step 7 executes the single synchronization barrier for the entire 8-step depth cascade.

---

## 4. 100-Iteration Stress Test Benchmark Results

Full statistics parsed across warm iterations (iterations 2 through 100) from `stress_test.log`:

```
--------------------------------------------------------------------------------
Stress Test Metric                       | Measured Result
--------------------------------------------------------------------------------
Total Iterations                         | 100
Warm Iterations Evaluated                | 99
Minimum Frame Latency                    | 71.0 ms
Maximum Frame Latency                    | 88.0 ms
Mean Frame Latency                       | 77.48 ms
Median Frame Latency                     | 77.0 ms   (< 80.0 ms budget: PASS)
90th Percentile (P90) Latency            | 81.2 ms
95th Percentile (P95) Latency            | 83.0 ms
99th Percentile (P99) Latency            | 88.0 ms
--------------------------------------------------------------------------------
Peak Resident Memory Footprint (VmRSS)   | 5,128 MB  (< 5,500 MB budget: PASS)
Memory Leak / Creep                      | 0 MB (Strictly flat at 5,128 MB)
--------------------------------------------------------------------------------
Temporal Stack Latency (32 Layers)       | 55.0 ms (2.0 ms build + 53.0 ms GPU)
Depth Cascade Latency (8 Codebooks)      | 12.0 ms total (1.5 ms per codebook)
Host Dispatch / Copy Overhead            | ~10.0 ms
--------------------------------------------------------------------------------
```

---

## 5. Geometric and Cascade Verifications

1. **Depth Cascade Validation (`--mode depth_cascade`):**
   * Output: `[SUCCESS] Depth-step 0 validation completed!`
   * Geometry confirmed: 8 depth heads mapped (`audio_heads=8/8`), `dep_ctx=8`, `head_dim=64`, `depth_n_heads=16`, `depth_n_layers=6`.
2. **Temporal Cascade Validation (`--mode temporal_cascade`):**
   * Output: `[SUCCESS] Temporal validation cascade completed!`
   * Confirmed exact 32-layer forward progression without NaN or divergence.
