# BMO Standard Moshi 8-Codebook Jetson Orin Nano Deployment Package

This package contains the updated C++ inference engine (`bmo_engine`, `libbmo.so`, `bmo_kernel_bench`) specifically optimized for the **Jetson Orin Nano 8GB (Ampere sm_87)** running standard 8-codebook Kyutai Moshi (`dep_q = 8`, `num_codebooks = 8`).

---

## 1. Key Architectural Upgrades from PersonaPlex

1. **Standard 8-Codebook Depth Geometry (`dep_q = 8`):**
   - Reduced depformer depth steps per frame from 16 to 8 (50% reduction in depth autoregressive steps).
   - Depformer KV cache explicitly sized for 8 steps (`depth_n_ctx = 8`), reducing cache memory and per-frame depth latency by half.
2. **Clean Standard GGUF with Zero Sidecars:**
   - Completely retired custom `BMO_TIER` multi-tier sidecars (`.packed_weights`, `.tile_tiers`, `.tier_offsets`, `.outlier_*`).
   - Gating layers (`gating_linear_in_weight`, `gating_linear_out_weight`, and step-specific depth gating) load as standard `ggml_tensor` of type `GGML_TYPE_Q4_0`.
3. **High-Throughput Native Tensor Core GEMV Dispatch:**
   - Standard `ggml_mul_mat` directly triggers `ggml-cuda`'s vectorized `vec_dot_q4_0_q8_1` kernel with 128-bit loads and `dp4a` INT8/INT4 math.
   - Restores peak LPDDR5 bus saturation from ~37 GB/s (legacy CSR branch divergence) to **>= 70 GB/s** (memory bandwidth ceiling ~102 GB/s).
4. **CUDA Graph Compatible:**
   - Because all ops reside in the standard `ggml_cgraph` without transient CPU/GPU staging unpacks, the forward pass can be captured into a single hardware CUDA graph to eliminate kernel launch bubbles and break below the **80 ms frame budget** (12.5 fps real-time audio).

---

## 2. Directory Structure

```
jetson_deployment/
├── CMakeLists.txt          # Jetson sm_87 CMake build configuration
├── build_jetson.sh         # Turnkey 1-click build script
├── run_benchmarks.sh       # Comprehensive automated benchmark harness
├── bmo_config.json         # 8-codebook model configuration
├── bmo_engine.py           # Zero-copy Python ctypes C-ABI binding
├── bmo_trt_mimi.py         # TensorRT FP16 SEANet + fast cuBLAS RVQ codec
├── export_trt_mimi.py      # ONNX export & TensorRT FP16 engine compiler
├── test_offline_pipeline.py# End-to-end offline verification pipeline
├── test_realtime_stream.py # Full-duplex interactive microphone stream
├── bmo_moshi_8cb_q4.gguf   # 4.05 GiB clean hardware-aligned GGUF model
└── src/
    ├── bmo.h               # Core data structures (DEP_Q=8, NUM_CODEBOOKS=8)
    ├── bmo.cpp             # Model loader and KV cache allocator
    ├── bmo_compute.cpp     # Graph builder with direct ggml_mul_mat bindings
    ├── bmo_api.h           # C-ABI header
    ├── bmo_api.cpp         # C-ABI implementation (CUDA Graph capture)
    ├── main.cpp            # bmo_engine CLI and benchmark harness
    ├── bmo_kernel_bench.cu # Standalone Q4_0 GEMV bandwidth microbenchmark
    ├── bmo_cuda_kernels.cu # Hardware acceleration routines
    └── bmo_cuda_kernels_proto.cu
```

---

## 3. TensorRT Mimi Audio Codec Acceleration

To eliminate the ~26.6 ms PyTorch SEANet codec bottleneck, Mimi is accelerated via TensorRT FP16 and cuBLAS:
1. **SEANet Encoder Engine:** 0.71 ms median GPU compute (down from 3.83 ms in PyTorch).
2. **SEANet Decoder Engine:** 0.74 ms median GPU compute (down from 3.43 ms in PyTorch).
3. **Fast GEMM Quantizer:** Replaces iterative `torch.cdist` with $x \cdot e_k - \frac{1}{2}\|e_k\|^2$ via cuBLAS (1.3 ms).
4. **Overall Mimi Round-Trip:** **10.79 ms** (down from 26.6 ms).

Compile TensorRT engines:
```bash
python3 export_trt_mimi.py
```

---

## 4. Quick Start & Verification on Jetson Orin Nano

### Build C++ Engine & Library:
```bash
bash build_jetson.sh
```

### 1. Benchmark libbmo Engine:
```bash
# Verify memory bandwidth (Target: >= 70 GB/s)
./build/bmo_kernel_bench --warmup 20 --iters 100

# Benchmark end-to-end frame latency (100 iterations, Target: < 80 ms)
./build/bmo_engine bmo_moshi_8cb_q4.gguf --mode stress_test --n-iterations 100
```

### 2. Run Offline End-to-End Pipeline (Mimi + libbmo):
```bash
python3 test_offline_pipeline.py
```
- Median Frame Latency: **78.8 ms** (RTF: 0.984x, faster than real-time 80 ms budget).
- Stable VmRSS: ~6.15 GB (drift < 2 MB).

### 3. Run Live Interactive Duplex Voice Stream:
```bash
# Live microphone & speaker stream (speak naturally to BMO):
python3 test_realtime_stream.py --device cuda --use-mic --duration 30

# Or run automated simulated audio duplex test:
python3 test_realtime_stream.py --device cuda --duration 15
```
