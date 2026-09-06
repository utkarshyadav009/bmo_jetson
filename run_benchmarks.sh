#!/usr/bin/env bash
# ==============================================================================
# run_benchmarks.sh - Run kernel bandwidth and end-to-end latency benchmarks
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
MODEL_GGUF="${1:-${SCRIPT_DIR}/bmo_moshi_8cb_q4.gguf}"

if [ ! -f "${BUILD_DIR}/bmo_kernel_bench" ] || [ ! -f "${BUILD_DIR}/bmo_engine" ]; then
    echo "[INFO] Binaries not found. Running build_jetson.sh first..."
    bash "${SCRIPT_DIR}/build_jetson.sh"
fi

echo "=============================================================================="
echo " STEP 1: Kernel Bandwidth Microbenchmark (vec_dot_q4_0_q8_1 on sm_87)"
echo " Target: Achieved Bandwidth >= 70 GB/s (LPDDR5 memory ceiling: 102 GB/s)"
echo "=============================================================================="
"${BUILD_DIR}/bmo_kernel_bench" --warmup 20 --iters 100

echo ""
echo "=============================================================================="
echo " STEP 2: Depth Cascade Validation (8-Codebook Geometry)"
echo " Model: ${MODEL_GGUF}"
echo "=============================================================================="
if [ ! -f "${MODEL_GGUF}" ]; then
    echo "[WARNING] Model file not found at: ${MODEL_GGUF}"
    echo "Please specify model path: ./run_benchmarks.sh /path/to/bmo_moshi_8cb_q4.gguf"
    exit 1
fi

"${BUILD_DIR}/bmo_engine" "${MODEL_GGUF}" --mode depth_cascade

echo ""
echo "=============================================================================="
echo " STEP 3: Temporal Cascade Validation (32 Layers Standard Q4_0 Gating)"
echo "=============================================================================="
"${BUILD_DIR}/bmo_engine" "${MODEL_GGUF}" --mode temporal_cascade

echo ""
echo "=============================================================================="
echo " STEP 4: End-to-End Frame Latency Stress Test (Target: < 80 ms per frame)"
echo "=============================================================================="
"${BUILD_DIR}/bmo_engine" "${MODEL_GGUF}" --mode stress_test --n-iterations 100

echo ""
echo "=============================================================================="
echo ">>> [BENCHMARK SUITE COMPLETED SUCCESSFULLY] <<<"
echo "=============================================================================="
