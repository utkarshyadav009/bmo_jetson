#!/usr/bin/env bash
# ==============================================================================
# build_jetson.sh - Automated build script for Jetson Orin Nano (sm_87)
# ==============================================================================
set -e

echo "=== [1/4] Preparing Jetson Orin Nano Build Environment ==="
if command -v jetson_clocks &> /dev/null; then
    echo "Locking Jetson clocks to maximum frequency for reproducible benchmarks..."
    sudo jetson_clocks || true
else
    echo "[INFO] jetson_clocks not found or not running with sudo; continuing..."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== [2/4] Configuring with CMake for sm_87 ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DBMO_TARGET_JETSON=ON \
    -DBMO_ENABLE_CUDA=ON \
    -DBMO_USE_FETCHCONTENT=ON

echo "=== [3/4] Compiling bmo_engine, bmo_shared, and bmo_kernel_bench ==="
NPROC=$(nproc || echo 4)
make -j"${NPROC}"

echo "=== [4/4] Verifying Build Artifacts ==="
for bin in bmo_engine bmo_kernel_bench libbmo.so; do
    if [ -f "${bin}" ]; then
        echo "  [OK] Built ${BUILD_DIR}/${bin}"
    else
        echo "  [ERROR] Missing expected artifact: ${bin}"
        exit 1
    fi
done

echo ""
echo "=============================================================================="
echo ">>> [SUCCESS] Jetson Orin Nano compilation completed!"
echo ">>> Executables ready in: ${BUILD_DIR}"
echo "    - bmo_kernel_bench : Microbenchmark for Q4_0 GEMV bandwidth (>= 70 GB/s)"
echo "    - bmo_engine       : Full forward pass and frame latency benchmark"
echo "    - libbmo.so        : C-ABI shared library for Python runtime"
echo "=============================================================================="
