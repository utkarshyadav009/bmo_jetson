#!/usr/bin/env bash
# ==============================================================================
# start_bmo.sh - Launch BMO Real-Time Full-Duplex Voice Engine
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure Jetson environment safeguards
export NO_TORCH_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Audio endpoint configuration
if [ "$1" == "--bluetooth" ] || [ "$1" == "-bt" ]; then
    python3 "${SCRIPT_DIR}/bmo_audio_routing.py" --mode bluetooth
    shift
elif [ "$1" == "--usb" ]; then
    python3 "${SCRIPT_DIR}/bmo_audio_routing.py" --mode usb
    shift
fi

exec python3 "${SCRIPT_DIR}/test_realtime_stream.py" --use-mic "$@"
