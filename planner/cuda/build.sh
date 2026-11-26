#!/usr/bin/env bash
# Build planner/cuda/libikchain.so for this machine's GPU (AGX Orin: sm_87). Needs nvcc (JetPack: /usr/local/cuda).
#   planner/cuda/build.sh            # arch from nvidia-smi / tegra (default sm_87)
#   ARCH=sm_120 planner/cuda/build.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
ARCH="${ARCH:-sm_87}"
# FP32 by default (the Orin's GPU is far slower in FP64); FP64=1 builds the double-precision variant.
FLAGS=()
[ "${FP64:-0}" = 1 ] && FLAGS+=(-DIK_FP64)
"$NVCC" -O3 -arch="$ARCH" "${FLAGS[@]}" -shared -Xcompiler -fPIC -o "$HERE/${OUT:-libikchain.so}" "$HERE/ik_chain.cu"
echo "built $HERE/${OUT:-libikchain.so} ($ARCH${FLAGS:+, FP64})"
