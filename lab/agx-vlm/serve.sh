#!/usr/bin/env bash
# Gemma 4 26B-A4B on vLLM, OpenAI API at 127.0.0.1:8100 on the AGX Orin (agx@192.168.0.72).
# Takes over the Spark container gemma4-26b-a4b (2026-11-26): same served name, port, parsers and
# structured-output backend, so the console only needs its 8100 tunnel pointed here.
# Differences from the Spark, all because of the 64 GB Orin (Ampere sm_87, memory shared with the CPU):
#   - weights: cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4 (Google's QAT checkpoint as compressed-tensors W4A16,
#     17.2 GB, Marlin kernels); the Spark's NVFP4 needs Blackwell.
#   - --max-num-seqs 8 instead of 64. The KV cache is fixed at 10 GiB like the Spark's --kv-cache-memory
#     (about 43k tokens, more than one 32k request; the console sends a few thousand tokens with two images).
#     With --gpu-memory-utilization 0.6 alone vLLM took a 23.9 GiB KV cache and the engine held 46.9 GiB of the
#     61.4 GiB the OS sees, leaving 11.6 GiB. 0.6 stays as the start-up free-memory check. The old Qwen3-VL-8B transformers server
#     (~/vlm_tape_20261118/qwen_server.py, 19 GB, port 8000) was stopped on 2026-11-26 to make room.
#   - --max-num-batched-tokens 8192: one Gemma 4 image can be up to 2496 tokens and multimodal input is not
#     chunked, so the 2048 default refuses to start.
#   - the model's own chat template with enable_thinking off by default (the Spark's no-thinking template file
#     is not readable from hkim); the console also sends enable_thinking: false itself.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export LD_LIBRARY_PATH="$PWD/cudss-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   # torch 2.11 needs libcudss.so.0 (install.sh)
# vLLM forces TRITON_ATTN for Gemma 4; the Jetson triton 3.6 wheel ships no cuda.h and only a Blackwell ptxas
# (triton_check.py), so both come from JetPack's CUDA 12.6.
export CPATH="/usr/local/cuda/include${CPATH:+:$CPATH}"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
MODEL="${MODEL:-$HOME/models/gemma-4-26B-A4B-it-qat-AWQ-INT4}"
exec vllm serve "$MODEL" \
  --served-model-name gemma-4-26b-a4b-it \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8100}" \
  --trust-remote-code \
  --max-model-len "${MAX_LEN:-32768}" --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "${GPU_UTIL:-0.6}" \
  --kv-cache-memory-bytes "${KV_BYTES:-10737418240}" \
  --limit-mm-per-prompt '{"image": 4}' \
  --tool-call-parser gemma4 --reasoning-parser gemma4 --enable-auto-tool-choice \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --default-chat-template-kwargs '{"enable_thinking": false}'
