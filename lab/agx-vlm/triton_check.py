"""Check that Triton can build and run a kernel on the AGX Orin.

vLLM forces its TRITON_ATTN backend for Gemma 4 (head_dim 256 and 512 mixed), so Triton must work there.
The Jetson triton 3.6 wheel ships neither cuda.h nor an sm_87 ptxas, so serve.sh points CPATH and
TRITON_PTXAS_PATH at the JetPack CUDA 12.6 install. Run with the same environment:

  CPATH=/usr/local/cuda/include TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    LD_LIBRARY_PATH=$PWD/cudss-lib .venv/bin/python triton_check.py
"""
import torch
import triton
import triton.language as tl
from triton.backends.nvidia.driver import CudaUtils

CudaUtils()
print("triton cuda_utils OK")


@triton.jit
def add(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)


x = torch.randn(4096, device="cuda")
y = torch.randn(4096, device="cuda")
o = torch.empty_like(x)
add[(4,)](x, y, o, 4096, B=1024)
torch.cuda.synchronize()
print("triton kernel OK", float((o - x - y).abs().max()))
