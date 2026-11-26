#!/usr/bin/env bash
# Install vLLM on the AGX Orin (agx@192.168.0.72) into ~/gemma4-vllm/.venv.
# JetPack 6.2 = CUDA 12.6, Python 3.10, no docker and no sudo there, so: uv venv + the jetson-ai-lab wheel index
# (torch / vllm / triton / flashinfer built for sm_87) first, PyPI for everything else.
# The index order matters: uv puts --extra-index-url AHEAD of --index-url, which on 2026-11-26 pulled PyPI's
# CUDA 13 sbsa torch/vllm (+ nvidia-*-cu13 libs) that cannot run on JetPack 6. --index (first) +
# --default-index (PyPI, last) keeps every package the Jetson index has on the Jetson build.
# transformers is pinned to 5.10.2 (the version the checkpoint was saved with): gemma4 needs >=5.5 (uv picks 4.57
# otherwise), and 5.20 makes config.head_dim a per-layer attribute that vLLM 0.20 cannot read
# (AmbiguousGlobalPerLayerAttributeError at start-up).
# The Jetson vllm wheel declares fewer requirements than PyPI's vllm 0.20.0, so the missing ones are named here:
# xgrammar (structured outputs; Jetson index 0.1.32), torchvision 0.26.0 (Gemma4Processor), compressed-tensors
# 0.15.0.1 (the W4A16 checkpoint format) and opencv-python-headless. torchaudio 2.11 is not on the Jetson index
# and Gemma 4 26B has no audio, so it is left out. The Jetson torch 2.11 links libcudss.so.0, which JetPack 6.2 does not ship and apt needs
# sudo for, so the cuDSS 0.7.1 Jetson (linux-aarch64, CUDA 12) archive is unpacked next to the venv; serve.sh
# puts it on LD_LIBRARY_PATH.
#
#   scp install.sh serve.sh agx@192.168.0.72:~/gemma4-vllm/ && ssh agx@192.168.0.72 '~/gemma4-vllm/install.sh'
set -euo pipefail
cd "$(dirname "$0")"
command -v ~/.local/bin/uv >/dev/null || pip3 install --user uv   # the system python has no ensurepip
[ -d .venv ] || ~/.local/bin/uv venv --python /usr/bin/python3.10 .venv
source .venv/bin/activate
~/.local/bin/uv pip install "vllm==0.20.0+cu126" "triton==3.6.0" "transformers==5.10.2" "torchvision==0.26.0" xgrammar \
  "compressed-tensors==0.15.0.1" "opencv-python-headless>=4.13.0" \
  --index https://pypi.jetson-ai-lab.io/jp6/cu126 --default-index https://pypi.org/simple
if [ ! -e cudss-lib/libcudss.so.0 ]; then
  python - <<'PY'
import hashlib, json, os, tarfile, urllib.request
base = "https://developer.download.nvidia.com/compute/cudss/redist/"
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "python"}), timeout=60).read()
ent = json.loads(get(base + "redistrib_0.7.1.json"))["libcudss"]["linux-aarch64"]["cuda12"]
data = get(base + ent["relative_path"])
assert hashlib.sha256(data).hexdigest() == ent["sha256"], "cuDSS sha256 mismatch"
open("cudss.tar.xz", "wb").write(data)
tarfile.open("cudss.tar.xz").extractall("cudss")
os.remove("cudss.tar.xz")
PY
  ln -sfn "$(dirname "$(find cudss -name libcudss.so.0 | head -1)")" cudss-lib
fi
export LD_LIBRARY_PATH="$PWD/cudss-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import torch, vllm, transformers; print('OK torch', torch.__version__, torch.cuda.is_available(), \
torch.cuda.get_device_name(0), 'vllm', vllm.__version__, 'transformers', transformers.__version__)"
