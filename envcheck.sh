#!/usr/bin/env bash
# CS3006 Assignment 4 -- environment check. Run from the repository root:
# bash envcheck.sh 2>&1 | tee envcheck.txt
if [ "$(uname -m)" = "x86_64" ]; then SFX=_x86; else SFX=; fi
echo "== OS: $(grep PRETTY_NAME /etc/os-release | cut -d= -f2)"
echo "== glibc: $(ldd --version | head -1)"
echo "== arch: $(uname -m)"
echo "== GPU:"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
--format=csv,noheader || echo "FAIL: no NVIDIA GPU is visible"
nvidia-smi | grep -o 'CUDA Version: [0-9.]*'
echo "== nvcc: $(nvcc --version 2>&1 | grep release)"
echo "== libraries the reference binaries need:"
if ldd scan/cudaScan_ref$SFX render/render_ref$SFX 2>&1 | grep 'not found'
then echo "FAIL: the lines above name what is missing"
else echo "ok: every library found"; fi
echo "== reference smoke tests (both must print a success line):"
(cd scan && ./cudaScan_ref$SFX -m scan -n 1000003 2>&1 | tail -1)
(cd render && ./render_ref$SFX -c -s 256 -f /tmp/envcheck rgb 2>&1 \
| grep -E 'passed|ERROR|rror' | head -3)
