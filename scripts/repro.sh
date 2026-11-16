#!/usr/bin/env bash
# 三组对照（shared / short / random），每组一个子进程，避免显存没释放导致 OOM。
#
# 必须在 learn-nano-vllm 仓库根目录执行（脚本要 import nanovllm）：
#   git clone https://github.com/CZYCW/learn-nano-vllm.git && cd learn-nano-vllm
#   pip install -e .
#   export NANOVLLM_MODEL=~/huggingface/Qwen3-0.6B
#   bash /path/to/nano-vllm-lab/scripts/repro.sh
#
# 默认写到 results/rerun-<时间>.jsonl，不覆盖本仓已存的原始数据。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$HERE/experiments/29_prefix_cache_ablation.py"
OUT="${OUT:-$HERE/results}"
STAMP="$(date +%Y%m%d-%H%M%S)"

: "${NANOVLLM_MODEL:?请先 export NANOVLLM_MODEL=/path/to/Qwen3-0.6B}"

mkdir -p "$OUT"
python "$SCRIPT" --arm all --jsonl "$OUT/rerun-$STAMP.jsonl" "$@" | tee "$OUT/rerun-$STAMP.log"

echo "done -> $OUT/rerun-$STAMP.jsonl"