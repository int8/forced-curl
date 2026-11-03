#!/usr/bin/env bash
# 100k-sample destination probe, native tool-call format, no lead, all four models back to back.
# 3 tasks x 3 prefills x 11112 = 100,008 samples per model. Nothing is executed; no network.
cd "$(dirname "$0")"
echo $$ > results/run100k_native.pid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 DEST_CONTEXT=toolcall DEST_FORMAT=native DEST_LEAD=none DEST_N=11112
export DEST_OUT=results/dest100k_native.jsonl DEST_RAW=results/dest100k_native.jsonl.raw.jsonl
run() {  # model batch quant
  echo "===== $1 batch=$2 start $(date '+%F %T') ====="
  MODEL_ID=$1 DEST_BATCH=$2 QUANT=$3 \
    unshare -r -n .venv/bin/python src/dest_bias.py 2>&1 | grep -v -i "warn\|torch_dtype\|falling back"
  echo "===== $1 end $(date '+%F %T') ====="
}
run zai-org/GLM-4-9B-0414 32 ""
run Qwen/Qwen3.5-9B 22 ""
run google/gemma-4-12B-it-qat-q4_0-unquantized 48 4bit
run google/gemma-4-E4B-it 48 ""
echo "ALLDONE $(date '+%F %T')"
rm -f results/run100k_native.pid
