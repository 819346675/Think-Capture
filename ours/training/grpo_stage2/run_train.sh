#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA_PATH:?Set DATA_PATH to the combined Stage 2 JSONL}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and logs}"
exec torchrun --standalone --nproc_per_node="${NUM_GPUS:-2}" \
  "$SCRIPT_DIR/train.py" \
  --data_path "$DATA_PATH" --output_dir "$OUTPUT_DIR" \
  --model_path "${VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}" \
  --num_generations "${NUM_GENERATIONS:-8}" "$@"
