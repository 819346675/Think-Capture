#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
BENCHMARK="${1:?Usage: bash ours/evaluation/run_eval.sh mindcube|mmsi|viewspatial [options]}"
shift
case "$BENCHMARK" in
  mindcube) EVAL_SCRIPT=eval_mindcube.py ;;
  mmsi) EVAL_SCRIPT=eval_mmsi_bench.py ;;
  viewspatial) EVAL_SCRIPT=ours/evaluation/eval_viewspatial.py ;;
  *) echo "Unknown benchmark: $BENCHMARK" >&2; exit 2 ;;
esac
exec python "$EVAL_SCRIPT" --num_samples 0 "$@"
