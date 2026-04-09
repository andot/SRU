#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then PY=python; fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [backend] [-m model] [-s size] [-o out] [--fp16] [-q] [-v]

backend: one of all|onnx|coreml|ncnn (default: all)
If -m/--model is omitted, script scans models/*.pth and models/*/*.pth and exports all discovered models.
EOF
  exit 1
}

# detect backend if provided as first positional
BACKEND=${1:-all}
if [[ "$BACKEND" =~ ^(all|onnx|coreml|ncnn)$ ]]; then
  shift || true
else
  BACKEND=all
fi

# parse options
MODEL=""
SIZE=""
OUT=""
FP16=""
QUANTIZE=""
VALIDATE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      MODEL="$2"; shift 2;;
    -s|--size)
      SIZE="$2"; shift 2;;
    -o|--output)
      OUT="$2"; shift 2;;
    --fp16)
      FP16="--fp16"; shift;;
    -q|--quantize-8bit)
      QUANTIZE="--quantize-8bit"; shift;;
    -v|--validate)
      VALIDATE="--validate"; shift;;
    -h|--help)
      usage;;
    *)
      echo "Unknown argument: $1"; usage;;
  esac
done

EXTRA_ARGS=""
[[ -n "$OUT" ]] && EXTRA_ARGS="$EXTRA_ARGS -o $OUT"
[[ -n "$SIZE" ]] && EXTRA_ARGS="$EXTRA_ARGS -s $SIZE"
EXTRA_ARGS="$EXTRA_ARGS $FP16 $QUANTIZE $VALIDATE"

# gather model list
if [[ -n "$MODEL" ]]; then
  MODELS=("$MODEL")
else
  IFS=$'\n' read -r -d '' -a MODELS < <(find "$REPO_ROOT/models" -type f -name "*.pth" -print 2>/dev/null | sed -E 's#.*/([^/]+)\.pth$#\1#' | sort -u && printf '\0')
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "[Warn] No models found to export (use -m to specify a model)."; exit 0
fi

for m in "${MODELS[@]}"; do
  echo "[Info] Exporting model: $m -> backend: $BACKEND"
  # call exporter; keep going on error
  "$PY" "$SCRIPT_DIR/export.py" "$BACKEND" -m "$m" $EXTRA_ARGS || echo "[Warn] export failed for $m (continuing)"
done
