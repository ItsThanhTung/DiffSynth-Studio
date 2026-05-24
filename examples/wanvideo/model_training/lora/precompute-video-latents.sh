#!/usr/bin/env bash
# Precompute Wan Animate VAE tensors (writes precomputed/latents/*.pt and metadata_precomputed_latents.csv).
# Run after T5/CLIP precompute if you use both: point metadata at metadata_precomputed.csv so video paths stay available.
#
# Env overrides (optional): WAN_ANIMATE_MODEL_DIR, WAN_ANIMATE_DATA_ROOT (default: $PROJECT_ROOT/ai_dance if present), WAN_ANIMATE_METADATA_IN,
# WAN_ANIMATE_NUM_FRAMES, WAN_PRECOMPUTE_DEVICE, WAN_PRECOMPUTE_OUT_METADATA_LATENTS (default metadata_precomputed_latents.csv),
# WAN_TRAIN_CONDA_ENV (default meanflow), e.g. export WAN_TRAIN_CONDA_ENV=animate
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFSYNTH_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PROJECT_ROOT="$(cd "$DIFFSYNTH_ROOT/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV="${WAN_TRAIN_CONDA_ENV:-meanflow}"
conda activate "$CONDA_ENV"

MODEL_DIR="${WAN_ANIMATE_MODEL_DIR:-/home/nmhung/training/Wan2.2-Animate-14B}"

if [[ -n "${WAN_ANIMATE_DATA_ROOT:-}" ]]; then
  DATA_ROOT="${WAN_ANIMATE_DATA_ROOT}"
elif [[ -d "$PROJECT_ROOT/ai_dance" ]]; then
  DATA_ROOT="$PROJECT_ROOT/ai_dance"
else
  DATA_ROOT="$PROJECT_ROOT/data/diffsynth_example_dataset/wanvideo/Wan2.2-Animate-14B"
fi

if [[ -n "${WAN_ANIMATE_METADATA_IN:-}" ]]; then
  META_IN="${WAN_ANIMATE_METADATA_IN}"
else
  META_IN="$DATA_ROOT/metadata_precomputed.csv"
fi
DEVICE="${WAN_PRECOMPUTE_DEVICE:-cuda}"
OUT_META="${WAN_PRECOMPUTE_OUT_METADATA_LATENTS:-metadata_precomputed_latents.csv}"

HEIGHT="${WAN_ANIMATE_HEIGHT:-480}"
WIDTH="${WAN_ANIMATE_WIDTH:-832}"
NUM_FRAMES="${WAN_ANIMATE_NUM_FRAMES:-33}"

echo "data root: $DATA_ROOT"
echo "model dir: $MODEL_DIR"
echo "device: $DEVICE"
echo "input metadata: $META_IN"
echo "output metadata: $OUT_META"

cd "$DIFFSYNTH_ROOT"
export PYTHONPATH="${DIFFSYNTH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_SKIP_DOWNLOAD=true

exec python examples/wanvideo/model_training/precompute_wan_animate_latents.py \
  --dataset_base_path "$DATA_ROOT" \
  --dataset_metadata_path "$META_IN" \
  --model_dir "$MODEL_DIR" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --device "$DEVICE" \
  --out_metadata_name "$OUT_META"
