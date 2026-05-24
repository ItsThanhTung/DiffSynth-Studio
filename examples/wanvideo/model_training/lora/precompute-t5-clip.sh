#!/usr/bin/env bash
# Precompute T5 + CLIP embeddings for Wan Animate training (writes precomputed/t5/*.pt, precomputed/clip/*.pt,
# and metadata_precomputed.csv under the dataset folder). Then train with USE_PRECOMPUTED=1 in Wan2.2-Animate-14B-meanflow.sh
#
# Env overrides (optional):
#   WAN_ANIMATE_MODEL_DIR, WAN_ANIMATE_DATA_ROOT (default: $PROJECT_ROOT/ai_dance if present), WAN_ANIMATE_METADATA_IN,
#   WAN_PRECOMPUTE_DEVICE (default cuda), WAN_PRECOMPUTE_OUT_METADATA (default metadata_precomputed.csv),
#   WAN_PRECOMPUTE_CLIP_FP8=1  -> use CLIP fp8 weights (optional; default is full bf16 .pth)
#   WAN_ANIMATE_TOKENIZER_PATH  -> UMT5 tokenizer dir (default: $MODEL_DIR/google/umt5-xxl)
#   WAN_PRECOMPUTE_SHARED_T5=1  -> one shared T5 .pt for all rows; optional WAN_PRECOMPUTE_SHARED_T5_PROMPT
#   WAN_CLIP_FRAME_MODE=first|random  (default first) — CLIP uses first frame vs random frame per row
#   WAN_CLIP_GEOMETRY_MODE=crop_fixed|area_budget  (default crop_fixed). area_budget: CLIP frame resized to
#       pixel area ≤ width*height, aspect preserved (no center-crop to a fixed box).
#   WAN_PRECOMPUTE_NUM_WORKERS  (default 1) — spawn N GPU worker processes (shard by row index %% N)
#   WAN_PRECOMPUTE_DEVICES      e.g. cuda:0,cuda:1,cuda:2,cuda:3 (optional; cycles if fewer than workers)
#   WAN_CLIP_BATCH_SIZE         (default 8) — CLIP images per forward pass
#   WAN_PRECOMPUTE_SKIP_EXISTING=1  — skip rows that already have .pt files
#   WAN_TRAIN_CONDA_ENV  -> conda env name (default meanflow). Example: export WAN_TRAIN_CONDA_ENV=animate
#   (Or: cd DiffSynth-Studio && pip install -e .  in your conda env so `import diffsynth` works without PYTHONPATH.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFSYNTH_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PROJECT_ROOT="$(cd "$DIFFSYNTH_ROOT/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV="${WAN_TRAIN_CONDA_ENV:-animate}"
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
elif [[ -f "$DATA_ROOT/metadata_training.csv" ]]; then
  META_IN="$DATA_ROOT/metadata_training.csv"
else
  META_IN="$DATA_ROOT/metadata.csv"
fi
DEVICE="${WAN_PRECOMPUTE_DEVICE:-cuda}"
OUT_META="${WAN_PRECOMPUTE_OUT_METADATA:-metadata_precomputed.csv}"

HEIGHT="${WAN_ANIMATE_HEIGHT:-720}"
WIDTH="${WAN_ANIMATE_WIDTH:-1280}"
NUM_FRAMES="${WAN_ANIMATE_NUM_FRAMES:-77}"


echo "data root: $DATA_ROOT"
echo "model dir: $MODEL_DIR"
echo "device: $DEVICE"
echo "input metadata: $META_IN"
echo "output metadata: $OUT_META"
echo "height: $HEIGHT"
echo "width: $WIDTH"
echo "num frames: $NUM_FRAMES"
echo "conda env: $CONDA_ENV"

echo "clip frame mode: ${WAN_CLIP_FRAME_MODE:-first}"
echo "clip geometry: ${WAN_CLIP_GEOMETRY_MODE:-crop_fixed}"
echo "num workers: ${WAN_PRECOMPUTE_NUM_WORKERS:-1}"
echo "clip batch size: ${WAN_CLIP_BATCH_SIZE:-8}"
if [[ -n "${WAN_PRECOMPUTE_DEVICES:-}" ]]; then
  echo "devices: $WAN_PRECOMPUTE_DEVICES"
fi

cd "$DIFFSYNTH_ROOT"
export PYTHONPATH="${DIFFSYNTH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_SKIP_DOWNLOAD=true

CLIP_ARGS=()
if [[ "${WAN_PRECOMPUTE_CLIP_FP8:-0}" == "1" ]]; then
  CLIP_ARGS+=(--use_clip_fp8)
fi
CLIP_FRAME_ARGS=()
if [[ -n "${WAN_CLIP_FRAME_MODE:-}" ]]; then
  CLIP_FRAME_ARGS+=(--clip_frame_mode "$WAN_CLIP_FRAME_MODE")
fi
if [[ -n "${WAN_CLIP_RANDOM_SEED:-}" ]]; then
  CLIP_FRAME_ARGS+=(--clip_random_seed "$WAN_CLIP_RANDOM_SEED")
fi
CLIP_GEOM_ARGS=()
if [[ -n "${WAN_CLIP_GEOMETRY_MODE:-}" ]]; then
  CLIP_GEOM_ARGS+=(--clip_geometry_mode "$WAN_CLIP_GEOMETRY_MODE")
fi
SHARED_ARGS=()
if [[ "${WAN_PRECOMPUTE_SHARED_T5:-0}" == "1" ]]; then
  SHARED_ARGS+=(--shared_t5)
  if [[ -n "${WAN_PRECOMPUTE_SHARED_T5_PROMPT:-}" ]]; then
    SHARED_ARGS+=(--shared_t5_prompt "$WAN_PRECOMPUTE_SHARED_T5_PROMPT")
  fi
fi
TOK_ARGS=()
if [[ -n "${WAN_ANIMATE_TOKENIZER_PATH:-}" ]]; then
  TOK_ARGS+=(--tokenizer_path "$WAN_ANIMATE_TOKENIZER_PATH")
fi
WORKER_ARGS=(--num_workers "${WAN_PRECOMPUTE_NUM_WORKERS:-1}")
if [[ -n "${WAN_PRECOMPUTE_DEVICES:-}" ]]; then
  WORKER_ARGS+=(--devices "$WAN_PRECOMPUTE_DEVICES")
fi
BATCH_ARGS=(--clip_batch_size "${WAN_CLIP_BATCH_SIZE:-8}")
SKIP_ARGS=()
if [[ "${WAN_PRECOMPUTE_SKIP_EXISTING:-0}" == "1" ]]; then
  SKIP_ARGS+=(--skip_existing)
fi
DIR_ARGS=()
if [[ -n "${WAN_PRECOMPUTED_DIR:-}" ]]; then
  DIR_ARGS+=(--precomputed_dir "$WAN_PRECOMPUTED_DIR")
fi

exec python examples/wanvideo/model_training/precompute_t5_clip_embeddings.py \
  --dataset_base_path "$DATA_ROOT" \
  --dataset_metadata_path "$META_IN" \
  --model_dir "$MODEL_DIR" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --device "$DEVICE" \
  --out_metadata_name "$OUT_META" \
  "${TOK_ARGS[@]}" \
  "${SHARED_ARGS[@]}" \
  "${CLIP_ARGS[@]}" \
  "${CLIP_FRAME_ARGS[@]}" \
  "${CLIP_GEOM_ARGS[@]}" \
  "${WORKER_ARGS[@]}" \
  "${BATCH_ARGS[@]}" \
  "${SKIP_ARGS[@]}" \
  "${DIR_ARGS[@]}"
