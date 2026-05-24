#!/usr/bin/env bash
# Run Wan T5+CLIP precompute for tiktok-videos_77 metadata (see DiffSynth-Studio/.../precompute-t5-clip.sh).
#
# Usage:
#   ./precompute_t5_clip_tiktok77.sh first
#   ./precompute_t5_clip_tiktok77.sh random [seed]
#
# Env overrides: WAN_ANIMATE_MODEL_DIR, WAN_TRAIN_CONDA_ENV, WAN_PRECOMPUTE_DEVICE,
#   WAN_CLIP_GEOMETRY_MODE=crop_fixed|area_budget  (default crop_fixed). area_budget: CLIP frame resized to
#       pixel area ≤ width*height, aspect preserved (no center-crop to a fixed box).
#   WAN_ANIMATE_HEIGHT / WAN_ANIMATE_WIDTH (defaults 720 / 1280)
#   WAN_PRECOMPUTE_SHARED_T5=1 (default in this script): one T5 .pt for all rows; WAN_PRECOMPUTE_SHARED_T5_PROMPT overrides text
#   WAN_PRECOMPUTE_NUM_WORKERS, WAN_PRECOMPUTE_DEVICES, WAN_CLIP_BATCH_SIZE (see precompute-t5-clip.sh)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-first}"
SEED="${2:-42}"
LORA_SH="$ROOT/DiffSynth-Studio/examples/wanvideo/model_training/lora/precompute-t5-clip.sh"

export WAN_ANIMATE_DATA_ROOT="${WAN_ANIMATE_DATA_ROOT:-$ROOT/wan_animate_data/tiktok-videos_77}"
export WAN_ANIMATE_METADATA_IN="${WAN_ANIMATE_METADATA_IN:-$ROOT/wan_animate_data/tiktok-videos_77/metadata_training.csv}"
export WAN_ANIMATE_MODEL_DIR="${WAN_ANIMATE_MODEL_DIR:-$ROOT/Wan2.2-Animate-14B}"
export WAN_ANIMATE_HEIGHT="${WAN_ANIMATE_HEIGHT:-720}"
export WAN_ANIMATE_WIDTH="${WAN_ANIMATE_WIDTH:-1280}"
# Encode T5 once; reuse for every row (same prompt as training CSV).
export WAN_PRECOMPUTE_SHARED_T5="${WAN_PRECOMPUTE_SHARED_T5:-1}"
export WAN_PRECOMPUTE_SHARED_T5_PROMPT="${WAN_PRECOMPUTE_SHARED_T5_PROMPT:-视频中的人在做动作}"

if [[ "$MODE" == "first" ]]; then
  export WAN_CLIP_FRAME_MODE="first"
  export WAN_PRECOMPUTE_OUT_METADATA="${WAN_PRECOMPUTE_OUT_METADATA:-metadata_precomputed_clip_first.csv}"
elif [[ "$MODE" == "random" ]]; then
  export WAN_CLIP_FRAME_MODE="random"
  export WAN_CLIP_RANDOM_SEED="${WAN_CLIP_RANDOM_SEED:-$SEED}"
  export WAN_PRECOMPUTE_OUT_METADATA="${WAN_PRECOMPUTE_OUT_METADATA:-metadata_precomputed_clip_random.csv}"
else
  echo "usage: $0 first | random [seed]" >&2
  exit 1
fi

exec bash "$LORA_SH"
