#!/usr/bin/env bash
# Wan2.2-Animate-14B LoRA training (4× GPU, precomputed T5+CLIP).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEANFLOW_SH="${ROOT}/DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.2-Animate-14B-meanflow.sh"

# --- Data & models ---
export USE_PRECOMPUTED=1
export WAN_TRAIN_CONDA_ENV="${ROOT}/env_dir/animate"
export WAN_ANIMATE_MODEL_DIR="${ROOT}/Wan2.2-Animate-14B"
export WAN_ANIMATE_DATA_ROOT="${ROOT}/wan_animate_data/tiktok-videos_77"
export WAN_ANIMATE_METADATA="${ROOT}/wan_animate_data/tiktok-videos_77/metadata_precomputed_clip_random.csv"

# --- Checkpoints every 1000 optimizer steps → step-1000.safetensors, step-2000.safetensors, … ---
export WAN_ANIMATE_OUTPUT_PATH="${WAN_ANIMATE_OUTPUT_PATH:-${ROOT}/checkpoint_random_frame}"
export WAN_SAVE_STEPS="${WAN_SAVE_STEPS:-1000}"
mkdir -p "$WAN_ANIMATE_OUTPUT_PATH"

# --- Training 1280×720 (matches extract_embedding.sh) ---
export WAN_ANIMATE_NUM_FRAMES="${WAN_ANIMATE_NUM_FRAMES:-77}"
export WAN_ANIMATE_NUM_EPOCHS="${WAN_ANIMATE_NUM_EPOCHS:-5}"
export WAN_ANIMATE_HEIGHT="${WAN_ANIMATE_HEIGHT:-480}"
export WAN_ANIMATE_WIDTH="${WAN_ANIMATE_WIDTH:-832}"
export WAN_ANIMATE_MAX_PIXELS="${WAN_ANIMATE_MAX_PIXELS:-$((WAN_ANIMATE_HEIGHT * WAN_ANIMATE_WIDTH))}"

# --- 4× GPU DDP ---
export WAN_TRAIN_NUM_GPUS="${WAN_TRAIN_NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export WAN_NCCL_MODE="${WAN_NCCL_MODE:-local}"

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Debug only when needed: WAN_CUDA_DEBUG=1 bash train.sh
if [[ "${WAN_CUDA_DEBUG:-0}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
  export NCCL_DEBUG=INFO
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
fi

exec bash "$MEANFLOW_SH"
