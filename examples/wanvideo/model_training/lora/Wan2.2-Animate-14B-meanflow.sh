#!/usr/bin/env bash
# Wan2.2-Animate-14B LoRA training (conda env WAN_TRAIN_CONDA_ENV, default meanflow), local weights, and DiffSynth example data (ModelScope: diffsynth_example_dataset/wanvideo/Wan2.2-Animate-14B).
# Multi-GPU: Accelerate DDP on 4 GPUs (accelerate_config_multi_gpu.yaml). Set WAN_TRAIN_NUM_GPUS=1 for single-GPU only.
export WAN_TRAIN_NUM_GPUS="${WAN_TRAIN_NUM_GPUS:-4}"
#
# T5: prompt -> token embeddings (context). CLIP: first frame -> clip_feature. Both are fused inside DiT (text_embedding + img_emb).
#
# VRAM: set USE_PRECOMPUTED=1 after running precompute (saves T5+CLIP VRAM during training; still loads DiT+VAE+adapter):
#   bash examples/wanvideo/model_training/lora/precompute-t5-clip.sh
#
# Stronger: USE_VIDEO_LATENTS=1 after precompute-t5-clip.sh and precompute-video-latents.sh (no VAE at train time; DiT+adapter only):
#   bash examples/wanvideo/model_training/lora/precompute-video-latents.sh
#
# Optional env: WAN_ANIMATE_NUM_FRAMES (e.g. 81), WAN_ANIMATE_NUM_EPOCHS (default 5).
# Resolution (default: area budget, aspect preserved):
#   WAN_ANIMATE_MAX_PIXELS — cap per frame. Default 480*832 (official Wan Animate recipe; fits ~95GB/GPU with 77f + VAE).
#     720*1280 often OOMs per GPU even with USE_PRECOMPUTED=1. For 720p training use USE_VIDEO_LATENTS=1 (precompute latents).
#   WAN_ANIMATE_FIXED_SIZE=1 — legacy center-crop to exact height x width.
#   WAN_TRAIN_NUM_GPUS — number of GPUs (default 4). DDP replicates the full model on each GPU (does not shard VRAM).
# Multi-GPU DDP: set WAN_FIND_UNUSED_PARAMETERS=1 only if training errors on unused params.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# DiffSynth-Studio repo root (examples/wanvideo/model_training/lora -> ../../../..)
DIFFSYNTH_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
# Project root containing ./data (one level above DiffSynth-Studio)
PROJECT_ROOT="$(cd "$DIFFSYNTH_ROOT/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV="${WAN_TRAIN_CONDA_ENV:-meanflow}"
conda activate "$CONDA_ENV"

MODEL_DIR="${WAN_ANIMATE_MODEL_DIR:-/home/nmhung/training/Wan2.2-Animate-14B}"
# Default: $PROJECT_ROOT/ai_dance if present, else ModelScope example path.
if [[ -n "${WAN_ANIMATE_DATA_ROOT:-}" ]]; then
  DATA_ROOT="${WAN_ANIMATE_DATA_ROOT}"
elif [[ -d "$PROJECT_ROOT/ai_dance" ]]; then
  DATA_ROOT="$PROJECT_ROOT/ai_dance"
else
  DATA_ROOT="$PROJECT_ROOT/data/diffsynth_example_dataset/wanvideo/Wan2.2-Animate-14B"
fi

cd "$DIFFSYNTH_ROOT"
export PYTHONPATH="${DIFFSYNTH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Use local checkpoint files only (no ModelScope/HF download for listed weights).
export DIFFSYNTH_SKIP_DOWNLOAD=true

USE_PRECOMPUTED="${USE_PRECOMPUTED:-0}"
USE_VIDEO_LATENTS="${USE_VIDEO_LATENTS:-0}"

if [[ "$USE_VIDEO_LATENTS" == "1" ]]; then
  META_PATH="${WAN_ANIMATE_METADATA:-$DATA_ROOT/metadata_precomputed_latents.csv}"
  MODEL_SPECS="${MODEL_DIR}:diffusion_pytorch_model*.safetensors"
  # Do not list `prompt` in data_file_keys: UnifiedDataset would treat it as a media path (breaks on Chinese text).
  DATA_KEYS="wan_latent_cache,t5_context,clip_feature"
  EXTRA_INS="t5_context,clip_feature"
  PRECOMP_FLAG=(--precomputed_t5_clip --precomputed_video_latents)
  FP8_FLAG=()
elif [[ "$USE_PRECOMPUTED" == "1" ]]; then
  META_PATH="${WAN_ANIMATE_METADATA:-$DATA_ROOT/metadata_precomputed.csv}"
  MODEL_SPECS="${MODEL_DIR}:diffusion_pytorch_model*.safetensors,${MODEL_DIR}:Wan2.1_VAE.pth"
  DATA_KEYS="video,animate_pose_video,animate_face_video,t5_context,clip_feature"
  EXTRA_INS="input_image,animate_pose_video,animate_face_video,t5_context,clip_feature"
  PRECOMP_FLAG=(--precomputed_t5_clip)
  FP8_FLAG=()
else
  if [[ -n "${WAN_ANIMATE_METADATA:-}" ]]; then
    META_PATH="${WAN_ANIMATE_METADATA}"
  elif [[ -f "$DATA_ROOT/metadata_training.csv" ]]; then
    META_PATH="$DATA_ROOT/metadata_training.csv"
  else
    META_PATH="$DATA_ROOT/metadata.csv"
  fi
  MODEL_SPECS="${MODEL_DIR}:diffusion_pytorch_model*.safetensors,${MODEL_DIR}:models_t5_umt5-xxl-enc-bf16.pth,${MODEL_DIR}:Wan2.1_VAE.pth,${MODEL_DIR}:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  DATA_KEYS="video,animate_pose_video,animate_face_video"
  EXTRA_INS="input_image,animate_pose_video,animate_face_video"
  PRECOMP_FLAG=()
  FP8_FLAG=()
fi

# Training pixel budget (defaults match official Wan2.2-Animate-14B.sh: 480x832).
WAN_ANIMATE_HEIGHT="${WAN_ANIMATE_HEIGHT:-480}"
WAN_ANIMATE_WIDTH="${WAN_ANIMATE_WIDTH:-832}"
WAN_ANIMATE_MAX_PIXELS="${WAN_ANIMATE_MAX_PIXELS:-$((WAN_ANIMATE_HEIGHT * WAN_ANIMATE_WIDTH))}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESIZE_ARGS=(--max_pixels "$WAN_ANIMATE_MAX_PIXELS")
if [[ "${WAN_ANIMATE_FIXED_SIZE:-0}" == "1" ]]; then
  RESIZE_ARGS=(--height "$WAN_ANIMATE_HEIGHT" --width "$WAN_ANIMATE_WIDTH")
  echo "resize: fixed crop ${WAN_ANIMATE_HEIGHT}x${WAN_ANIMATE_WIDTH}"
else
  echo "resize: area budget max_pixels=${WAN_ANIMATE_MAX_PIXELS} (reference ${WAN_ANIMATE_HEIGHT}x${WAN_ANIMATE_WIDTH}), aspect preserved"
fi

if [[ "$WAN_TRAIN_NUM_GPUS" == "1" ]]; then
  ACCEL_CONFIG="examples/wanvideo/model_training/full/accelerate_config_1gpu.yaml"
  ACCEL_NUM_PROCS=1
else
  ACCEL_CONFIG="examples/wanvideo/model_training/full/accelerate_config_multi_gpu.yaml"
  ACCEL_NUM_PROCS="$WAN_TRAIN_NUM_GPUS"
fi
echo "accelerate: ${ACCEL_CONFIG} num_processes=${ACCEL_NUM_PROCS} (DDP, no DeepSpeed)"

# GCP multi-GPU: NCCL/gIB + unset TORCH_NCCL_ASYNC_ERROR_HANDLING (see lora/wan_nccl_env.sh).
# shellcheck source=/dev/null
source "$SCRIPT_DIR/wan_nccl_env.sh"
wan_nccl_configure "$ACCEL_NUM_PROCS"

DDP_ARGS=()
INIT_CPU_ARGS=()
if [[ "$ACCEL_NUM_PROCS" != "1" ]]; then
  if [[ "${WAN_FIND_UNUSED_PARAMETERS:-0}" == "1" ]]; then
    DDP_ARGS+=(--find_unused_parameters)
  fi
  # Wan 14B: load weights on CPU first to avoid 4 ranks hammering NVMe + GPU during init.
  INIT_CPU_ARGS=(--initialize_model_on_cpu)
fi

MAIN_PORT_ARGS=()
if [[ "$ACCEL_NUM_PROCS" != "1" ]]; then
  MAIN_PORT_ARGS=(--main_process_port "${WAN_MAIN_PROCESS_PORT:-0}")
fi

SAVE_STEPS_ARGS=()
if [[ -n "${WAN_SAVE_STEPS:-}" ]]; then
  SAVE_STEPS_ARGS=(--save_steps "$WAN_SAVE_STEPS")
fi

accelerate launch --config_file "$ACCEL_CONFIG" --num_processes "$ACCEL_NUM_PROCS" "${MAIN_PORT_ARGS[@]}" examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATA_ROOT" \
  --dataset_metadata_path "$META_PATH" \
  --data_file_keys "$DATA_KEYS" \
  "${RESIZE_ARGS[@]}" \
  --num_frames "${WAN_ANIMATE_NUM_FRAMES:-77}" \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "$MODEL_SPECS" \
  "${FP8_FLAG[@]}" \
  --learning_rate 1e-4 \
  --num_epochs "${WAN_ANIMATE_NUM_EPOCHS:-5}" \
  "${SAVE_STEPS_ARGS[@]}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${WAN_ANIMATE_OUTPUT_PATH:-./models/train/Wan2.2-Animate-14B_lora}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "$EXTRA_INS" \
  --use_gradient_checkpointing_offload \
  --no_redirect_common_files \
  "${INIT_CPU_ARGS[@]}" \
  "${DDP_ARGS[@]}" \
  "${PRECOMP_FLAG[@]}"

