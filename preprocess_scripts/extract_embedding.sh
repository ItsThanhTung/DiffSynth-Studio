#!/usr/bin/env bash
set -euo pipefail
# Match conda env used by precompute-t5-clip.sh (default env name: animate).
export WAN_TRAIN_CONDA_ENV="/home/tungdo/storage/tungdt/env_dir/animate"

export WAN_PRECOMPUTE_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3
export WAN_PRECOMPUTE_NUM_WORKERS=4
export WAN_CLIP_BATCH_SIZE=16

export WAN_CLIP_GEOMETRY_MODE=area_budget
export WAN_ANIMATE_HEIGHT=720
export WAN_ANIMATE_WIDTH=1280
export WAN_ANIMATE_DATA_ROOT=/home/tungdo/storage/tungdt/wan_animate_data/tiktok-videos_77
export WAN_ANIMATE_METADATA_IN=/home/tungdo/storage/tungdt/wan_animate_data/tiktok-videos_77/metadata_training.csv
export WAN_ANIMATE_MODEL_DIR=/home/tungdo/storage/tungdt/Wan2.2-Animate-14B
export WAN_CLIP_FRAME_MODE=random
# Single T5 encoding for all rows (CLIP still per-video frame).
export WAN_PRECOMPUTE_SHARED_T5=1
export WAN_PRECOMPUTE_SHARED_T5_PROMPT="${WAN_PRECOMPUTE_SHARED_T5_PROMPT:-视频中的人在做动作}"
export WAN_PRECOMPUTE_OUT_METADATA=metadata_precomputed_clip_random.csv
# Multi-GPU + larger CLIP batches (tune batch size if OOM).
if [[ -z "${WAN_PRECOMPUTE_NUM_WORKERS:-}" ]]; then
  _ngpu="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [[ -z "$_ngpu" || "$_ngpu" -lt 1 ]] && _ngpu=1
  export WAN_PRECOMPUTE_NUM_WORKERS="$_ngpu"
fi
if [[ -z "${WAN_PRECOMPUTE_DEVICES:-}" ]]; then
  _devs=()
  for (( _i=0; _i<WAN_PRECOMPUTE_NUM_WORKERS; _i++ )); do
    _devs+=("cuda:${_i}")
  done
  export WAN_PRECOMPUTE_DEVICES="$(
    IFS=,
    echo "${_devs[*]}"
  )"
fi
export WAN_CLIP_BATCH_SIZE="${WAN_CLIP_BATCH_SIZE:-16}"
bash /home/tungdo/storage/tungdt/DiffSynth-Studio/examples/wanvideo/model_training/lora/precompute-t5-clip.sh


# export WAN_ANIMATE_DATA_ROOT=/home/tungdo/storage/tungdt/wan_animate_data/tiktok-videos_77
# export WAN_ANIMATE_METADATA_IN=/home/tungdo/storage/tungdt/wan_animate_data/tiktok-videos_77/metadata_training.csv
# export WAN_ANIMATE_MODEL_DIR=/home/tungdo/storage/tungdt/Wan2.2-Animate-14B
# export WAN_CLIP_FRAME_MODE=random
# export WAN_CLIP_RANDOM_SEED=42
# export WAN_PRECOMPUTE_SHARED_T5=1
# export WAN_PRECOMPUTE_SHARED_T5_PROMPT="视频中的人在做动作"
# export WAN_PRECOMPUTE_OUT_METADATA=metadata_precomputed_clip_random.csv
# bash /home/tungdo/storage/tungdt/DiffSynth-Studio/examples/wanvideo/model_training/lora/precompute-t5-clip.sh