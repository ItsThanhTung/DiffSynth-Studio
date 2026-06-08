# Wan Animate Training Fix Notes

This note summarizes the local changes made to align Wan Animate training with the inference layout used by LightX2V-style Animate.

## Problem

Original training encoded the target as a continuous full video:

```python
input_latents = vae.encode(full_video)  # e.g. 81 frames -> 21 latent slots
```

But Animate conditioning inserts pose into `x[:, :, 1:]`, and inference decodes after dropping slot 0. That implies this layout:

```text
slot 0   = reference/control latent
slot 1.. = output video latents
```

For the default 81-frame training window:

```text
loaded frames          = 81
output/target frames   = 77
reference latent slots = 1
target latent slots    = 20
total latent slots     = 21
```

The old code instead treated slot 0 as the first latent of a continuous 81-frame target video, while pose started at slot 1.

## Code Changes

### 1. `diffsynth/pipelines/wan_video.py`

Update `WanVideoUnit_InputVideoEmbedder` so Animate training builds reference-prefixed latents:

```python
elif pipe.scheduler.training and input_image is not None and animate_pose_video is not None and animate_face_video is not None:
    target_video = input_video[:, :, :input_video.shape[2] - 4]
    ref_latents = pipe.vae.encode(input_video[:, :, :1], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
    target_latents = pipe.vae.encode(target_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
    input_latents = torch.concat([ref_latents, target_latents], dim=2)
```

Also add `input_image`, `animate_pose_video`, and `animate_face_video` to that unit's `input_params` and `process()` signature.

Update `WanVideoUnit_ImageEmbedderVAE` so Animate training builds a split `y`:

```text
y = concat(y_ref, y_target_zeros, dim=time)
```

where:

```text
y_ref          = mask/ref latent for the one-frame reference slot
y_target_zeros = zero-mask/zero-latents for the 77-frame target slots
```

Also add `animate_pose_video` and `animate_face_video` to that unit's `input_params` and `process()` signature.

### 2. `examples/wanvideo/model_training/precompute_wan_animate_latents.py`

Update latent cache generation to match the online training layout:

```python
ref_latents = pipe.vae.encode(ref_pv, device=pipe.device, tiled=False).to(dtype=pipe.torch_dtype)
target_latents = pipe.vae.encode(pv[:, :, : n - 4], device=pipe.device, tiled=False).to(dtype=pipe.torch_dtype)
input_latents = torch.concat([ref_latents, target_latents], dim=2)
```

Update `compute_y_tensor()` to build:

```text
y = concat(y_ref, y_reft, dim=time)
```

instead of one continuous `[first_frame, zeros...]` condition.

### 3. `examples/wanvideo/model_training/precompute_t5_clip_embeddings.py`

If using random CLIP frame augmentation, save the exact chosen frame index:

```text
precomputed_random/clip_frame_index/000000.txt
```

and write it into metadata as:

```text
clip_frame_index
```

Then `precompute_wan_animate_latents.py` reads `clip_frame_index` and uses that same frame for:

```text
CLIP reference frame
VAE reference latent
y_ref
```

This keeps random reference augmentation aligned.

## Run Order For Precomputed Training

Regenerate caches after applying the changes. Old caches use the old continuous-video layout.

```bash
# 1. Precompute T5/CLIP, optionally with random CLIP frame.
WAN_CLIP_FRAME_MODE=random \
WAN_CLIP_RANDOM_SEED=123 \
examples/wanvideo/model_training/lora/precompute-t5-clip.sh

# 2. Precompute Wan Animate VAE-side latents using metadata with clip_frame_index.
examples/wanvideo/model_training/lora/precompute-video-latents.sh

# 3. Train with precomputed T5/CLIP and video latents.
USE_VIDEO_LATENTS=1 \
examples/wanvideo/model_training/lora/Wan2.2-Animate-14B-meanflow.sh
```

## Expected Alignment

After the fix:

```text
input_latents[:, :, 0]   = reference latent
input_latents[:, :, 1:]  = target/output video latents
pose_latents             = 77-frame pose VAE latents, aligns with input_latents[:, :, 1:]
face_pixel_values        = 77 frames, face encoder prepends one pad slot
y[:, :, 0]               = reference condition
y[:, :, 1:]              = zero target condition
```

This matches the Animate inference convention:

```text
sample 21 latent slots
decode slots 1..20
output 77 frames
```

## Quick Check

Run:

```bash
python -m py_compile \
  diffsynth/pipelines/wan_video.py \
  examples/wanvideo/model_training/precompute_t5_clip_embeddings.py \
  examples/wanvideo/model_training/precompute_wan_animate_latents.py
```
