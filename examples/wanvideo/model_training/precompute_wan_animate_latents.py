#!/usr/bin/env python3
"""
Precompute Wan Animate VAE-side tensors for one training step: main video latents, pose latents,
face pixel values (after the same trim as WanVideoUnit_AnimateVideoSplit), and the first-frame `y`
tensor (same as WanVideoUnit_ImageEmbedderVAE without end_image).

Order matches the pipeline: build one reference latent slot plus target-video latent slots from
len(video)-4 frames; trim pose/face to len(video)-4; then encode pose and preprocess face. Writes
one dict per row via torch.save and a CSV column `wan_latent_cache`.

Training: --precomputed_video_latents, metadata with wan_latent_cache (+ t5_context/clip_feature if using
--precomputed_t5_clip). Omit Wan2.1_VAE.pth from --model_id_with_origin_paths.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio  # type: ignore
import pandas as pd
import torch
from PIL import Image

from diffsynth.core.data.operators import ImageCropAndResize, LoadVideo, ToAbsolutePath
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.pipelines.wan_video import WanVideoPipeline


class _DitStub:
    require_clip_embedding = True
    require_vae_embedding = True
    has_image_pos_emb = False


def encode_y_condition(
    pipe: WanVideoPipeline,
    ref_frame_pil,
    num_frames: int,
    height: int,
    width: int,
    *,
    mask_first_frame: bool,
) -> torch.Tensor:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    msk = torch.zeros(1, num_frames, height // 8, width // 8, device=pipe.device)
    vae_input = torch.zeros(3, num_frames, height, width, device=pipe.device, dtype=pipe.torch_dtype)
    if ref_frame_pil is not None:
        image = pipe.preprocess_image(ref_frame_pil.resize((width, height))).to(pipe.device)
        vae_input[:, :1] = image.transpose(0, 1).to(dtype=pipe.torch_dtype)
    if mask_first_frame:
        msk[:, :1] = 1
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.transpose(1, 2)[0]
    y = pipe.vae.encode(
        [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
        device=pipe.device,
        tiled=False,
    )[0]
    y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
    y = torch.concat([msk, y])
    y = y.unsqueeze(0).to(dtype=pipe.torch_dtype)
    return y


def compute_y_tensor(pipe: WanVideoPipeline, ref_frame_pil, target_num_frames: int, height: int, width: int) -> torch.Tensor:
    """Match WanVideoUnit_ImageEmbedderVAE for Animate training."""
    y_ref = encode_y_condition(pipe, ref_frame_pil, 1, height, width, mask_first_frame=True)
    y_target = encode_y_condition(pipe, None, target_num_frames, height, width, mask_first_frame=False)
    return torch.concat([y_ref, y_target], dim=2).cpu()


def load_reference_frame(video_path: str, frame_index: int, frame_processor) -> object:
    reader = imageio.get_reader(video_path)
    try:
        total = int(reader.count_frames())
        if total <= 0:
            raise ValueError(f"video has no frames: {video_path}")
        frame_index = max(0, min(int(frame_index), total - 1))
        frame = reader.get_data(frame_index)
    finally:
        reader.close()
    return frame_processor(Image.fromarray(frame).convert("RGB"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_base_path", type=str, required=True)
    p.add_argument("--dataset_metadata_path", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True, help="Folder containing Wan2.1_VAE.pth")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=33)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_metadata_name", type=str, default="metadata_precomputed_latents.csv")
    args = p.parse_args()

    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    base = os.path.abspath(args.dataset_base_path)
    meta_path = os.path.abspath(args.dataset_metadata_path)
    out_csv = os.path.join(base, args.out_metadata_name)
    model_dir = os.path.abspath(args.model_dir)

    required = ("video", "animate_pose_video", "animate_face_video")
    df = pd.read_csv(meta_path)
    for c in required:
        if c not in df.columns:
            sys.exit(f"metadata must contain column {c!r}")

    video_loader = (
        ToAbsolutePath(base)
        >> LoadVideo(
            args.num_frames,
            4,
            1,
            frame_processor=ImageCropAndResize(args.height, args.width, None, 16, 16),
        )
    )
    ref_frame_processor = ImageCropAndResize(args.height, args.width, None, 16, 16)
    pose_loader = video_loader
    face_loader = (
        ToAbsolutePath(base)
        >> LoadVideo(
            args.num_frames,
            4,
            1,
            frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
        )
    )

    dtm = DiffusionTrainingModule()
    vae_spec = f"{model_dir}:Wan2.1_VAE.pth"
    configs = dtm.parse_model_configs(None, vae_spec, fp8_models=None, offload_models=None, device=args.device)
    pipe = WanVideoPipeline(device=args.device, torch_dtype=torch.bfloat16)
    pipe.dit = _DitStub()
    model_pool = pipe.download_and_load_models(configs)
    pipe.vae = model_pool.fetch_model("wan_video_vae")

    os.makedirs(os.path.join(base, "precomputed", "latents"), exist_ok=True)
    rel_paths = []

    with torch.no_grad():
        for i, row in df.iterrows():
            vid = video_loader(row["video"])
            pose = pose_loader(row["animate_pose_video"])
            face = face_loader(row["animate_face_video"])
            n = len(vid)
            if n < 5:
                sys.exit(f"row {i}: need at least 5 frames after load, got {n}")
            target_num_frames = n - 4
            pose_trim = pose[: n - 4]
            face_trim = face[: n - 4]

            pv = pipe.preprocess_video(vid)
            if "clip_frame_index" in row and not pd.isna(row["clip_frame_index"]):
                video_path = row["video"]
                if not os.path.isabs(str(video_path)):
                    video_path = os.path.join(base, str(video_path))
                ref_frame = load_reference_frame(str(video_path), int(row["clip_frame_index"]), ref_frame_processor)
            else:
                ref_frame = vid[0]
            ref_pv = pipe.preprocess_video([ref_frame])
            ref_latents = pipe.vae.encode(ref_pv, device=pipe.device, tiled=False).to(dtype=pipe.torch_dtype)
            target_latents = pipe.vae.encode(pv[:, :, :target_num_frames], device=pipe.device, tiled=False).to(dtype=pipe.torch_dtype)
            input_latents = torch.concat([ref_latents, target_latents], dim=2)

            pose_pv = pipe.preprocess_video(pose_trim)
            pose_latents = pipe.vae.encode(pose_pv, device=pipe.device, tiled=False).to(dtype=pipe.torch_dtype)

            face_pixel_values = pipe.preprocess_video(face_trim)

            w_pil, h_pil = vid[0].size
            h2, w2, n2 = pipe.check_resize_height_width(h_pil, w_pil, n, verbose=0)
            if n2 != n:
                sys.exit(
                    f"row {i}: pipeline shape check would change num_frames {n} -> {n2}. "
                    "Use a num_frames compatible with the pipeline time_division_factor / remainder (same as training)."
                )
            y = compute_y_tensor(pipe, ref_frame, target_num_frames, h2, w2)

            bundle = {
                "input_latents": input_latents.cpu().to(torch.bfloat16),
                "pose_latents": pose_latents.cpu().to(torch.bfloat16),
                "face_pixel_values": face_pixel_values.cpu().to(torch.bfloat16),
                "y": y.to(torch.bfloat16),
                "height": h2,
                "width": w2,
                "num_frames": n2,
            }
            rel = f"precomputed/latents/{i:06d}.pt"
            torch.save(bundle, os.path.join(base, rel))
            rel_paths.append(rel)

    df = df.copy()
    df["wan_latent_cache"] = rel_paths
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with column wan_latent_cache ({len(df)} rows).")


if __name__ == "__main__":
    main()
