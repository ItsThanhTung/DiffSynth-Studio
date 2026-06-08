#!/usr/bin/env python3
"""
Precompute Wan T5 text embeddings and CLIP image embeddings for Wan Animate (and similar) training.

What each model does (in this pipeline):
- T5 (wan_video_text_encoder): maps the prompt string to token embeddings `context` (fed to dit.text_embedding in model_fn).
- CLIP (wan_video_image_encoder): maps one video frame (see ``--clip_frame_mode``) to `clip_feature` (fed to dit.img_emb, then concatenated with text).
  Geometry is controlled by ``--clip_geometry_mode``: fixed crop box (legacy) vs. **area budget** (resize with constant aspect so pixel count ``≈ width * height``).

This script loads only T5 + CLIP (+ tokenizer), runs them once per CSV row, saves float tensors to disk, and writes
`metadata_precomputed.csv` with extra columns `t5_context` and `clip_feature` (absolute paths to .pt files).
Default CLIP checkpoint is full bf16 (`models_clip_...-14.pth`); pass `--use_clip_fp8` to use the fp8 file instead.

Multi-GPU: pass ``--num_workers N`` (and optional ``--devices cuda:0,cuda:1,...``). Each worker loads models on one GPU
and processes rows where ``position % N == worker_rank``. Only the parent process writes the output metadata CSV.

Training: use --precomputed_t5_clip and omit T5/CLIP from --model_id_with_origin_paths; keep DiT, VAE, etc.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio  # type: ignore
import pandas as pd
import torch
from PIL import Image

from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.core.loader.config import ModelConfig
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
from diffsynth.pipelines.wan_video import WanVideoPipeline


class _DitStub:
    """Minimal flags used by WanVideoUnit_ImageEmbedderCLIP during precompute."""

    require_clip_embedding = True
    require_vae_embedding = True
    has_image_pos_emb = False


def resize_pil_to_target_pixel_area(im: Image.Image, target_area: int) -> Image.Image:
    """Uniform scale so ``W*H <= target_area`` and as large as possible (aspect ratio preserved)."""
    w0, h0 = im.size
    if w0 <= 0 or h0 <= 0 or target_area <= 0:
        return im
    s = math.sqrt(target_area / (w0 * h0))
    wn, hn = 1, 1
    for _ in range(32):
        wn = max(1, int(math.floor(w0 * s)))
        hn = max(1, int(round(wn * h0 / w0)))
        wn = max(1, int(round(hn * w0 / h0)))
        if wn * hn <= target_area:
            break
        s *= math.sqrt(target_area / max(1, wn * hn)) * 0.999
    return im.resize((wn, hn), Image.Resampling.LANCZOS)


def encode_prompt_text(pipe: WanVideoPipeline, prompt: str) -> torch.Tensor:
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


def write_output_metadata(
    df: pd.DataFrame,
    base: str,
    out_csv: str,
    *,
    shared_t5: bool,
    shared_t5_rel: str,
    precomputed_dir: str,
    include_clip_frame_index: bool = False,
) -> None:
    t5_paths = []
    clip_paths = []
    clip_frame_indices = []
    for i, _row in df.iterrows():
        if shared_t5:
            t5_paths.append(os.path.abspath(os.path.join(base, shared_t5_rel)))
        else:
            t5_paths.append(os.path.abspath(os.path.join(base, f"{precomputed_dir}/t5/{i:06d}.pt")))
        clip_paths.append(os.path.abspath(os.path.join(base, f"{precomputed_dir}/clip/{i:06d}.pt")))
        if include_clip_frame_index:
            frame_index_path = os.path.join(base, f"{precomputed_dir}/clip_frame_index/{i:06d}.txt")
            if not os.path.isfile(frame_index_path):
                sys.exit(
                    f"missing CLIP frame index file: {frame_index_path}. "
                    "Regenerate random CLIP precompute outputs so VAE latents can use the same reference frame."
                )
            with open(frame_index_path, "r", encoding="utf-8") as f:
                clip_frame_indices.append(int(f.read().strip()))
    df = df.copy()
    df["t5_context"] = t5_paths
    df["clip_feature"] = clip_paths
    if include_clip_frame_index:
        df["clip_frame_index"] = clip_frame_indices
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with columns t5_context, clip_feature ({len(df)} rows).", flush=True)


def wait_for_file(path: str, timeout_s: float = 3600.0, poll_s: float = 0.5) -> None:
    deadline = time.monotonic() + timeout_s
    while not os.path.isfile(path):
        if time.monotonic() > deadline:
            sys.exit(f"Timed out waiting for file: {path}")
        time.sleep(poll_s)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_base_path", type=str, required=True)
    p.add_argument("--dataset_metadata_path", type=str, required=True)
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Local Wan2.2-Animate-14B (or compatible) folder with T5 .pth and CLIP .pth/.fp8.pth",
    )
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num_frames", type=int, default=33)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--use_clip_fp8",
        action="store_true",
        help="Load CLIP fp8 weights (less VRAM). Default: full CLIP bf16 (.pth), no fp8.",
    )
    p.add_argument(
        "--out_metadata_name",
        type=str,
        default="metadata_precomputed.csv",
        help="Written under dataset_base_path (not the same folder as input CSV unless you set path accordingly).",
    )
    p.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Directory with UMT5 tokenizer files (tokenizer.json, etc.). Default: <model_dir>/google/umt5-xxl if present.",
    )
    p.add_argument(
        "--shared_t5",
        action="store_true",
        help="Encode T5 once and reuse one .pt for every row (same prompt). CLIP still per row (see --clip_frame_mode).",
    )
    p.add_argument(
        "--shared_t5_prompt",
        type=str,
        default=None,
        help="With --shared_t5: use this string for T5 instead of validating the prompt column.",
    )
    p.add_argument(
        "--clip_frame_mode",
        type=str,
        choices=("first", "random"),
        default="first",
        help="Which frame to feed CLIP: first frame of the file, or one uniform random frame per row.",
    )
    p.add_argument(
        "--precomputed_dir",
        type=str,
        default=None,
        help="Subfolder under dataset_base_path for output .pt files (default: 'precomputed' for first, "
        "'precomputed_random' for random).",
    )
    p.add_argument(
        "--clip_random_seed",
        type=int,
        default=None,
        help="With clip_frame_mode=random: seed for reproducible frame picks (single RNG across rows).",
    )
    p.add_argument(
        "--clip_geometry_mode",
        type=str,
        choices=("crop_fixed", "area_budget"),
        default="crop_fixed",
        help="crop_fixed: scale+center-crop to (height,width) like training loaders. "
        "area_budget: target pixel count = height*width; resize preserving aspect (no crop), area ≤ that budget.",
    )
    p.add_argument(
        "--clip_batch_size",
        type=int,
        default=8,
        help="Number of CLIP images per forward pass (higher uses more VRAM, better GPU utilization).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Spawn N independent processes (sharded by row position %% N).",
    )
    p.add_argument(
        "--worker_rank",
        type=int,
        default=None,
        help="Internal: worker index in [0, num_workers). Processes only its shard.",
    )
    p.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated devices for multi-worker mode, e.g. cuda:0,cuda:1,cuda:2,cuda:3.",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip rows whose CLIP .pt already exists (and per-row T5 .pt when not using --shared_t5).",
    )
    return p.parse_args(argv)


def launch_workers(args: argparse.Namespace) -> None:
    n = int(args.num_workers)
    worker_script = os.path.abspath(__file__)
    devices = None
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        if not devices:
            sys.exit("--devices was provided but no valid devices were found")

    base_cmd = [
        sys.executable,
        worker_script,
        "--dataset_base_path",
        args.dataset_base_path,
        "--dataset_metadata_path",
        args.dataset_metadata_path,
        "--model_dir",
        args.model_dir,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_frames",
        str(args.num_frames),
        "--out_metadata_name",
        args.out_metadata_name,
        "--clip_frame_mode",
        args.clip_frame_mode,
        "--clip_geometry_mode",
        args.clip_geometry_mode,
        "--clip_batch_size",
        str(args.clip_batch_size),
        "--num_workers",
        str(n),
    ]
    pdir = args.precomputed_dir or ("precomputed_random" if args.clip_frame_mode == "random" else "precomputed")
    base_cmd += ["--precomputed_dir", pdir]
    if args.use_clip_fp8:
        base_cmd.append("--use_clip_fp8")
    if args.tokenizer_path is not None:
        base_cmd += ["--tokenizer_path", args.tokenizer_path]
    if args.shared_t5:
        base_cmd.append("--shared_t5")
        if args.shared_t5_prompt is not None:
            base_cmd += ["--shared_t5_prompt", args.shared_t5_prompt]
    if args.clip_random_seed is not None:
        base_cmd += ["--clip_random_seed", str(args.clip_random_seed)]
    if args.skip_existing:
        base_cmd.append("--skip_existing")

    procs: list[subprocess.Popen] = []
    for r in range(n):
        worker_device = devices[r % len(devices)] if devices else args.device
        cmd = base_cmd + ["--device", worker_device, "--worker_rank", str(r)]
        print(f"Starting worker {r}/{n} on {worker_device}", flush=True)
        procs.append(subprocess.Popen(cmd))

    exit_code = 0
    for proc in procs:
        rc = proc.wait()
        if rc != 0 and exit_code == 0:
            exit_code = int(rc)
    if exit_code != 0:
        sys.exit(exit_code)

    base = os.path.abspath(args.dataset_base_path)
    meta_path = os.path.abspath(args.dataset_metadata_path)
    out_csv = os.path.join(base, args.out_metadata_name)
    df = pd.read_csv(meta_path)
    pdir = args.precomputed_dir or ("precomputed_random" if args.clip_frame_mode == "random" else "precomputed")
    shared_t5_rel = f"{pdir}/t5/shared_single_prompt.pt"
    write_output_metadata(
        df,
        base,
        out_csv,
        shared_t5=args.shared_t5,
        shared_t5_rel=shared_t5_rel,
        precomputed_dir=pdir,
        include_clip_frame_index=args.clip_frame_mode == "random",
    )


def run_worker(args: argparse.Namespace) -> None:
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    base = os.path.abspath(args.dataset_base_path)
    meta_path = os.path.abspath(args.dataset_metadata_path)
    out_csv = os.path.join(base, args.out_metadata_name)

    model_dir = os.path.abspath(args.model_dir)
    clip_name = (
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14-fp8.pth"
        if args.use_clip_fp8
        else "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )
    t5_spec = f"{model_dir}:models_t5_umt5-xxl-enc-bf16.pth"
    clip_spec = f"{model_dir}:{clip_name}"
    fp8_models = clip_spec if args.use_clip_fp8 else None

    dtm = DiffusionTrainingModule()
    configs = dtm.parse_model_configs(
        None,
        f"{t5_spec},{clip_spec}",
        fp8_models=fp8_models,
        offload_models=None,
        device=args.device,
    )

    pipe = WanVideoPipeline(device=args.device, torch_dtype=torch.bfloat16)
    pipe.dit = _DitStub()
    model_pool = pipe.download_and_load_models(configs)
    pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
    pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")

    if args.tokenizer_path is not None:
        tok_path = os.path.abspath(args.tokenizer_path)
    else:
        bundled = os.path.join(model_dir, "google", "umt5-xxl")
        if os.path.isdir(bundled) and os.path.isfile(os.path.join(bundled, "tokenizer.json")):
            tok_path = bundled
        else:
            tokenizer_config = ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"
            )
            tokenizer_config.download_if_necessary()
            tok_path = tokenizer_config.path
            if isinstance(tok_path, list):
                tok_path = tok_path[0] if len(tok_path) == 1 else None
            if not tok_path or not os.path.isdir(str(tok_path)):
                sys.exit(
                    "Could not find UMT5 tokenizer. With DIFFSYNTH_SKIP_DOWNLOAD, place files under "
                    f"{bundled} or pass --tokenizer_path /path/to/google/umt5-xxl"
                )
    pipe.tokenizer = HuggingfaceTokenizer(name=tok_path, seq_len=512, clean="whitespace")

    target_area = max(1, int(args.width) * int(args.height))
    rank = args.worker_rank
    n_workers = int(args.num_workers)
    is_multi = n_workers > 1 and rank is not None
    worker_label = f"worker {rank}/{n_workers} on {args.device}" if is_multi else args.device

    if args.clip_geometry_mode == "area_budget":
        print(
            f"[{worker_label}] CLIP geometry: area_budget "
            f"(target pixel count width*height = {args.width}*{args.height} = {target_area})",
            flush=True,
        )
    else:
        print(
            f"[{worker_label}] CLIP geometry: crop_fixed ({args.height}x{args.width}, 16-div crop+resize)",
            flush=True,
        )
    print(f"[{worker_label}] CLIP batch size: {args.clip_batch_size}", flush=True)

    clip_frame_proc = None
    if args.clip_geometry_mode == "crop_fixed":
        clip_frame_proc = ImageCropAndResize(args.height, args.width, None, 16, 16)
    clip_rng = random.Random()

    def resolve_video_path(rel_or_abs: str) -> str:
        pth = str(rel_or_abs).strip()
        if os.path.isabs(pth):
            return os.path.abspath(pth)
        return os.path.abspath(os.path.join(base, pth))

    def load_clip_pil(video_path: str, *, row_pos: int) -> tuple[Image.Image, int]:
        reader = imageio.get_reader(video_path)
        try:
            total = int(reader.count_frames())
            if total <= 0:
                raise ValueError("video has no frames")
            if args.clip_frame_mode == "first":
                idx = 0
            elif args.clip_random_seed is not None:
                idx = random.Random(args.clip_random_seed + row_pos).randint(0, total - 1)
            else:
                idx = clip_rng.randint(0, total - 1)
            frame = reader.get_data(idx)
        finally:
            reader.close()
        im = Image.fromarray(frame).convert("RGB")
        if args.clip_geometry_mode == "crop_fixed":
            assert clip_frame_proc is not None
            return clip_frame_proc(im), idx
        return resize_pil_to_target_pixel_area(im, target_area), idx

    df = pd.read_csv(meta_path)
    if "video" not in df.columns or "prompt" not in df.columns:
        sys.exit("metadata must contain at least columns: video, prompt")

    pdir = args.precomputed_dir or ("precomputed_random" if args.clip_frame_mode == "random" else "precomputed")
    os.makedirs(os.path.join(base, pdir, "t5"), exist_ok=True)
    os.makedirs(os.path.join(base, pdir, "clip"), exist_ok=True)
    if args.clip_frame_mode == "random":
        os.makedirs(os.path.join(base, pdir, "clip_frame_index"), exist_ok=True)

    shared_t5_rel = f"{pdir}/t5/shared_single_prompt.pt"
    shared_t5_abs = os.path.join(base, shared_t5_rel)

    if args.shared_t5:
        if rank is None or rank == 0:
            if args.shared_t5_prompt is not None:
                t5_prompt = str(args.shared_t5_prompt)
            else:
                prompts = [str(x) for x in df["prompt"].tolist()]
                if len(set(prompts)) != 1:
                    sys.exit(
                        "--shared_t5 needs one prompt for all rows, or pass --shared_t5_prompt. "
                        f"Found {len(set(prompts))} distinct prompt(s)."
                    )
                t5_prompt = prompts[0]
            if not (args.skip_existing and os.path.isfile(shared_t5_abs)):
                with torch.no_grad():
                    shared_ctx = encode_prompt_text(pipe, t5_prompt).cpu()
                torch.save(shared_ctx, shared_t5_abs)
                print(f"[{worker_label}] Wrote shared T5 -> {shared_t5_rel} (chars={len(t5_prompt)})", flush=True)
        elif is_multi:
            wait_for_file(shared_t5_abs)

    clip_batch_size = max(1, int(args.clip_batch_size))
    pending_idx: list = []
    pending_images: list[torch.Tensor] = []
    n_done = n_skip = 0

    def flush_clip_batch() -> None:
        nonlocal n_done
        if not pending_images:
            return
        with torch.no_grad():
            clip_out = pipe.image_encoder.encode_image(pending_images)
        clip_out = clip_out.to(dtype=pipe.torch_dtype, device="cpu")
        for b, row_idx in enumerate(pending_idx):
            clip_f = clip_out[b : b + 1] if clip_out.dim() > 1 else clip_out
            clip_rel = f"{pdir}/clip/{row_idx:06d}.pt"
            torch.save(clip_f, os.path.join(base, clip_rel))
            n_done += 1
        pending_idx.clear()
        pending_images.clear()

    def preprocess_clip_tensor(clip_pil: Image.Image) -> torch.Tensor:
        if args.clip_geometry_mode == "crop_fixed":
            pil = clip_pil.resize((args.width, args.height))
        else:
            pil = clip_pil
        return pipe.preprocess_image(pil).to(pipe.device)

    def row_already_done(row_idx) -> bool:
        clip_abs = os.path.join(base, f"{pdir}/clip/{row_idx:06d}.pt")
        if not os.path.isfile(clip_abs):
            return False
        if args.shared_t5:
            return True
        return os.path.isfile(os.path.join(base, f"{pdir}/t5/{row_idx:06d}.pt"))

    with torch.no_grad():
        for pos, (row_idx, row) in enumerate(df.iterrows()):
            if is_multi and pos % n_workers != rank:
                continue

            if args.skip_existing and row_already_done(row_idx):
                n_skip += 1
                continue

            rel_video = row["video"]
            prompt = str(row["prompt"])
            abs_video = resolve_video_path(rel_video)
            if not os.path.isfile(abs_video):
                sys.exit(f"missing video file: {abs_video}")

            if not args.shared_t5:
                t5_rel = f"{pdir}/t5/{row_idx:06d}.pt"
                t5_abs = os.path.join(base, t5_rel)
                if not (args.skip_existing and os.path.isfile(t5_abs)):
                    ctx = encode_prompt_text(pipe, prompt)
                    torch.save(ctx.cpu(), t5_abs)

            clip_pil, clip_frame_index = load_clip_pil(abs_video, row_pos=pos)
            if args.clip_frame_mode == "random":
                with open(os.path.join(base, pdir, "clip_frame_index", f"{row_idx:06d}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{clip_frame_index}\n")
            pending_idx.append(row_idx)
            pending_images.append(preprocess_clip_tensor(clip_pil))
            if len(pending_images) >= clip_batch_size:
                flush_clip_batch()

        flush_clip_batch()

    if is_multi:
        print(f"[{worker_label}] done: encoded={n_done} skipped={n_skip}", flush=True)
        return

    write_output_metadata(
        df,
        base,
        out_csv,
        shared_t5=args.shared_t5,
        shared_t5_rel=shared_t5_rel,
        precomputed_dir=pdir,
        include_clip_frame_index=args.clip_frame_mode == "random",
    )
    if n_skip:
        print(f"Skipped {n_skip} rows (--skip_existing).", flush=True)


def main():
    args = parse_args()
    if args.worker_rank is None and int(args.num_workers) > 1:
        launch_workers(args)
        return
    run_worker(args)


if __name__ == "__main__":
    main()
