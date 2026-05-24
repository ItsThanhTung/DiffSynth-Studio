import os

# GCP gIB + PyTorch 2.x: on multi-GPU workers, nccl-shim requires these unset before import torch.
if os.environ.get("LOCAL_RANK") is not None:
    _wan_nccl_mode = os.environ.get("WAN_NCCL_MODE", "")
    for _nccl_key in (
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "WAN_NCCL_MODE",
    ):
        os.environ.pop(_nccl_key, None)
    # gib_only: drop user IB-disable overrides; keep image tuning (NCCL_IB_TC, etc.).
    if _wan_nccl_mode in ("", "gib_only"):
        for _nccl_key in ("NCCL_IB_DISABLE", "NCCL_IBEXT_DISABLE", "NCCL_SOCKET_IFNAME"):
            os.environ.pop(_nccl_key, None)

import torch, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath, LoadTorchPickle
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# When using precomputed T5 + latents, text conditioning comes from `t5_context` files, not this string.
# Must not put free-text `prompt` in --data_file_keys (UnifiedDataset would run the video path loader on it).
WAN_ANIMATE_FIXED_TRAINING_PROMPT = "A person performing natural motion for character animation training."


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        redirect_common_files=True,
        precomputed_t5_clip=False,
        precomputed_video_latents=False,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        # With precomputed T5 embeddings, the prompt embedder never runs the tokenizer; skip loading it (avoids HF path issues under DIFFSYNTH_SKIP_DOWNLOAD).
        if precomputed_t5_clip and tokenizer_path is None:
            tokenizer_config = None
        elif tokenizer_path is not None:
            tokenizer_config = ModelConfig(path=os.path.abspath(tokenizer_path))
        else:
            tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            redirect_common_files=redirect_common_files,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.precomputed_video_latents = precomputed_video_latents
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "t5_context":
                # Precomputed Wan T5 text embeddings (same tensor as PromptEmbedder.encode_prompt output).
                inputs_shared["precomputed_context"] = data["t5_context"]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            vid = data.get("video")
            if vid is not None:
                inputs_shared["num_frames"] = 4 * (len(vid) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_nega = {}
        if self.precomputed_video_latents and "wan_latent_cache" in data:
            inputs_posi = {
                "prompt": os.environ.get("WAN_ANIMATE_TRAINING_PROMPT", WAN_ANIMATE_FIXED_TRAINING_PROMPT),
            }
            b = data["wan_latent_cache"]
            if not isinstance(b, dict):
                raise TypeError("wan_latent_cache must be a dict loaded from .pt (see precompute_wan_animate_latents.py).")
            inputs_shared = {
                "input_video": None,
                "input_image": None,
                "height": int(b["height"]),
                "width": int(b["width"]),
                "num_frames": int(b["num_frames"]),
                "input_latents": b["input_latents"],
                "pose_latents": b["pose_latents"],
                "y": b.get("y"),
                "animate_pose_video": None,
                "animate_face_video": None,
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
            }
            fv = b["face_pixel_values"]
            inputs_posi["face_pixel_values"] = fv
            inputs_nega["face_pixel_values"] = torch.zeros_like(fv) - 1
        else:
            inputs_posi = {"prompt": data["prompt"]}
            inputs_shared = {
                # Assume you are using this pipeline for inference,
                # please fill in the input parameters.
                "input_video": data["video"],
                "height": data["video"][0].size[1],
                "width": data["video"][0].size[0],
                "num_frames": len(data["video"]),
                # Please do not modify the following parameters
                # unless you clearly know what this will cause.
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
            }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument(
        "--no_redirect_common_files",
        default=False,
        action="store_true",
        help="Disable redirect of shared Wan weights to ModelScope safetensors; use when loading from a local Wan folder with .pth files.",
    )
    parser.add_argument(
        "--precomputed_t5_clip",
        default=False,
        action="store_true",
        help="Metadata includes columns t5_context and clip_feature (paths to .pt tensors from precompute_t5_clip_embeddings.py). Omit T5/CLIP from model configs; tokenizer is not loaded unless --tokenizer_path is set.",
    )
    parser.add_argument(
        "--precomputed_video_latents",
        default=False,
        action="store_true",
        help="Metadata column wan_latent_cache (path to .pt dict: input_latents, pose_latents, face_pixel_values, y, height, width, num_frames). Skips VAE encode in the training loop; omit Wan2.1_VAE.pth from model configs.",
    )
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
            **(
                {
                    "t5_context": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(map_location="cpu"),
                    "clip_feature": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(map_location="cpu"),
                }
                if args.precomputed_t5_clip
                else {}
            ),
            **(
                {"wan_latent_cache": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(map_location="cpu")}
                if args.precomputed_video_latents
                else {}
            ),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        redirect_common_files=not args.no_redirect_common_files,
        precomputed_t5_clip=args.precomputed_t5_clip,
        precomputed_video_latents=args.precomputed_video_latents,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
