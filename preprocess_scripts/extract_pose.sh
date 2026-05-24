conda run -p /home/tungdo/storage/tungdt/env_dir/animate python /home/tungdo/storage/tungdt/preprocessing_code/extract_pose_videos.py \
  --input-dir /home/tungdo/storage/tungdt/wan_animate_data/tiktok-videos_77/chunks_77f_30fps \
  --output-dir /home/tungdo/storage/tungdt/wan_animate_data/tiktok-videos_77/chunks_77f_30fps_pose \
  --ckpt-dir /home/tungdo/storage/tungdt/Wan2.2-Animate-14B/process_checkpoint \
  --backend vitpose \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --num-workers 8 \
  --batch-size 32 \
  --mode black \
  --skip-existing