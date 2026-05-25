python learning/train_jax_ppo.py \
  --env_name=ParaFR3Grasp \
  --impl=warp \
  --play_only=True \
  --load_checkpoint_path="logs/ParaFR3Grasp-20260524-165214/checkpoints/002076180480" \
  --rscope_envs=100 \
  --deterministic_rscope=True \
  --run_evals=False \
  --num_videos=0