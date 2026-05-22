python learning/train_jax_ppo.py \
  --env_name=ParaFR3Grasp \
  --use_wandb=False \
  --impl=warp \
  --num_timesteps=1_000_000 \
  --num_envs=64 \
  --num_videos=0 \
  --rscope_envs=3 \
  --deterministic_rscope=True