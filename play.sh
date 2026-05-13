python learning/train_jax_ppo.py \
  --env_name=ParaNontendonFR3Grasp \
  --impl=warp \
  --play_only=True \
  --load_checkpoint_path="logs/ParaNontendonFR3Grasp-20260513-150427/checkpoints" \
  --rscope_envs=1 \
  --deterministic_rscope=True \
  --run_evals=False \
  --num_videos=1