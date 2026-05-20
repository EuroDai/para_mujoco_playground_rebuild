python learning/train_jax_ppo.py \
  --env_name=ParaNontendonFR3Grasp \
  --impl=warp \
  --play_only=True \
  --load_checkpoint_path="logs/ParaNontendonFR3Grasp-20260520-032433_stage_2.9/checkpoints/last_002076180480" \
  --rscope_envs=100 \
  --deterministic_rscope=True \
  --run_evals=False \
  --num_videos=0