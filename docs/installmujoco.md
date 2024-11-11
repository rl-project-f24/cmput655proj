# Notes for installing environments:

## For mujoco 

```
poetry install -E mujoco
```

### To remove

```
poetry remove mujoco
```


## Running
python cleanrl/evals/ddpg_eval.py

python cleanrl/ddpg_continuous_action.py --env-id HalfCheetah-v4 --learning-starts 100 --batch-size 32 --total-timesteps 10000 --capture_video

python cleanrl/ddpg_continuous_action.py --env-id Hopper-v4 --learning-starts 100 --batch-size 32 --total-timesteps 105 
python cleanrl/ppo_continuous_action.py --env-id Hopper-v4 --num-envs 1 --num-steps 64 --total-timesteps 128 


python cleanrl/ddpg_continuous_action.py \
    --seed 1 \
    --env-id Hopper-v4 \
    --total-timesteps 50000 \
    --capture_video


### Benchmarking mujoco
https://docs.cleanrl.dev/rl-algorithms/ddpg/?h=mujoco#explanation-of-the-logged-metrics_1

