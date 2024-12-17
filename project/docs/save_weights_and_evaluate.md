# To save weights and evaluate videos from the weights:
PPO_mujoco.py --env-id InvertedPendulum-v4 --batch-size 32 --total-timesteps_per_iteration 100000\  --D 4 --num_seeds 10 --run-evaluation --device cpu --use-multithreading --num-processes 3 --save_model_weights_at_eval


## Flags:
```
# (to run in real time)
--run-evaluation 
# (to save for later use:)
--save_model_weights_at_eval
```

Then, find the model path that --save_model_weights_at_eval has saved to




## FOR EACH SETUP

### MUJOCO:

```
PPO_mujoco.py --env-id InvertedPendulum-v4 --batch-size 32 --total-timesteps_per_iteration 100000\  --D 4 --num_seeds 10 --run-evaluation --device cpu --use-multithreading --num-processes 3 --save_model_weights_at_eval

PPO_mujoco_saved_weight_evaluation.py --env-id InvertedPendulum-v4 --device cpu --path-to-load ../../project/models/InvertedPendulum-v4__PPO_mujoco__1733614749/seed\ 0/cp\ 0/agent_type\ Actual/step\ 8192,\ PPO\ training\ iteration\ 4

```

### SAC:
```
SAC_mujoco.py --env-id HalfCheetah-v4 --batch-size 32 --total-timesteps_per_iteration 10000 --learning_starts 4000 --eval_frequency 1000 --run-evaluation --save-model-weights-at-eval 

SAC_saved_weight_evaluation.py --env-id HalfCheetah-v4 --device cpu --path-to-load ../../models/seed\ 1/cp\ 0/agent_type\ actual/step\ 3000 
```

### Discrete ppo:
```
PPO_discrete.py --env-id CartPole-v1 --batch-size 32 --total-timesteps_per_iteration 100000 --D 3 --num_seeds 10 --run-evaluation --save-model-weights-at-eval 

PPO_discrete_saved_weight_evaluation.py --env-id CartPole-v1 --device cpu --path-to-load ../../project/models/CartPole-v1__PPO_discrete__1733614858/seed\ 0/cp\ 0/agent_type\ Actual/step\ 20480,\ PPO\ training\ iteration\ 10
```



