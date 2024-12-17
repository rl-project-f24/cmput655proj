# 655 Project
## How Does Corruption Affect Preference-Based Reinforcement Learning?

Repo for CMPUT 655 (Intro to Reinforcement Learning) Final Project

Authors: Euijin Baek, Andrew Freeman, and Minh Pham



## Installing
details of cleanRL installation: https://docs.cleanrl.dev/get-started/installation/

```
# create a virtual environment
poetry install
poetry install -E mujoco
```

See docs/installnotes.md for more details on installing the project


## Contributions
#### Report
[project/RL1_Final_Paper.pdf](https://github.com/rl-project-f24/cmput655proj/blob/main/project/RL1_Final_Paper.pdf) 

#### The main project files for the report:

[project/PPO_discrete.py](https://github.com/rl-project-f24/cmput655proj/blob/main/project/PPO_discrete.py)

[project/PPO_mujoco.py](https://github.com/rl-project-f24/cmput655proj/blob/main/project/PPO_mujoco.py) 

#### Files for additional data:

[project/SAC_discrete.py](https://github.com/rl-project-f24/cmput655proj/blob/main/project/PPO_discrete.py)

[project/SAC_mujoco.py](https://github.com/rl-project-f24/cmput655proj/blob/main/project/SAC_mujoco.py)

[project/aggregate_plot.py](https://github.com/rl-project-f24/cmput655proj/blob/main/project/aggregate_plot.py) 


## Running

For small sample runs
```
python project/PPO_discrete.py --env-id CartPole-v1 --total-timesteps_per_iteration 20000 --D 2 --num_seeds 1
python project/PPO_mujoco.py --env-id InvertedPendulum-v4 --total-timesteps_per_iteration 20000 --D 2 --num_seeds 1
```

For collected PPO results (main reported results)
```
python project/PPO_discrete.py --env-id CartPole-v1 --batch-size 32 --total-timesteps_per_iteration 100000 --D 4 --num_seeds 10 --use_multithreading
python project/PPO_discrete.py --env-id Acrobot-v1 --batch-size 32 --total-timesteps_per_iteration 100000 --D 4 --num_seeds 10 --use_multithreading
python project/PPO_mujoco.py --env-id InvertedPendulum-v4 --batch-size 32 --total-timesteps_per_iteration 100000 --D 4 --num_seeds 10 --use_multithreading
python project/PPO_mujoco.py --env-id HalfCheetah-v4 --batch-size 32 --total-timesteps_per_iteration 100000 --D 4 --num_seeds 10 --use_multithreading
```

For collected SAC results (appendix)
```
python project/SAC_mujoco.py --env-id InvertedPendulum-v4 --batch-size 32 --total-timesteps 15000 --learning_starts 5000 --D 4 --num_seeds 3
python project/SAC_mujoco.py --env-id HalfCheetah-v4 --batch-size 32 --total-timesteps 15000 --learning_starts 5000 --D 4 --num_seeds 3
```







