import os
import random
import time
import copy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt



def evaluate_result(agent_key, agent_instance, run_name, device, args, log_string):
        # Evaluation step: Record videos and log rewards
    evaluation_run_name = f"{run_name}_evaluation"
    os.makedirs(f"videos/{evaluation_run_name}", exist_ok=True)

    print("\n=== Starting Evaluation ===\n")
    eval_rewards = {}

    print(f"Evaluating {agent_key} agent...")
    eval_rewards[agent_key] = []
    video_folder = f"videos/{evaluation_run_name}/{log_string}"

    # Set up the environment to record videos
    eval_env_fn = lambda: gym.wrappers.RecordVideo(
        gym.make(args.env_id, render_mode="rgb_array"),
        # f"videos/{evaluation_run_name}/{agent_key}",
        video_folder,
        episode_trigger=lambda episode_id: True,  # Record every episode,
        # name_prefix=f""
        disable_logger=True
    )
    
    env = eval_env_fn()
    for episode in range(5):  # Evaluate for 5 playthroughs
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent_instance.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        eval_rewards[agent_key].append(total_reward)
        env.close()

    # Log results for this agent
    avg_reward = np.mean(eval_rewards[agent_key])
    print(f"Evaluation Results - {agent_key}: Average Reward = {avg_reward:.2f}, Rewards = {eval_rewards[agent_key]}")

    # Save evaluation rewards for reference
    # with open(f"runs/{run_name}/evaluation_rewards.txt", "w") as f:
    with open(f"{video_folder}/evaluation_results.txt", "w") as f:
        for agent_key, rewards in eval_rewards.items():
            f.write(f"{agent_key} total episodic rewards: {rewards}\n")
        f.write("\n")
    print(f"Evaluation videos and rewards saved to videos/{evaluation_run_name}/")

