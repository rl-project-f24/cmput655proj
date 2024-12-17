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
import multiprocessing

import multiprocessing
import os
import signal
import atexit

# Global process list to track all child processes
processes = []

def cleanup():
    """Terminate all child processes."""
    print("Cleaning up processes...")
    for process in processes:
        if process.is_alive():
            print(f"Terminating process {process.pid}")
            process.terminate()
            process.join()

# Register cleanup to run when the program exits
atexit.register(cleanup)



# Function wrapper to call `evaluate_result` in a separate process
def evaluate_in_process(agent_key, agent_instance, run_name, device, args, log_string, num_episodes_to_record = 1):
    process = multiprocessing.Process(
        target=evaluate_result,
        args=(agent_key, agent_instance, run_name, device, args, log_string, num_episodes_to_record)
    )    
    processes.append(process)
    process.start()
    processes.append(process)
    # process.join()  # Wait for the evaluation process to finish, blocks original thread



def evaluate_result(agent_key, agent_instance, run_name, device, args, log_string, num_episodes_to_record = 1):
    # Evaluation step: Record videos and log rewards
    evaluation_run_name = f"{run_name}_evaluation"
    os.makedirs(f"videos/{evaluation_run_name}", exist_ok=True)

    print("\n=== Starting Evaluation ===\n")
    eval_rewards = {}
    print(f"Evaluating {agent_key} agent...")
    eval_rewards[agent_key] = []
    video_folder = f"videos/{evaluation_run_name}/{log_string}"

    for episode in range(num_episodes_to_record):  # Evaluate for 5 playthroughs
        try:
            start_time = time.time()
            env = gym.wrappers.RecordVideo(
                gym.make(args.env_id, render_mode="rgb_array"),
                video_folder,
                episode_trigger=lambda episode_id: True,  # Record every episode,
                disable_logger=True,
                name_prefix = f"rl-video-{episode}",
            )
            obs, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                # Direct tensor conversion for observations
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = agent_instance.get_action(obs_tensor)
                action = action.cpu().numpy()[0]  # Ensure proper formatting for the environment

                # Step in the environment
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

            eval_rewards[agent_key].append(total_reward)
            end_time = time.time()
            print(f"Episode {episode + 1} completed in {end_time - start_time:.2f} seconds with reward {total_reward:.2f}")
        finally:
            env.close()  # Ensure the environment is closed after recording

    # Calculate and log average reward
    avg_reward = np.mean(eval_rewards[agent_key])
    print(f"Evaluation Results - {agent_key}: Average Reward = {avg_reward:.2f}, Rewards = {eval_rewards[agent_key]}")

    # Save evaluation rewards for reference
    with open(f"{video_folder}/evaluation_results.txt", "w") as f:
        for agent, rewards in eval_rewards.items():
            f.write(f"{agent} total episodic rewards: {rewards}\n")
        f.write("\n")
    print(f"Evaluation videos and rewards saved to {os.path.abspath(video_folder)}/")

