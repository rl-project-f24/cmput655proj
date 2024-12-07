# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import time
from dataclasses import dataclass

import gymnasium as gym
import torch
import tyro

from project.evaluate_result_sac import evaluate_in_process, evaluate_result
from project.SAC_mujoco import Actor


@dataclass
class Args:
    seed: int = 1
    """Seed of the experiment."""
    cuda: bool = True
    """If toggled, cuda will be enabled by default."""
    env_id: str = "Hopper-v4"
    """The environment ID of the task."""
    device: str = ""
    """Device to be used for training."""
    path_to_load: str = "models/seed 1/cp 0/agent_type actual/step 3000"
    """Path to the directory containing the model checkpoint."""


def load_model_weights(actor, directory):
    """Load the most recent actor model weights from the specified directory."""
    # Find all `actor_*.pth` files in the directory
    actor_files = [f for f in os.listdir(directory) if f.startswith("actor_") and f.endswith(".pth")]
    if not actor_files:
        raise FileNotFoundError(f"No actor_*.pth files found in {directory}. Make sure the directory contains valid checkpoints.")

    # Sort by step number extracted from filenames
    actor_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

    # Load the latest actor checkpoint
    latest_actor_file = actor_files[-1]
    actor_path = os.path.join(directory, latest_actor_file)
    actor.load_state_dict(torch.load(actor_path))
    print(f"Actor weights loaded from {actor_path}")


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    # Parse arguments
    args = tyro.cli(Args)
    run_name = f"{args.env_id}_evaluation_{int(time.time())}"

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.device:
        device = torch.device(args.device)

    # Set up the environment
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)])

    # Initialize the actor model
    actor = Actor(envs).to(device)

    # Load weights for the actor model
    try:
        load_model_weights(actor, directory=args.path_to_load)
    except FileNotFoundError as e:
        print(str(e))
        exit(1)

    # Run evaluation
    log_string = f"eval_{os.path.basename(args.path_to_load)}"
    # evaluate_in_process("actor", actor, run_name, device, args, log_string, num_episodes_to_record=5)
    evaluate_result("actor", actor, run_name, device, args, log_string, num_episodes_to_record=5)
