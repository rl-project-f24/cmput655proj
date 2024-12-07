# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import time
from dataclasses import dataclass

import gymnasium as gym
import torch
import tyro

from project.evaluate_result import evaluate_result
from project.PPO_discrete import AgentDiscrete, make_env




@dataclass
class Args:
    seed: int = 1
    """Seed of the experiment."""
    num_envs: int = 1
    """the number of parallel game environments"""
    cuda: bool = True
    """If toggled, cuda will be enabled by default."""
    env_id: str = "Hopper-v4"
    """The environment ID of the task."""
    device: str = ""
    """Device to be used for evaluation."""
    path_to_load: str = "models/seed 1/cp 0/agent_type actual/step 3000"
    """Path to the directory containing the model checkpoint."""


def load_latest_weights(agent, directory):
    """Load the most recent PPO agent weights from the specified directory."""
    # Find all `ppo_agent_*.pth` files in the directory
    agent_files = [f for f in os.listdir(directory) if f.startswith("ppo_agent_") and f.endswith(".pth")]
    if not agent_files:
        raise FileNotFoundError(f"No ppo_agent_*.pth files found in {directory}. Make sure the directory contains valid checkpoints.")

    # Sort by step number extracted from filenames
    agent_files.sort(key=lambda f: int(f.split('_')[2].split('.')[0]))

    # Load the latest checkpoint
    latest_agent_file = agent_files[-1]
    agent_path = os.path.join(directory, latest_agent_file)
    agent.load_state_dict(torch.load(agent_path))
    print(f"PPO agent weights loaded from {agent_path}")



if __name__ == "__main__":
    # Parse arguments
    args = tyro.cli(Args)
    run_name = f"{args.env_id}_evaluation_{int(time.time())}"

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.device:
        device = torch.device(args.device)

    # Set up the environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, False, run_name) for i in range(args.num_envs)] # ! Untested num_envs
    )

    # Initialize the PPO agent
    agent = AgentDiscrete(envs).to(device)

    # Load weights for the PPO agent
    try:
        load_latest_weights(agent, directory=args.path_to_load)
    except FileNotFoundError as e:
        print(str(e))
        exit(1)

    # Run evaluation
    log_string = f"eval_{os.path.basename(args.path_to_load)}"
    evaluate_result("agent", agent, run_name, device, args, log_string, num_episodes_to_record=5)