import os
import torch

def save_model_weights(agent, directory="models", step=0):
    """Saves PPO agent model weights."""
    os.makedirs(directory, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(directory, f"ppo_agent_{step}.pth"))
    print(f"Agent model weights saved at step {step} to {directory}")


def load_model_weights(agent, directory="models", step=0):
    """Loads PPO agent model weights."""
    agent_path = os.path.join(directory, f"ppo_agent_{step}.pth")
    if os.path.exists(agent_path):
        agent.load_state_dict(torch.load(agent_path))
        print(f"Agent model weights loaded from {agent_path}")
    else:
        raise FileNotFoundError(f"No weights found at {agent_path}. Make sure the file exists.")
