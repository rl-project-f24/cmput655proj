import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import copy
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        hidden_size = 256
        activation_fn = nn.ReLU
        super().__init__()
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
            activation_fn(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            activation_fn(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
            activation_fn(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            activation_fn(),
            layer_init(nn.Linear(hidden_size, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# Reward Predictor Network
class RewardPredictorNetwork(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        input_dim = np.prod(observation_space.shape)
        self.network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1)),
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, observation_dim]
        batch_size, sequence_length, obs_dim = x.shape
        x = x.view(-1, obs_dim)  # Flatten to [batch_size * sequence_length, obs_dim]
        outputs = self.network(x)  # [batch_size * sequence_length, 1]
        outputs = outputs.view(batch_size, sequence_length, -1)  # Reshape back
        return outputs  # Shape: [batch_size, sequence_length, 1]


# Dataset for preference comparisons
class PreferenceDataset(Dataset):
    def __init__(self, segments0, segments1, preferences, segment_length):
        self.segments0 = segments0
        self.segments1 = segments1
        self.preferences = preferences
        self.segment_length = segment_length

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        seg1_id, seg2_id, pref = self.preferences[idx]
        seg1 = self.pad_or_truncate(self.segments0[seg1_id])
        seg2 = self.pad_or_truncate(self.segments1[seg2_id])
        return (
            torch.FloatTensor(seg1),
            torch.FloatTensor(seg2),
            torch.FloatTensor(pref),
        )

    def pad_or_truncate(self, segment):
        length = len(segment)
        if length < self.segment_length:
            # Pad with zeros
            pad_size = self.segment_length - length
            padding = np.zeros((pad_size, *segment.shape[1:]))
            segment = np.concatenate([segment, padding], axis=0)
        else:
            # Truncate to the fixed length
            segment = segment[:self.segment_length]
        return segment


# Trajectory Collector
class TrajectoryCollector:
    def __init__(self, env_fn, agent=None, num_steps=50, device='cpu', use_random_policy=False):
        self.env_fn = env_fn
        self.agent = agent
        self.num_steps = num_steps
        self.device = device
        self.use_random_policy = use_random_policy

    def collect_trajectory(self):
        # collect 2 trajectories with the same start
        env = self.env_fn()
        states0, states1 = [], []
        

        obs, _ = env.reset() # start state, shared by both trajectories
        obs1 = copy.deepcopy(obs)

        # trajectory 0
        total_reward0 = 0
        states0.append(obs)
        if not self.use_random_policy:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)

        for _ in range(self.num_steps):
            if self.use_random_policy:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            states0.append(obs)
            total_reward0 += reward
            if terminated or truncated:
                break
            if not self.use_random_policy:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)

        total_reward1 = 0
        # trajectory 1
        obs = copy.deepcopy(obs1)
        states1.append(obs)
        if not self.use_random_policy:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)

        for _ in range(self.num_steps):
            if self.use_random_policy:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            states1.append(obs)
            total_reward1 += reward
            if terminated or truncated:
                break
            if not self.use_random_policy:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)

        env.close()
        return np.array(states0), total_reward0, np.array(states1), total_reward1

    def collect_trajectories(self, n_trajectories):
        trajectories0 = {}
        trajectories1 = {}
        for i in range(n_trajectories):
            states0, reward0, states1, reward1 = self.collect_trajectory()
            trajectories0[i] = (states0, reward0)
            trajectories1[i] = (states1, reward1)
        return trajectories0, trajectories1


# Reward Trainer
class RewardTrainer:
    def __init__(self, predictor, device="cpu", lr=1e-4, corruption_percentage=0.0):
        self.predictor = predictor.to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.device = device
        self.corruption_percentage = corruption_percentage / 100.0  # Convert percentage to probability

    def generate_preferences(self, trajectories0, trajectories1, n_preferences):
        preferences = []
        traj_ids = list(trajectories0.keys())

        for _ in range(n_preferences):
            i = np.random.choice(traj_ids, size=1, replace=True)[0]
            reward_i = trajectories0[i][1]
            reward_j = trajectories1[i][1]

            if reward_i > reward_j:
                pref = [1.0, -1.0]
            elif reward_j > reward_i:
                pref = [-1.0, 1.0]
            else:
                pref = [0, 0]

            # Apply corruption
            if np.random.rand() < self.corruption_percentage:
                if np.random.rand() < 0.5:
                    pref = [pref[1], pref[0]]  # Swap preferences
                else:
                    pref = [pref[0], pref[1]]

            preferences.append((i, i, pref))

        return preferences

    def train_on_dataloader(self, dataloader, n_epochs=1):
        for epoch in range(n_epochs):
            epoch_losses = []
            epoch_accuracies = []
            for s1, s2, prefs in dataloader:
                s1 = s1.to(self.device)
                s2 = s2.to(self.device)
                prefs = prefs.to(self.device)

                # Get predicted rewards
                r1 = self.predictor(s1).squeeze(-1)  # shape: [batch_size, sequence_length]
                r2 = self.predictor(s2).squeeze(-1)

                # Sum over time steps
                r1_sum = r1.sum(dim=1)  # shape: [batch_size]
                r2_sum = r2.sum(dim=1)

                # Stack for softmax
                logits = torch.stack([r1_sum, r2_sum], dim=1)  # shape: [batch_size, 2]

                # Calculate loss
                loss = nn.functional.cross_entropy(logits, prefs.argmax(dim=1))

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy
                with torch.no_grad():
                    predictions = logits.argmax(dim=1)
                    targets = prefs.argmax(dim=1)
                    accuracy = (predictions == targets).float().mean()

                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy.item())

            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            print(f"Reward Predictor Training - Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")