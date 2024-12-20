import os
import random
import threading
import time
import copy
from dataclasses import dataclass
import logging
from tqdm import tqdm

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
import wandb

np.float_ = np.float64

from evaluate_result import evaluate_result
from collections import deque

import multiprocessing
from functools import partial
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
import cProfile
import pstats
import io



def monitor_iterations(counter, interval=10, window_size=10):
    """Continuously logs iterations per second, running average, and last 10-second average."""
    last_count = 0
    total_elapsed_time = 0  # Keep track of total elapsed time
    iteration_history = deque(maxlen=window_size)  # Store iteration counts for the last `window_size` seconds

    while True:
        time.sleep(interval)  # Wait for the specified interval
        current_count = counter["value"]
        iterations_per_second = (current_count - last_count) / interval
        last_count = current_count

        # Update total elapsed time
        total_elapsed_time += interval

        # Update rolling history
        iteration_history.append(iterations_per_second)

        # Calculate 10-second average
        if len(iteration_history) > 0:
            last_10_second_average = sum(iteration_history) / len(iteration_history)
        else:
            last_10_second_average = 0

        # Calculate running average iterations per second
        running_average = current_count / total_elapsed_time

        print(
            f"Iterations per second (instantaneous): {iterations_per_second:.2f}, "
            f"Running average: {running_average:.2f}, "
            f"Last {window_size}-second average: {last_10_second_average:.2f}, "
            f"{total_elapsed_time} seconds have elapsed"
        )
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    use_wandb: bool = False
    """whether to use wandb or not"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    reward_min: int = -10
    """the min clip of environment reward funciton"""
    reward_max: int = 10
    """the max clip of environment reward funciton"""
    num_seeds: int = 1
    """number of seeds to run everythhing"""
    total_timesteps_per_iteration: int = 10000
    """total timesteps per outer loop iteration"""
    D: int = 5
    """number of outer loop iterations"""
    learning_rate: float = 3e-5
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.02
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Reward predictor specific arguments
    reward_learning_rate: float = 1e-4
    """learning rate for the reward predictor"""
    num_trajectories: int = 100
    """number of trajectories to collect for reward predictor training"""
    # num_preferences: int = 1000
    # """number of preference comparisons to generate"""
    reward_training_epochs: int = 9
    """number of epochs to train the reward predictor"""
    corruption_percentage: float = 0.0
    """percentage of preference data to corrupt (e.g., 10.0 for 10%)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations_per_outer_loop: int = 0
    """the number of iterations per outer loop (computed in runtime)"""
    total_timesteps: int = 0
    """the total timesteps (computed in runtime)"""
    
    run_evaluation: bool = False
    """if toggled, will run the result evaluation including storing video"""
    use_multithreading: bool = False
    """Whether to run the seeds in thread pool, or sequentially"""
    num_processes: int = 0
    """number of processes to use in multithreading, if 0 or less, uses total count of cpu cores python can count (all cores)"""
    use_profiler: bool = False
    """Whether to run the profiler and save to local dir after completion"""
    device: str = "" 
    """Device to be used for training"""








def episode_trigger(episode_number):
    return False
    # return episode_number % 10 == 0  # Record every 10th episode

def step_trigger(step_number):
    return False
    # return step_number % 1000 == 0  # Record every 1000 steps   


def make_env(env_id, idx, capture_video, run_name, gamma, reward_min, reward_max):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, reward_min, reward_max))
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        hidden_size = 128
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=0.1),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
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
    def __init__(self, observation_space, action_space, reward_min, reward_max):
        super().__init__()
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n # one-hot encoded action
        else:
            action_dim = np.prod(action_space.shape)
        input_dim = np.prod(observation_space.shape) + action_dim
        hidden_size = 256
        self.network = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1)),
        )
        self.sigmoid = nn.Sigmoid()
        self.action_space = action_space
        self.reward_min = reward_min
        self.reward_max = reward_max

    def forward(self, x, actions):
        # x shape: [batch_size, sequence_length, observation_dim]
        # actions shape: [batch_size, sequence_length]
        batch_size, sequence_length, obs_dim = x.shape

        if isinstance(self.action_space, gym.spaces.Discrete):
            num_actions = self.action_space.n
            actions = actions.long().view(-1)  # Flatten to [batch_size * sequence_length]
            actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=num_actions).float()
            actions_one_hot = actions_one_hot.view(batch_size, sequence_length, -1)
            action_dim = num_actions
        else:
            action_dim = self.action_space.shape[0]
            actions = actions.view(batch_size, sequence_length, action_dim)
            actions_one_hot = actions

        x = torch.cat([x, actions_one_hot], dim=-1)  # Concatenate on the last dimension
        x = x.view(-1, obs_dim + action_dim)  # Flatten to [batch_size * sequence_length, obs_dim + action_dim]
        outputs_original = self.network(x)  # [batch_size * sequence_length, 1]
        # outputs = self.network(x)  # [batch_size * sequence_length, 1]
        outputs = self.sigmoid(outputs_original)  # Apply sigmoid
        
        # scale the outputs to the desired reward range
        outputs = outputs * (self.reward_max - self.reward_min) + self.reward_min

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
        seg1_obs = self.pad_or_truncate(self.segments0[seg1_id][0])
        seg1_actions = self.pad_or_truncate_actions(self.segments0[seg1_id][1])
        seg2_obs = self.pad_or_truncate(self.segments1[seg2_id][0])
        seg2_actions = self.pad_or_truncate_actions(self.segments1[seg2_id][1])
        return (
            torch.FloatTensor(seg1_obs),
            torch.FloatTensor(seg1_actions),
            torch.FloatTensor(seg2_obs),
            torch.FloatTensor(seg2_actions),
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

    def pad_or_truncate_actions(self, actions):
        length = len(actions)
        if length < self.segment_length:
            # Pad with zeros
            pad_size = self.segment_length - length
            if isinstance(actions[0], np.integer):
                padding = np.zeros(pad_size, dtype=np.int64)
            else: 
                padding = np.zeros((pad_size, *actions.shape[1:]))
            actions = np.concatenate([actions, padding], axis=0)
        else:
            # Truncate to the fixed length
            actions = actions[:self.segment_length]
        return actions

# Trajectory Collector
class TrajectoryCollector:
    def __init__(self, env_fn, agent=None, num_steps=50, device='cpu', use_random_policy=False):
        self.env_fn = env_fn
        self.agent = agent
        self.num_steps = num_steps
        self.device = device
        self.use_random_policy = use_random_policy

    def collect_trajectory(self, counter=None):
        # collect 2 trajectories with the same start
        env = self.env_fn()
        states0, states1 = [], []
        actions0, actions1 = [], []
        
        seed = np.random.randint(0, 2**16 - 1)
        
        # trajectory 0
        obs, _ = env.reset(seed=seed) # start state, shared by both trajectories
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
            actions0.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            counter["value"] += 1
            states0.append(obs)
            total_reward0 += reward
            if terminated or truncated:
                break
            if not self.use_random_policy:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)

        # trajectory 1
        obs, _ = env.reset(seed=seed) # start state, shared by both trajectories
        total_reward1 = 0
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
            actions1.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            counter["value"] += 1
            states1.append(obs)
            total_reward1 += reward
            if terminated or truncated:
                break
            if not self.use_random_policy:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)

        env.close()
        return np.array(states0), total_reward0, np.array(actions0), np.array(states1), total_reward1, np.array(actions1)

    def collect_trajectories(self, n_trajectories, counter=None):
        trajectories0 = {}
        trajectories1 = {}
        for i in range(n_trajectories):
            states0, reward0, actions0, states1, reward1, actions1 = self.collect_trajectory(counter=counter)
            trajectories0[i] = (states0, actions0, reward0)
            trajectories1[i] = (states1, actions1, reward1)
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
        # traj_ids = list(trajectories0.keys())

        for i in range(n_preferences):
            # i = np.random.choice(traj_ids, size=1, replace=True)[0]
            reward_i = trajectories0[i][2]
            reward_j = trajectories1[i][2]

            if reward_i - reward_j > 0:
                pref = [1.0, -1.0]
            elif reward_i - reward_j < 0:
                pref = [-1.0, 1.0]
            else:
                pref = [0, 0]

            # Apply corruption
            if np.random.rand() < self.corruption_percentage:
                pref = [pref[1], pref[0]]  # Swap preferences
            # if np.random.rand() < self.corruption_percentage:
            #     if np.random.rand() < 0.5:
            #         pref = [pref[1], pref[0]]  # Swap preferences
            #     else:
            #         pref = [pref[0], pref[1]]

            preferences.append((i, i, pref))

        return preferences

    def train_on_dataloader(self, dataloader, n_epochs=1, counter=None):
        reward_accuracy = 0
        for epoch in range(n_epochs):
            epoch_losses = []
            epoch_accuracies = []
            for s1, a1, s2, a2, prefs in dataloader:
                s1 = s1.to(self.device)
                s2 = s2.to(self.device)
                a1 = a1.to(self.device)
                a2 = a2.to(self.device)
                prefs = prefs.to(self.device)

                # Get predicted rewards
                r1 = self.predictor(s1, a1).squeeze(-1)  # shape: [batch_size, sequence_length]
                r2 = self.predictor(s2, a2).squeeze(-1)

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
                counter["value"] += 1 # good

                # Calculate accuracy
                with torch.no_grad():
                    predictions = logits.argmax(dim=1)
                    targets = prefs.argmax(dim=1)
                    accuracy = (predictions == targets).float().mean()

                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy.item())

            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            # if epoch == n_epochs-1:
            #     print(f"Reward Predictor Training - Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            #     reward_accuracy = avg_accuracy
        return reward_accuracy


def expected_return(agent, env_fn, device, seed, num_episodes=20, gamma=0.99, counter=None):
    returns = []
    for _ in range(num_episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed)
        done = False
        total_return = 0.0
        # discount = 1.0
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            counter["value"] += 1 # good
            total_return += reward
            # discount *= gamma
            done = terminated or truncated
        returns.append(total_return)
        env.close()
    return returns  # Return list of returns for each episode


def smooth(data, span):
    return np.convolve(data, np.ones(span) / span, mode='valid')

def run_subprocess(seed, run_name, args, counter):
    eval_flag = args.run_evaluation and seed == evaluation_seed # only evaluate if the seed is the evaluation one
    start_time = time.time()
    # print(f"SEED: {seed} STARTING!!!!!")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.device != "":
        device = torch.device(args.device)
    # logger.info(f"Using device {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, args.reward_min, args.reward_max) for i in range(args.num_envs)]
    )
    env_fn = lambda capture_video=args.capture_video: make_env(args.env_id, 0, capture_video, run_name, args.gamma, args.reward_min, args.reward_max)()
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    expected_returns_this = {}
    steps_this = {}
    reward_accuracy_this = []

    # # Initialize step counters and expected return histories for agents
    # step_counter = {'Predicted': 0, 'Actual': 0}
    # expected_returns = {'Predicted': [], 'Actual': []}
    # steps = {'Predicted': [], 'Actual': []}

    segment_length = 50  # or any fixed length you prefer

    # Define corruption percentages
    corruption_percentages = [0, 5, 20, 50]

    corruption_bar = tqdm(corruption_percentages, desc=f"Seed {seed} Corruption", unit="%", leave=False, position=1)
    for cp in corruption_bar:
        corruption_bar.set_postfix({"Corruption %": cp})
        reward_accuracy_this_cp = []
        # print(f"Seed: {seed} | Processing Corruption Percentage: {cp}%")
        tqdm.write(f"Seed: {seed} | Processing Corruption Percentage: {cp}%")
        args.corruption_percentage = cp

        # Initialize Reward Predictor and Trainer
        reward_predictor = RewardPredictorNetwork(envs.single_observation_space, envs.single_action_space, args.reward_min, args.reward_max)
        reward_trainer = RewardTrainer(
            reward_predictor,
            device=device,
            lr=args.reward_learning_rate,
            corruption_percentage=args.corruption_percentage,
        )

        # Initialize Agent and Optimizer
        agent_predicted = Agent(envs).to(device)
        optimizer_predicted = optim.Adam(agent_predicted.parameters(), lr=args.learning_rate, eps=1e-5)

        if cp == 0:
            agent_actual = Agent(envs).to(device)
            optimizer_actual = optim.Adam(agent_actual.parameters(), lr=args.learning_rate, eps=1e-5)

        # Global step and start time
        global_step = 0

        # Reset step counters and expected returns for agents
        if cp == 0:
            step_counter = {'Predicted': 0, 'Actual': 0}
            expected_returns = {'Predicted': [], 'Actual': []}
            steps = {'Predicted': [], 'Actual': []}
        else:
            step_counter = {'Predicted': 0}
            expected_returns = {'Predicted': []}
            steps = {'Predicted': []}
            
        # for actual agent
        next_obs, _ = envs.reset(seed=seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        # for d in range(args.D):
        
        # for d in range(args.D):
            # print(f"Seed: {seed} | Outer iteration {d+1}/{args.D}")
        if cp == 0:

            # agent_actual = copy.deepcopy(agent_end_of_d_minus_one)
            # optimizer_actual = optim.Adam(agent_actual.parameters(), lr=args.learning_rate, eps=1e-5)
            # agents  = [('Actual', agent_actual, optimizer_actual)]
            agents  = [('Actual', agent_actual, optimizer_actual), ('Predicted', agent_predicted, optimizer_predicted)]
        else:
            agents = [('Predicted', agent_predicted, optimizer_predicted)]

            
        global_step = 0

        agent_bar =  tqdm(agents, desc="Training Actual/Predicted Agent", unit="agent", leave=False, total=len(agents))
        for agent_type, agent_instance, optimizer_instance in agent_bar:
            agent_bar.set_postfix({"Agent Type": agent_type})

            global_step = 0
            if args.use_wandb:
                assert wandb.run is None
                run = wandb.init(
                    # set the wandb project where this run will be logged
                    project="PPO_mujooco_clockspeed",
                    group=args.exp_name,  # Group all seeds under the same experiment name
                    # track hyperparameters and run metadata
                    config={
                    "architecture": "PPO_discrete",
                    "run_id": int(time.time()),
                    "env_id": args.env_id,
                    "corruption_percent": cp,
                    "agent_type": agent_type,
                    "batch_size": args.batch_size,
                    "D / Reward training steps": args.D,
                    "total-timesteps_per_iteration": args.total_timesteps_per_iteration,
                    "run_name": run_name,
                    "seed": seed,
                    },
                    reinit=True  # Allows reinitialization in the same process
                )
            # print(run.resumed)
            outer_loop_bar = tqdm(range(args.D), desc=f"Seed {seed} Reward Trainer Version (Outer Loop)", unit="iteration", leave=False, position=2)
            for d in outer_loop_bar:
                outer_loop_bar.set_postfix({"Iteration": d + 1})


                # Collect trajectories
                use_random_policy = (d == 0)  # Use random policy in the first iteration
                collector = TrajectoryCollector(
                    lambda: env_fn(),
                    agent=agent_predicted if not use_random_policy else None,
                    num_steps=segment_length,
                    device=device,
                    use_random_policy=use_random_policy,
                )
                trajectories0, trajectories1 = collector.collect_trajectories(args.num_trajectories, counter=counter)
                segments0 = {k: [v[0], v[1]] for k, v in trajectories0.items()}  # Only store states and actions
                segments1 = {k: [v[0], v[1]] for k, v in trajectories1.items()}  # Only store states and actions

                # Generate preferences
                preferences = reward_trainer.generate_preferences(trajectories0, trajectories1, args.num_trajectories)

                # Create dataset and dataloader
                dataset = PreferenceDataset(segments0, segments1, preferences, segment_length)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                # Train reward predictor
                # print(f"Seed: {seed} | Training the Reward Predictor...")
                reward_accuracy = reward_trainer.train_on_dataloader(dataloader, n_epochs=args.reward_training_epochs, counter=counter)
                reward_accuracy_this_cp.append(reward_accuracy)

                # agent_end_of_d_minus_one = copy.deepcopy(agent)
                # agent_predicted = agent_end_of_d_minus_one
                # optimizer_predicted = optim.Adam(agent_predicted.parameters(), lr=args.learning_rate, eps=1e-5)
                # if d == args.D - 1:

                # Initialize PPO storage variables
                obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
                actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
                logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
                rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
                dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
                values = torch.zeros((args.num_steps, args.num_envs)).to(device)

                # Initialize environment
                if agent_type != 'Actual':
                    next_obs, _ = envs.reset(seed=seed)
                    next_obs = torch.Tensor(next_obs).to(device)
                    next_done = torch.zeros(args.num_envs).to(device)

                # PPO Training loop
                # for iteration in range(1, args.num_iterations_per_outer_loop + 1):
                                    # Nested progress bar for PPO iterations
                ppo_bar = tqdm(
                    total=args.num_iterations_per_outer_loop,
                    desc=f"PPO Training Loop for ({agent_type})",
                    unit="iteration",
                    leave=False,
                    position=4
                )
                avg_return = 0
                for iteration in range(1, args.num_iterations_per_outer_loop + 1):
                    ppo_bar.update(1)
                    ppo_bar.set_postfix({"Iteration": iteration, "Last Return": avg_return})
                    total_iterations = args.num_iterations_per_outer_loop * args.D
                    current_iteration = iteration + d * args.num_iterations_per_outer_loop
                    total_iterations = args.num_iterations_per_outer_loop * args.D
                    current_iteration = iteration + d * args.num_iterations_per_outer_loop
                    # Annealing the rate if instructed to do so.
                    if args.anneal_lr:
                    # Calculate total iterations and current iteration
                        frac = 1.0 - (current_iteration - 1.0) / total_iterations
                        lrnow = frac * args.learning_rate
                        optimizer_instance.param_groups[0]["lr"] = lrnow

                    for step in range(0, args.num_steps):
                        # Increment both global and agent-specific step counters
                        global_step += args.num_envs
                        step_counter[agent_type] += args.num_envs
                        obs[step] = next_obs
                        dones[step] = next_done

                        # ALGO LOGIC: action logic
                        with torch.no_grad():
                            action, logprob, _, value = agent_instance.get_action_and_value(next_obs)
                            values[step] = value.flatten()
                        actions[step] = action
                        logprobs[step] = logprob

                        # TRY NOT TO MODIFY: execute the game and log data.
                        prev_obs = next_obs
                        next_obs_np, actual_reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                        counter["value"] += 1 # good
                        next_obs = torch.Tensor(next_obs_np).to(device)
                        next_done = torch.Tensor(np.logical_or(terminations, truncations)).to(device)
                        actual_reward = torch.tensor(actual_reward).to(device).view(-1)

                        # Use Reward Predictor to estimate rewards or use actual rewards
                        if agent_type == 'Actual':
                            rewards[step] = actual_reward
                        else:
                            with torch.no_grad():
                                # next_obs shape: [num_envs, obs_dim]
                                predicted_reward = reward_predictor(prev_obs.unsqueeze(1), action.unsqueeze(1)).squeeze(-1).squeeze(-1)
                            rewards[step] = predicted_reward
                    # bootstrap value if not done
                    with torch.no_grad():
                        next_value = agent_instance.get_value(next_obs).reshape(1, -1)
                        advantages = torch.zeros_like(rewards).to(device)
                        lastgaelam = 0
                        for t in reversed(range(args.num_steps)):
                            if t == args.num_steps - 1:
                                nextnonterminal = 1.0 - next_done
                                nextvalues = next_value
                            else:
                                nextnonterminal = 1.0 - dones[t + 1]
                                nextvalues = values[t + 1]
                            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        returns = advantages + values

                    # flatten the batch
                    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                    b_logprobs = logprobs.reshape(-1)
                    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                    b_advantages = advantages.reshape(-1)
                    b_returns = returns.reshape(-1)
                    b_values = values.reshape(-1)

                    # Optimizing the policy and value network
                    b_inds = np.arange(args.batch_size)
                    clipfracs = []
                    for epoch in range(args.update_epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, args.batch_size, args.minibatch_size):
                            end = start + args.minibatch_size
                            mb_inds = b_inds[start:end]

                            _, newlogprob, entropy, newvalue = agent_instance.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()

                            with torch.no_grad():
                                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                            mb_advantages = b_advantages[mb_inds]
                            if args.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                            # Policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            # Value loss
                            newvalue = newvalue.view(-1)
                            if args.clip_vloss:
                                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                                v_clipped = b_values[mb_inds] + torch.clamp(
                                    newvalue - b_values[mb_inds],
                                    -args.clip_coef,
                                    args.clip_coef,
                                )
                                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                                v_loss = 0.5 * v_loss_max.mean()
                            else:
                                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                            entropy_loss = entropy.mean()
                            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                            optimizer_instance.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(agent_instance.parameters(), args.max_grad_norm)
                            optimizer_instance.step()
                            counter["value"] += 1 # good
                        if args.target_kl is not None and approx_kl > args.target_kl:
                            break

                    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                    avg_return = np.mean(expected_return(agent_instance, env_fn, device, seed, gamma=args.gamma, counter=counter))
                    expected_returns[agent_type].append(avg_return)
                    steps[agent_type].append(step_counter[agent_type])
                    if args.use_wandb:
                        wandb.log({
                            "corruption_percentage": cp,
                            "agent_type": agent_type,
                            "iteration": iteration,
                            "ppo_loss": loss.item(),
                            "policy_gradient_loss": pg_loss.item(),
                            "value_function_loss": v_loss.item(),
                            "entropy_loss": entropy_loss.item(),
                            "clip_fraction": np.mean(clipfracs),
                            "approx_kl": approx_kl.item(),
                            "learning_rate": optimizer_instance.param_groups[0]["lr"],
                            "expected_return": avg_return,
                        }, step=global_step)
                    # print(f"LOGGING STEP {global_step}")
                    # if iteration % 10 == 0:
                    #     print(f"Seed: {seed} | Iteration {iteration}/{args.num_iterations_per_outer_loop} | {agent_type} | cp={cp}% | Expected Return: {avg_return}")
                    log_string = f"seed {seed}/cp {cp}/agent_type {agent_type}/step {step_counter[agent_type]}, PPO training iteration {iteration}"
                    if eval_flag:
                        if iteration == args.num_iterations_per_outer_loop:
                            evaluate_result(agent_type, agent_instance, run_name, device, args, log_string)
                    # Force refresh to fix blank lines
                    # corruption_bar.refresh()
                    # outer_loop_bar.refresh()
                    # agent_bar.refresh()
                    # ppo_bar.refresh()
                        # Finish the current run
            if args.use_wandb:
                wandb.finish()
        
        # Store results for plotting
        reward_accuracy_this.append(reward_accuracy_this_cp)
        for agent_type in expected_returns.keys():
            key = f"{agent_type} cp={cp}%"
            expected_returns_this[key] = [expected_returns[agent_type]]
            steps_this[key] = steps[agent_type]
            # Store agent for evaluation
            # if agent_type == 'Predicted':
            #     agents_eval_this[key] = copy.deepcopy(agent_predicted)
            # elif agent_type == 'Actual' and cp == 0:
            #     agents_eval_this[key] = copy.deepcopy(agent_actual)

    envs.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Seed {seed} DONE!!!!!!!!!!!!!!!!! Execution time: {elapsed_time:.4f} seconds")
    return expected_returns_this, steps_this, reward_accuracy_this

if __name__ == "__main__":


    current_dir = os.path.dirname(os.path.abspath(__file__))
    sibling_folder = os.path.join(current_dir, '..', 'summary_data')
    if not os.path.exists(sibling_folder):
        os.makedirs(sibling_folder)
        logger.info(f"Directory '{sibling_folder}' created.")
    else:
        logger.info(f"Directory '{sibling_folder}' already exists.")
        
    args = tyro.cli(Args)
    args.total_timesteps = args.total_timesteps_per_iteration * args.D
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations_per_outer_loop = args.total_timesteps_per_iteration // args.batch_size or 1
    run_name = f"{args.env_id}__{args.exp_name}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    if args.env_id == "InvertedPendulum-v4":
        args.reward_min = 0
        args.reward_max = 1

    # Initialize dictionaries to store results
    expected_returns_all = {}
    steps_all = {}
    reward_accuracy_all = []
    # agents_eval = {}

    # Use a Manager to create shared state
    manager = multiprocessing.Manager()
    counter = manager.dict()
    counter["value"] = 0  # Initialize the shared counter

    # Launch the monitoring thread
    monitor_thread = threading.Thread(target=monitor_iterations, args=(counter,), daemon=True)
    monitor_thread.start()


    profiler = None
    use_profiler = args.use_profiler
    if use_profiler: 
        profiler = cProfile.Profile()
        profiler.enable()



    num_seeds = args.num_seeds
    seeds = list(range(num_seeds))
    # Define the seed for evaluation
    evaluation_seed = seeds[0]  # Select the first seed for evaluation

    results = []
    if args.use_multithreading: 
        num_processes = 3
        if args.num_processes > 0:
            num_processes = args.num_processes
        logger.info(f"Using {num_processes} threads to complete {num_seeds} seeds")
        run_subprocess_partial = partial(
            run_subprocess,
            run_name=run_name,
            args=args,
            counter=counter  # Pass the managed shared counter
        )

        with tqdm(total=len(seeds), desc="Running seeds", position=0) as seed_bar:
            with multiprocessing.Pool(processes=num_processes) as pool:
                for result in pool.imap_unordered(run_subprocess_partial, seeds):
                    seed_bar.update(1)
                    results.append(result)
    else:
        logger.info(f"Running sequentially with {num_seeds} seeds")
        with tqdm(seeds, desc="Seeds", unit="seed", position=0) as seed_bar:
            for seed in seed_bar:
                result = run_subprocess(seed, run_name, args, counter)
                results.append(result)

    # ! PROFILER HERE
    if use_profiler: 
        profiler.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        profiler.dump_stats(f"./profiles/profile_results_{run_name}.prof")
        with open(f"./profiles/profile_results_{run_name}.txt", "w") as f:
            f.write(s.getvalue())

    baseline_key = 'Actual cp=0%'
    baseline = []
    for expected_returns_this, steps_this, reward_accuracy_this in results:
        for key in expected_returns_this:
            if key == baseline_key:
                baseline.extend(expected_returns_this[key])
                continue
            if key not in expected_returns_all:
                expected_returns_all[key] = []
                steps_all[key] = steps_this[key]
            expected_returns_all[key].extend(expected_returns_this[key])
        reward_accuracy_all.append(reward_accuracy_this)

    # store reward accuracy of last iteration, averaged over all seeds
    arr = np.array(reward_accuracy_all)
    arr = arr.reshape(len(reward_accuracy_all), len(reward_accuracy_all[0]), len(reward_accuracy_all[0][0]))
    last_elements = arr[:, :, -1]
    reward_accuracy_last_iteration = list(np.mean(last_elements, axis=0))

        
    # Plotting the expected return comparison
    custom_labels = {
        'Actual cp=0%': 'PPO with task reward',
        'Predicted cp=0%': r'$\epsilon$ = 0',
        'Predicted cp=5%': r'$\epsilon$ = 0.05',
        'Predicted cp=20%': r'$\epsilon$ = 0.2',
        'Predicted cp=50%': r'$\epsilon$ = 0.5'
    }
    file = open(f'summary_data/{run_name}_{args.env_id}.txt', 'w')
    file.write("training accuracy for corruption percentages of 0, 5, 20, 50\n")
    file.write("\t".join([f"{num:.3f}" for num in reward_accuracy_last_iteration]))
    file.write("\n")
    file.write("error rate\tmean\tsd\n")

    plt.figure()
    
    data_stack = np.vstack(baseline)
    mean_values = np.mean(data_stack, axis=0)
    std_values = np.std(data_stack, axis=0)
    file.write(f"{baseline_key}\t{mean_values[-1]:.3f}\t{std_values[-1]:.3f}\n")
    plt.plot(steps_all['Predicted cp=0%'], mean_values, label=custom_labels[baseline_key], linestyle='--')
    plt.fill_between(steps_all['Predicted cp=0%'], mean_values - std_values, mean_values + std_values, alpha=0.15)

    
    for agent_type in expected_returns_all:
        data_stack = np.vstack(expected_returns_all[agent_type])
        mean_values = np.mean(data_stack, axis=0)
        std_values = np.std(data_stack, axis=0)
        file.write(f"{agent_type}\t{mean_values[-1]:.3f}\t{std_values[-1]:.3f}\n")
        plt.plot(steps_all[agent_type], mean_values, label=custom_labels.get(agent_type, agent_type))
        plt.fill_between(np.asarray(steps_all[agent_type]), mean_values - std_values, mean_values + std_values, alpha=0.15)
    
    file.close()
    
    plt.xlim(left=0, right=args.total_timesteps_per_iteration*args.D)
    if args.env_id == 'InvertedPendulum-v4':
        plt.ylim(bottom=0, top=1000)
    plt.grid(True, color='gray', alpha=0.3)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Return')
    plt.legend()
    plt.title(f'PPO on preference data with errors, in {args.env_id}')
    plt.savefig(f"runs/{run_name}/expected_return_comparison_{args.env_id}.png")
    plt.show()

    # plt.figure()
    # for key in all_advantages:
    #     plt.plot(all_advantages[key], label=key)
    # plt.xlabel('Steps')
    # plt.ylabel('Normalized Advantages')
    # plt.legend()
    # plt.title(f'Advantages Comparison: D={d}, env={args.env_id}, num_seeds={num_seeds}')
    # plt.show()

    
