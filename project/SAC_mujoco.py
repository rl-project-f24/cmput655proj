# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.data import Dataset, DataLoader
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from project.evaluate_result_sac import evaluate_in_process, evaluate_result
from project.save_load_weights import save_actor_model_weights, save_model_weights


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 9000
    """total timesteps of the experiments, per D iteration"""
    buffer_size: int = int(5e5)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 3000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    eval_frequency: int = 500
    """the frequency of evaluating the policy (delayed)"""


    # Preferences specific arguments
    # reward_min: int = -10
    # """the min clip of environment reward funciton"""
    # reward_max: int = 10
    # """the max clip of environment reward funciton"""
    total_timesteps_per_iteration: int = 10000
    """total timesteps per outer loop iteration"""
    D: int = 4
    """number of outer loop iterations"""
    # Reward predictor specific arguments
    reward_learning_rate: float = 3e-5
    """learning rate for the reward predictor"""
    num_trajectories: int = 300
    """number of trajectories to collect for reward predictor training"""
    reward_training_epochs: int = 9
    """number of epochs to train the reward predictor"""
    run_evaluation: bool = False
    """if toggled, will run the result evaluation including storing video"""
    device: str = "" 
    """Device to be used for training"""
    save_model_weights_at_eval: bool = False
    """Whether to save the model weights to disk at every evaluation step"""
    

    
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


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Reward Predictor Network
class RewardPredictorNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
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
        # self.reward_min = reward_min
        # self.reward_max = reward_max

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
        outputs = self.network(x)  # [batch_size * sequence_length, 1]
        # outputs_original = self.network(x)  # [batch_size * sequence_length, 1]
        # outputs = self.sigmoid(outputs_original)  # Apply sigmoid
        
        # scale the outputs to the desired reward range
        # outputs = outputs * (self.reward_max - self.reward_min) + self.reward_min

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

    def collect_trajectory(self):
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
                    action, _, _ = self.agent.get_action(obs_tensor)
                action = action.cpu().numpy()[0]
            actions0.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
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
                    action, _, _ = self.agent.get_action(obs_tensor)
                action = action.cpu().numpy()[0]
            actions1.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            states1.append(obs)
            total_reward1 += reward
            if terminated or truncated:
                break
            if not self.use_random_policy:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)

        env.close()
        return np.array(states0), total_reward0, np.array(actions0), np.array(states1), total_reward1, np.array(actions1)

    def collect_trajectories(self, n_trajectories):
        trajectories0 = {}
        trajectories1 = {}
        for i in range(n_trajectories):
            states0, reward0, actions0, states1, reward1, actions1 = self.collect_trajectory()
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

    def train_on_dataloader(self, dataloader, n_epochs=1):
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

                # Calculate accuracy
                with torch.no_grad():
                    predictions = logits.argmax(dim=1)
                    targets = prefs.argmax(dim=1)
                    accuracy = (predictions == targets).float().mean()

                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy.item())

            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            if epoch == n_epochs-1:
                print(f"Reward Predictor Training - Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
                reward_accuracy = avg_accuracy
        return reward_accuracy



def expected_return(actor, env_fn, device, seed, num_episodes=10, gamma=0.99):
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
                action, _, _ = actor.get_action(obs_tensor)
            action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
            # discount *= gamma
            done = terminated or truncated
        returns.append(total_return)
        env.close()
    return returns  # Return list of returns for each episode


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    eval_flag = args.run_evaluation
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # if args.env_id == "InvertedPendulum-v4":
    #     args.reward_min = 0
    #     args.reward_max = 1

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.device != "":
        device = torch.device(args.device)




    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    env_fn = lambda: make_env(args.env_id, args.seed, 0, args.capture_video, run_name)()
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # bookeeping for plots
    labels = []
    episode_returns_all = []
    steps_all = []

    # ACTUAL
    episode_returns = []
    steps = []

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    
    step_count = 0
    max_steps = (args.total_timesteps-2*args.eval_frequency) * args.D + 1
    for global_step in range(max_steps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step >= args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # evaluate current policy
            if global_step % args.eval_frequency == 0:
                avg_return = np.mean(expected_return(actor, env_fn, device, np.random.randint(0, 2**16 - 1), gamma=args.gamma))
                episode_returns.append(avg_return)
                steps.append(step_count)
                step_count += args.eval_frequency
                log_string = f"seed {args.seed}/cp 0/agent_type actual/step {step_count}"
                if eval_flag:
                    evaluate_in_process("SAC", actor, run_name, torch.device("cpu"), args, log_string)
                if args.save_model_weights_at_eval and max_steps - global_step <= args.eval_frequency * 4:
                    save_actor_model_weights(actor, qf1, qf2, qf1_target, qf2_target, directory=f"models/{log_string}", step=global_step)

                print(f"Step: {global_step} | Expected Return: {avg_return}")

    envs.close()
    episode_returns_all.append(episode_returns)
    steps_all.append(steps)
    labels.append('SAC with task reward')

    plt.figure()
    plt.plot(steps, episode_returns, label='SAC with task reward', linestyle='--')
    # plt.fill_between(steps_all['Predicted cp=0%'], mean_values - std_values, mean_values + std_values, alpha=0.15)

    # PREFERENCES
    segment_length = 50  # or any fixed length you prefer

    # Define corruption percentages
    corruption_percentages = [0, 20]
    for cp in corruption_percentages:
        print(f"Processing Corruption Percentage: {cp}%")
        # bookeeping for plots
        episode_returns = []
        steps = []
        # env setup
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        max_action = float(envs.single_action_space.high[0])

        actor = Actor(envs).to(device)
        qf1 = SoftQNetwork(envs).to(device)
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

        # Automatic entropy tuning
        if args.autotune:
            target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        else:
            alpha = args.alpha

        envs.single_observation_space.dtype = np.float32
        
        reward_predictor = RewardPredictorNetwork(envs.single_observation_space, envs.single_action_space)
        reward_trainer = RewardTrainer(
            reward_predictor,
            device=device,
            lr=args.reward_learning_rate,
            corruption_percentage=cp,
        )
        step_count = 0
        for d in range(args.D):
            print(f"Outer iteration {d}/{args.D}")
            # Recollect data
            rb = ReplayBuffer(
                args.buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                device,
                handle_timeout_termination=False,
            )

            # Collect trajectories
            use_random_policy = (d == 0)  # Use random policy in the first iteration
            collector = TrajectoryCollector(
                lambda: env_fn(),
                agent=actor if not use_random_policy else None,
                num_steps=segment_length,
                device=device,
                use_random_policy=use_random_policy,
            )
            trajectories0, trajectories1 = collector.collect_trajectories(args.num_trajectories)
            segments0 = {k: [v[0], v[1]] for k, v in trajectories0.items()}  # Only store states and actions
            segments1 = {k: [v[0], v[1]] for k, v in trajectories1.items()}  # Only store states and actions

            # Generate preferences
            preferences = reward_trainer.generate_preferences(trajectories0, trajectories1, args.num_trajectories)

            # Create dataset and dataloader
            dataset = PreferenceDataset(segments0, segments1, preferences, segment_length)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            # Train reward predictor
            # print("Training the Reward Predictor...")
            reward_accuracy = reward_trainer.train_on_dataloader(dataloader, n_epochs=args.reward_training_epochs)

            obs, _ = envs.reset(seed=args.seed)
            for global_step in range(args.total_timesteps+1):
                # ALGO LOGIC: put action logic here
                if global_step < args.learning_starts:
                    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                else:
                    actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                    actions = actions.detach().cpu().numpy()

                # TRY NOT TO MODIFY: execute the game and log data.
                prev_obs = torch.Tensor(obs).to(device)
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                with torch.no_grad():
                    actions_tensor = torch.Tensor(actions).to(device)
                    predicted_reward = reward_predictor(prev_obs.unsqueeze(1), actions_tensor.unsqueeze(1)).squeeze(-1).squeeze(-1)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                # if "final_info" in infos:
                #     for info in infos["final_info"]:
                #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                #         break

                # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
                real_next_obs = next_obs.copy()
                for idx, trunc in enumerate(truncations):
                    if trunc:
                        real_next_obs[idx] = infos["final_observation"][idx]
                rb.add(obs, real_next_obs, actions, predicted_reward, terminations, infos)

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

                # ALGO LOGIC: training.
                if global_step >= args.learning_starts:
                    data = rb.sample(args.batch_size)
                    with torch.no_grad():
                        next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                    qf1_a_values = qf1(data.observations, data.actions).view(-1)
                    qf2_a_values = qf2(data.observations, data.actions).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    # optimize the model
                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                        for _ in range(
                            args.policy_frequency
                        ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                            pi, log_pi, _ = actor.get_action(data.observations)
                            qf1_pi = qf1(data.observations, pi)
                            qf2_pi = qf2(data.observations, pi)
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)
                            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            actor_optimizer.step()

                            if args.autotune:
                                with torch.no_grad():
                                    _, log_pi, _ = actor.get_action(data.observations)
                                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                                a_optimizer.zero_grad()
                                alpha_loss.backward()
                                a_optimizer.step()
                                alpha = log_alpha.exp().item()

                    # update the target networks
                    if global_step % args.target_network_frequency == 0:
                        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    
                    # evaluate current policy
                    if global_step % args.eval_frequency == 0:
                        avg_return = np.mean(expected_return(actor, env_fn, device, np.random.randint(0, 2**16 - 1), gamma=args.gamma))
                        episode_returns.append(avg_return)
                        steps.append(step_count)
                        step_count += args.eval_frequency
                        log_string = f"seed {args.seed}/cp {cp}/agent_type preferences/step {step_count}"
                        if eval_flag:
                            evaluate_in_process("SAC", actor, run_name, torch.device("cpu"), args, log_string)
                        if args.save_model_weights_at_eval and args.total_timesteps - global_step <= args.eval_frequency * 4:
                            save_actor_model_weights(actor, qf1, qf2, qf1_target, qf2_target, directory=f"models/{run_name}/{log_string}", step=global_step)
                        print(f"Step: {global_step} | Expected Return: {avg_return}")
        
        envs.close()
        episode_returns_all.append(episode_returns)
        steps_all.append(steps)
        label = r'$\epsilon$ = ' + str(cp/100)
        labels.append(label)
        plt.plot(steps, episode_returns, label=label)

    plt.grid(True, color='gray', alpha=0.3)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Return')
    plt.legend()
    plt.title(f'SAC on preference data with errors, in {args.env_id}')
    plt.savefig(f"runs/{run_name}/expected_return_comparison_{args.env_id}.png")
    plt.show()

    writer.close()


