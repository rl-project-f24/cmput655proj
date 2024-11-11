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
import matplotlib.pyplot as plt


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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps_per_iteration: int = 10000
    """total timesteps per outer loop iteration"""
    D: int = 3
    """number of outer loop iterations"""
    learning_rate: float = 3e-4
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
    ent_coef: float = 0.0
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
    num_trajectories: int = 50
    """number of trajectories to collect for reward predictor training"""
    num_preferences: int = 1000
    """number of preference comparisons to generate"""
    reward_training_epochs: int = 5
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


def make_env(env_id, idx, capture_video, run_name, gamma):
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
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
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
    def __init__(self, segments, preferences, segment_length):
        self.segments = segments
        self.preferences = preferences
        self.segment_length = segment_length

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        seg1_id, seg2_id, pref = self.preferences[idx]
        seg1 = self.pad_or_truncate(self.segments[seg1_id])
        seg2 = self.pad_or_truncate(self.segments[seg2_id])
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
        env = self.env_fn()
        states = []
        total_reward = 0

        obs, _ = env.reset()
        states.append(obs)
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
            states.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
            if not self.use_random_policy:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)

        env.close()
        return np.array(states), total_reward

    def collect_trajectories(self, n_trajectories):
        trajectories = {}
        for i in range(n_trajectories):
            states, reward = self.collect_trajectory()
            trajectories[i] = (states, reward)
        return trajectories


# Reward Trainer
class RewardTrainer:
    def __init__(self, predictor, device="cpu", lr=1e-4, corruption_percentage=0.0):
        self.predictor = predictor.to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.device = device
        self.corruption_percentage = corruption_percentage / 100.0  # Convert percentage to probability

    def generate_preferences(self, trajectories, n_preferences):
        preferences = []
        traj_ids = list(trajectories.keys())

        for _ in range(n_preferences):
            i, j = np.random.choice(traj_ids, size=2, replace=False)
            reward_i = trajectories[i][1]
            reward_j = trajectories[j][1]

            if reward_i > reward_j:
                pref = [1.0, 0.0]
            elif reward_j > reward_i:
                pref = [0.0, 1.0]
            else:
                pref = [0.5, 0.5]

            # Apply corruption
            if np.random.rand() < self.corruption_percentage:
                pref = [pref[1], pref[0]]  # Swap preferences

            preferences.append((i, j, pref))

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


def evaluate_agent(agent, env_fn, device, num_episodes=10):
    returns = []
    for _ in range(num_episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
        env.close()
    return returns


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.total_timesteps = args.total_timesteps_per_iteration * args.D
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations_per_outer_loop = args.total_timesteps_per_iteration // args.batch_size
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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    env_fn = lambda: gym.make(args.env_id)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Initialize Reward Predictor and Trainer
    reward_predictor = RewardPredictorNetwork(envs.single_observation_space)
    reward_trainer = RewardTrainer(
        reward_predictor,
        device=device,
        lr=args.reward_learning_rate,
        corruption_percentage=args.corruption_percentage,
    )

    # Initialize Agent and Optimizer
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Global step and start time
    global_step = 0
    start_time = time.time()
    segment_length = 50  # or any fixed length you prefer

    # Initialize lists to store episodic returns
    episodic_returns_predicted = []
    episodic_returns_actual = []

    for d in range(args.D):
        print(f"Outer iteration {d+1}/{args.D}")

        # Collect trajectories
        use_random_policy = (d == 0)  # Use random policy in the first iteration
        collector = TrajectoryCollector(
            env_fn,
            agent=agent if not use_random_policy else None,
            num_steps=segment_length,
            device=device,
            use_random_policy=use_random_policy,
        )
        trajectories = collector.collect_trajectories(args.num_trajectories)
        segments = {k: v[0] for k, v in trajectories.items()} # Only store states

        # Generate preferences
        preferences = reward_trainer.generate_preferences(trajectories, args.num_preferences)

        # Create dataset and dataloader
        dataset = PreferenceDataset(segments, preferences, segment_length)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Train reward predictor
        print("Training the Reward Predictor...")
        reward_trainer.train_on_dataloader(dataloader, n_epochs=args.reward_training_epochs)

        # Save the agent at the end of iteration D-1
        if d == args.D - 1:
            agent_end_of_d_minus_one = copy.deepcopy(agent)

        # Set up agents and optimizers for the last iteration
        if d == args.D - 1:
            # Create two agents for the last iteration
            agent_predicted = agent_end_of_d_minus_one
            optimizer_predicted = optim.Adam(agent_predicted.parameters(), lr=args.learning_rate, eps=1e-5)
            agent_actual = copy.deepcopy(agent_end_of_d_minus_one)
            optimizer_actual = optim.Adam(agent_actual.parameters(), lr=args.learning_rate, eps=1e-5)
            agents = [('Predicted', agent_predicted, optimizer_predicted), ('Actual', agent_actual, optimizer_actual)]
        else:
            agents = [('Predicted', agent, optimizer)]

        for agent_type, agent_instance, optimizer_instance in agents:
            print(f"Training agent on {agent_type} rewards")
            # Initialize PPO storage variables
            obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
            actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
            logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
            rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
            dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
            values = torch.zeros((args.num_steps, args.num_envs)).to(device)

            # Initialize environment
            next_obs, _ = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)

            # PPO Training loop
            for iteration in range(1, args.num_iterations_per_outer_loop + 1):
                # Annealing the rate if instructed to do so.
                if args.anneal_lr:
                    frac = 1.0 - (global_step - 1.0) / args.total_timesteps
                    lrnow = frac * args.learning_rate
                    optimizer_instance.param_groups[0]["lr"] = lrnow

                for step in range(0, args.num_steps):
                    global_step += args.num_envs
                    obs[step] = next_obs
                    dones[step] = next_done

                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        action, logprob, _, value = agent_instance.get_action_and_value(next_obs)
                        values[step] = value.flatten()
                    actions[step] = action
                    logprobs[step] = logprob

                    # TRY NOT TO MODIFY: execute the game and log data.
                    next_obs_np, actual_reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                    next_obs = torch.Tensor(next_obs_np).to(device)
                    next_done = torch.Tensor(np.logical_or(terminations, truncations)).to(device)
                    actual_reward = torch.Tensor(actual_reward).to(device).view(-1)

                    # Use Reward Predictor to estimate rewards or use actual rewards
                    if agent_type == 'Actual':
                        rewards[step] = actual_reward
                    else:
                        with torch.no_grad():
                            # next_obs shape: [num_envs, obs_dim]
                            predicted_reward = reward_predictor(next_obs.unsqueeze(1)).squeeze(-1).squeeze(-1)
                        rewards[step] = predicted_reward
                        print(f"Predicted Reward: {predicted_reward}")
                        print(f"Actual Reward: {actual_reward}")

                    if "final_info" in infos:
                        for info in infos["final_info"]:
                            if info and "episode" in info:
                                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                                writer.add_scalar(f"charts/{agent_type}_episodic_return", info["episode"]["r"], global_step)
                                writer.add_scalar(f"charts/{agent_type}_episodic_length", info["episode"]["l"], global_step)
                                if agent_type == 'Actual':
                                    episodic_returns_actual.append((global_step, info['episode']['r']))
                                else:
                                    episodic_returns_predicted.append((global_step, info['episode']['r']))

                # PPO Update code remains the same...
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

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar(f"charts/{agent_type}_learning_rate", optimizer_instance.param_groups[0]["lr"], global_step)
                writer.add_scalar(f"losses/{agent_type}_value_loss", v_loss.item(), global_step)
                writer.add_scalar(f"losses/{agent_type}_policy_loss", pg_loss.item(), global_step)
                writer.add_scalar(f"losses/{agent_type}_entropy", entropy_loss.item(), global_step)
                writer.add_scalar(f"losses/{agent_type}_old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar(f"losses/{agent_type}_approx_kl", approx_kl.item(), global_step)
                writer.add_scalar(f"losses/{agent_type}_clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar(f"losses/{agent_type}_explained_variance", explained_var, global_step)
                print(f"Iteration {iteration}/{args.num_iterations_per_outer_loop} - {agent_type} Agent - SPS: {int(global_step / (time.time() - start_time))}")
                writer.add_scalar(f"charts/{agent_type}_SPS", int(global_step / (time.time() - start_time)), global_step)

    # Plotting the performance comparison
    if episodic_returns_predicted:
        global_steps_predicted, returns_predicted = zip(*episodic_returns_predicted)
    else:
        global_steps_predicted, returns_predicted = [], []
    if episodic_returns_actual:
        global_steps_actual, returns_actual = zip(*episodic_returns_actual)
    else:
        global_steps_actual, returns_actual = [], []

    plt.figure()
    plt.plot(global_steps_predicted, returns_predicted, 'b.', label='Predicted Rewards')
    plt.plot(global_steps_actual, returns_actual, 'r.', label='Actual Rewards')
    plt.xlabel('Global Step')
    plt.ylabel('Episodic Return')
    plt.legend()
    plt.title('Performance Comparison During Training')
    plt.savefig(f"runs/{run_name}/performance_comparison_training.png")
    plt.show()

    # Evaluate both agents
    returns_predicted_eval = evaluate_agent(agent_predicted, env_fn, device, num_episodes=10)
    returns_actual_eval = evaluate_agent(agent_actual, env_fn, device, num_episodes=10)

    # Plot evaluation results
    plt.figure()
    plt.bar(['Predicted Rewards', 'Actual Rewards'], [np.mean(returns_predicted_eval), np.mean(returns_actual_eval)],
            yerr=[np.std(returns_predicted_eval), np.std(returns_actual_eval)])
    plt.ylabel('Average Episodic Return')
    plt.title('Performance Comparison Evaluation')
    plt.savefig(f"runs/{run_name}/performance_comparison_evaluation.png")
    plt.show()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent_actual.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
