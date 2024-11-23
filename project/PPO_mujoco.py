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

from RL_classes import *

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
    clip_coef: float = 0.02
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


def expected_return(agent, env_fn, device, num_episodes=10, gamma=0.99):
    returns = []
    for _ in range(num_episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        total_return = 0.0
        # discount = 1.0
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
            # discount *= gamma
            done = terminated or truncated
        returns.append(total_return)
        env.close()
    return returns  # Return list of returns for each episode


def smooth(data, span):
    return np.convolve(data, np.ones(span) / span, mode='valid')


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

    # Define corruption percentages
    corruption_percentages = [0, 10, 20, 30]

    # Initialize dictionaries to store results
    expected_returns_all = {}
    steps_all = {}
    agents_eval = {}

    # # Initialize step counters and expected return histories for agents
    # step_counter = {'Predicted': 0, 'Actual': 0}
    # expected_returns = {'Predicted': [], 'Actual': []}
    # steps = {'Predicted': [], 'Actual': []}

    segment_length = 50  # or any fixed length you prefer

    for cp in corruption_percentages:
        print(f"\n=== Processing Corruption Percentage: {cp}% ===\n")
        args.corruption_percentage = cp

        # Initialize Reward Predictor and Trainer
        reward_predictor = RewardPredictorNetwork(envs.single_observation_space)
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
        start_time = time.time()

        # Reset step counters and expected returns for agents
        if cp == 0:
            step_counter = {'Predicted': 0, 'Actual': 0}
            expected_returns = {'Predicted': [], 'Actual': []}
            steps = {'Predicted': [], 'Actual': []}
        else:
            step_counter = {'Predicted': 0}
            expected_returns = {'Predicted': []}
            steps = {'Predicted': []}

        for d in range(args.D):
            print(f"Outer iteration {d+1}/{args.D}")

            # Collect trajectories
            use_random_policy = (d == 0)  # Use random policy in the first iteration
            collector = TrajectoryCollector(
                env_fn,
                agent=agent_predicted if not use_random_policy else None,
                num_steps=segment_length,
                device=device,
                use_random_policy=use_random_policy,
            )
            trajectories = collector.collect_trajectories(args.num_trajectories)
            segments = {k: v[0] for k, v in trajectories.items()}  # Only store states

            # Generate preferences
            preferences = reward_trainer.generate_preferences(trajectories, args.num_preferences)

            # Create dataset and dataloader
            dataset = PreferenceDataset(segments, preferences, segment_length)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Train reward predictor
            print("Training the Reward Predictor...")
            reward_trainer.train_on_dataloader(dataloader, n_epochs=args.reward_training_epochs)

            # agent_end_of_d_minus_one = copy.deepcopy(agent)
            # agent_predicted = agent_end_of_d_minus_one
            # optimizer_predicted = optim.Adam(agent_predicted.parameters(), lr=args.learning_rate, eps=1e-5)
            # if d == args.D - 1:
            if cp == 0:

                
                # agent_actual = copy.deepcopy(agent_end_of_d_minus_one)
                # optimizer_actual = optim.Adam(agent_actual.parameters(), lr=args.learning_rate, eps=1e-5)
                agents  = [('Predicted', agent_predicted, optimizer_predicted), ('Actual', agent_actual, optimizer_actual)]
            else:
                agents = [('Predicted', agent_predicted, optimizer_predicted)]


                # Reset global_step and start_time
                global_step = 0
                start_time = time.time()


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

                        if "final_info" in infos:
                            for info in infos["final_info"]:
                                if info and "episode" in info:
                                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                                    writer.add_scalar(f"charts/{agent_type}_cp{cp}_episodic_return", info["episode"]["r"], global_step)
                                    writer.add_scalar(f"charts/{agent_type}_cp{cp}_episodic_length", info["episode"]["l"], global_step)

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

                    # TRY NOT TO MODIFY: record losses for plotting purposes
                    writer.add_scalar(f"charts/{agent_type}_cp{cp}_learning_rate", optimizer_instance.param_groups[0]["lr"], global_step)
                    writer.add_scalar(f"losses/{agent_type}_cp{cp}_value_loss", v_loss.item(), global_step)
                    writer.add_scalar(f"losses/{agent_type}_cp{cp}_policy_loss", pg_loss.item(), global_step)
                    writer.add_scalar(f"losses/{agent_type}_cp{cp}_entropy", entropy_loss.item(), global_step)
                    writer.add_scalar(f"losses/{agent_type}_cp{cp}_old_approx_kl", old_approx_kl.item(), global_step)
                    writer.add_scalar(f"losses/{agent_type}_cp{cp}_approx_kl", approx_kl.item(), global_step)
                    writer.add_scalar(f"losses/{agent_type}_cp{cp}_clipfrac", np.mean(clipfracs), global_step)
                    writer.add_scalar(f"losses/{agent_type}_cp{cp}_explained_variance", explained_var, global_step)
                    print(f"Iteration {iteration}/{args.num_iterations_per_outer_loop} - {agent_type} Agent - SPS: {int(global_step / (time.time() - start_time))}")
                    writer.add_scalar(f"charts/{agent_type}_cp{cp}_SPS", int(global_step / (time.time() - start_time)), global_step)

                    # Only track expected return during the last iteration (d = D - 1)
                    # if d == args.D - 1:
                    avg_return = np.mean(expected_return(agent_instance, env_fn, device, num_episodes=10, gamma=args.gamma))
                    expected_returns[agent_type].append(avg_return)
                    steps[agent_type].append(step_counter[agent_type])
                    print(f"Expected Return ({agent_type}, cp={cp}%): {avg_return}")
                    writer.add_scalar(f"charts/{agent_type}_cp{cp}_expected_return", avg_return, step_counter[agent_type])

        # Store results for plotting
        for agent_type in expected_returns.keys():
            key = f"{agent_type} cp={cp}%"
            expected_returns_all[key] = expected_returns[agent_type]
            steps_all[key] = steps[agent_type]
            # Store agent for evaluation
            if agent_type == 'Predicted':
                agents_eval[key] = copy.deepcopy(agent_predicted)
            elif agent_type == 'Actual' and cp == 0:
                agents_eval[key] = copy.deepcopy(agent_actual)

    # Plotting the expected return comparison
    plt.figure()
    for agent_type in expected_returns_all:
        plt.plot(steps_all[agent_type], expected_returns_all[agent_type], label=agent_type)
    plt.xlabel('Steps')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.title('Expected Return Comparison During Training')
    plt.savefig(f"runs/{run_name}/expected_return_comparison.png")
    plt.show()

    # Evaluate all agents
    evaluation_returns = {}
    for agent_type in agents_eval:
        agent_instance = agents_eval[agent_type]
        returns_eval = expected_return(agent_instance, env_fn, device, num_episodes=10, gamma=args.gamma)
        evaluation_returns[agent_type] = returns_eval

    # Plot evaluation results
    plt.figure()
    agent_names = list(evaluation_returns.keys())
    mean_returns = [np.mean(evaluation_returns[agent]) for agent in agent_names]
    std_returns = [np.std(evaluation_returns[agent]) for agent in agent_names]
    plt.bar(agent_names, mean_returns, yerr=std_returns)
    plt.ylabel('Average Expected Return')
    plt.title('Performance Comparison Evaluation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"runs/{run_name}/performance_comparison_evaluation.png")
    plt.show()

    if args.save_model and 'Actual cp=0%' in agents_eval:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agents_eval['Actual cp=0%'].state_dict(), model_path)
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
