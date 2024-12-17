import numpy as np
import wandb
import matplotlib.pyplot as plt

# Initialize the wandb API
api = wandb.Api()

# Fetch a specific project and run
project_name = "PPO_discrete_clockspeed"
entity_name = "alfreem1-university-of-alberta"  # Your wandb username or organization
# run_id = "your_run_id"  # The unique ID for your run
desired_run_name = "CartPole-v1__PPO_discrete_clockspeed__1733466920"  # The name of the run you want to filter by


# Fetch all runs from the project
runs = api.runs(f"{entity_name}/{project_name}")

# Filter runs by the `run_name` parameter in config
filtered_runs = [
    run for run in runs if run.config.get("run_name") == desired_run_name
]

# Optional: Check the number of matching runs
print(f"Found {len(filtered_runs)} runs with the run_name '{desired_run_name}'.")

# Prepare data structures for plotting
if not filtered_runs:
    print(f"No runs found with run_name: {desired_run_name}")
    exit()


# Prepare data structures for plotting
if filtered_runs:
    run = filtered_runs[0]  # Take the first run

    # Fetch the full history dataframe
    history = run.history()

    # Print the dataframe columns to find available keys
    print("Available keys in history:")
    print(history.columns)
else:
    print(f"No runs found with run_name: {desired_run_name}")




expected_returns_all = {}
steps_all = {}

# Fetch and process data for plotting
for run in filtered_runs:
    # Retrieve metrics for the run
    history = run.history()
    if history.empty:
        print(f"No history data found for run: {run.id}")
        continue

    # Organize data by corruption percentage and agent type
    for _, row in history.iterrows():
        key = f"{row['agent_type']} cp={int(row['corruption_percentage'])}%"
        if key not in expected_returns_all:
            expected_returns_all[key] = []
            steps_all[key] = []
        expected_returns_all[key].append(row["expected_return"])
        steps_all[key].append(row["_step"])

# Custom labels for the plot
custom_labels = {
    "Actual cp=0%": "PPO with task reward",
    "Predicted cp=0%": r"$\epsilon$ = 0",
    "Predicted cp=5%": r"$\epsilon$ = 0.05",
    "Predicted cp=20%": r"$\epsilon$ = 0.2",
    "Predicted cp=50%": r"$\epsilon$ = 0.5",
}

# Plot the expected return comparison
plt.figure()

for key, returns in expected_returns_all.items():
    steps = np.array(steps_all[key])  # Array of steps for this key
    returns = np.array(returns)      # Convert returns to a numpy array

    # Ensure that steps and returns have matching lengths
    if len(steps) != len(returns):
        print(f"Length mismatch for {key}: Steps ({len(steps)}) vs Returns ({len(returns)})")
        min_length = min(len(steps), len(returns))
        steps = steps[:min_length]
        returns = returns[:min_length]

    # Aggregate data (mean and standard deviation)
    mean_values = np.mean(returns)  # Mean of the returns
    std_values = np.std(returns)    # Standard deviation of the returns

    # Validate shapes before plotting
    print(f"Steps shape: {steps.shape}, Returns shape: {returns.shape}")

    # Plot mean and fill between for standard deviation
    plt.plot(steps, returns, label=custom_labels.get(key, key))
    plt.fill_between(steps, returns - std_values, returns + std_values, alpha=0.15)
    print('test')



# Customize the plot
plt.xlabel("Timesteps")
plt.ylabel("Episode Return")
plt.grid(True, color="gray", alpha=0.3)
plt.title(f"PPO on preference data with errors, in {desired_run_name.split('__')[0]}")
plt.legend()
plt.savefig(f"{desired_run_name}_expected_return_comparison.png")
plt.show()
print('test')