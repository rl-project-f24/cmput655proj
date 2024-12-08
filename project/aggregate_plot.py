import matplotlib.pyplot as plt
import numpy as np

# Define environments and labels
environments = ['Cart Pole', 'Acrobot', 'Inverted Pendulum', 'Half Cheetah']
labels = [r'$\epsilon$ = 0', r'$\epsilon$ = 0.05', r'$\epsilon$ = 0.2', r'$\epsilon$ = 0.5']
num_envs = len(environments)
num_labels = len(labels)

# Custom values for upward bars (fill in with numbers between 0 and 1)
data = np.array([
    [480.570/419.975, 481.165/419.975, 479.525/419.975, 479.515/419.975],
    [98.860/130.745, 98.860/170.525, 98.860/171.765, 98.860/90.320],
    [435.350/619.655, 319.690/619.655, 426.210/619.655, 275.130/619.655],
    [254.520/403.640, 254.520/524.328, 254.520/498.572, 254.520/471.437]
])

# Custom values for downward bars (fill in with numbers between 0 and 1)
# inverse_data = np.array([
#     [0.526, 0.482, 0.543, 0.534],  # Values for Env1: e.g., [0.4, 0.3, 0.5, 0.6]
#     [0.566, 0.617, 0.571, 0.579],  # Values for Env2: e.g., [0.5, 0.6, 0.7, 0.8]
#     [0.813, 0.798, 0.766, 0.738]   # Values for Env3: e.g., [0.2, 0.4, 0.3, 0.5]
# ])

# Plot grouped bars for the positive graph and inverse graph with symmetric y-axis and labeled sections
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.figure(figsize=(12, 6))

# Plot the original graph (upward bars)
bar_width = 0.17
x = np.arange(num_envs)

for i, label in enumerate(labels):
    plt.bar(x + i * bar_width, data[:, i], width=bar_width, label=label, color=colors[(i+1) % len(colors)])

# Add a horizontal dashed blue line at y=1.0
plt.axhline(y=1.0, color=colors[0], linestyle='--', linewidth=2, label='Task Reward Performance')

# Customize the plot
plt.xlabel('Environments')
plt.ylabel('Relative Performance')
plt.title('Performance across Different Environments and Task Reward Performance Comparison')
plt.xticks(x + bar_width * (num_labels - 1) / 2, environments)
plt.legend(title='Labels')

# Set the y-axis to range from 0 to 1, with ticks every 0.2
# plt.ylim(0, 1)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

# Grid lines for visual clarity
plt.grid(axis='y', alpha=0.4)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()