import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

with open('_schwefelFB.json', 'r') as file:
    data = json.load(file)

x1_values = []
x2_values = []
fpf_values = []

for run in data:
    for individual in run['individualsWithFPF']:
        x1_values.append(individual['x1'])
        x2_values.append(individual['x2'])
        fpf_values.append(individual['fpfValue'])

grid_size = 20
x1_min, x1_max = min(x1_values), max(x1_values)
x2_min, x2_max = min(x2_values), max(x2_values)

x1_bins = np.arange(x1_min, x1_max, grid_size)
x2_bins = np.arange(x2_min, x2_max, grid_size)
x1_grid, x2_grid = np.meshgrid(x1_bins, x2_bins)
z_grid = np.zeros_like(x1_grid)
counts = np.zeros_like(x1_grid)

for x1, x2, fpf in zip(x1_values, x2_values, fpf_values):
    i = np.digitize(x1, x1_bins) - 1
    j = np.digitize(x2, x2_bins) - 1
    if 0 <= i < len(x1_bins)-1 and 0 <= j < len(x2_bins)-1:
        z_grid[j, i] += fpf
        counts[j, i] += 1

z_grid[counts > 0] /= counts[counts > 0]
z_grid[counts == 0] = np.nan

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(x1_bins) - 1):
    for j in range(len(x2_bins) - 1):
        if not np.isnan(z_grid[j, i]):
            ax.bar3d(x1_bins[i], x2_bins[j], 0, grid_size, grid_size, z_grid[j, i], shade=True, color='blue', alpha=0.6)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Average fpfValue')
ax.set_title('3D Landscape with Flat Platforms of Averaged FPF Values')
plt.show()

fig_plotly = go.Figure()
for i in range(len(x1_bins) - 1):
    for j in range(len(x2_bins) - 1):
        if not np.isnan(z_grid[j, i]):
            fig_plotly.add_trace(go.Bar3d(
                x=[x1_bins[i]],
                y=[x2_bins[j]],
                z=[0],
                width=[grid_size],
                depth=[grid_size],
                height=[z_grid[j, i]],
                marker=dict(color=z_grid[j, i], colorscale='Viridis')
            ))
fig_plotly.update_layout(
    title='Interactive 3D Landscape with Flat Platforms of Averaged FPF Values',
    scene=dict(
        xaxis_title='x1',
        yaxis_title='x2',
        zaxis_title='Average fpfValue'
    )
)
fig_plotly.show()