import json
import numpy as np
import matplotlib.pyplot as plt

with open('_schwefelFT.json', 'r') as file:
    data = json.load(file)

# Define grid resolution
tile_size = 50

# Determine the min and max values for x1 and x2
x1_values = [ind['x1'] for entry in data for ind in entry['individualsWithFPF'] if ind['fpfValue'] != 1.0]
x2_values = [ind['x2'] for entry in data for ind in entry['individualsWithFPF'] if ind['fpfValue'] != 1.0]
x1_min, x1_max = min(x1_values), max(x1_values)
x2_min, x2_max = min(x2_values), max(x2_values)

# Create grid
x1_bins = np.arange(x1_min, x1_max + tile_size, tile_size)
x2_bins = np.arange(x2_min, x2_max + tile_size, tile_size)

grid_values = np.full((len(x1_bins), len(x2_bins)), np.nan)
count_values = np.zeros((len(x1_bins), len(x2_bins)))

# Accumulate values into grid
for entry in data:
    for individual in entry['individualsWithFPF']:
        if individual['fpfValue'] != 1.0:
            x1_idx = np.searchsorted(x1_bins, individual['x1'], side='right') - 1
            x2_idx = np.searchsorted(x2_bins, individual['x2'], side='right') - 1

            if 0 <= x1_idx < len(x1_bins) and 0 <= x2_idx < len(x2_bins):
                if np.isnan(grid_values[x1_idx, x2_idx]):
                    grid_values[x1_idx, x2_idx] = 0
                grid_values[x1_idx, x2_idx] += individual['fpfValue']
                count_values[x1_idx, x2_idx] += 1

# Compute the average
with np.errstate(invalid='ignore'):
    grid_values /= count_values

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
c = ax.pcolormesh(x1_bins, x2_bins, grid_values.T, cmap='viridis', shading='auto')
plt.colorbar(c, label='Average fpfValue')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Average FPF Landscape with Tile Size 20x20')

plt.show()