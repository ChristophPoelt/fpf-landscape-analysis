import json
import numpy as np
import matplotlib.pyplot as plt

with open('_h1DB.json', 'r') as file:
    data = json.load(file)

x1 = []
x2 = []
fpfValue = []

for entry in data:
    for individual in entry['individualsWithFPF']:
        x1.append(individual['x1'])
        x2.append(individual['x2'])
        fpfValue.append(individual['fpfValue'])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x1, x2, fpfValue, c=fpfValue, cmap='viridis', marker='o')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('fpfValue')
ax.set_title('H1 Diversity Based; Average 53.78 generations; Average Result: 0.141024')

plt.show()