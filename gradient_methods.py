import json
import numpy as np
from scipy.spatial import KDTree

def discrete_gradient_method(data, k=20, alpha=0.1, epsilon=0.01, max_iterations=100):

    points = np.array([(x1, x2) for x1, x2, _ in data])
    values = np.array([fpf for _, _, fpf in data])
    values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-6)  # Normalize to [0,1]

    kdtree = KDTree(points)

    # Randomly initialize starting point
    idx = np.random.choice(len(points))
    xk = points[idx]

    trajectory = [xk]
    for iteration in range(max_iterations):
        # Find k-nearest neighbors
        distances, indices = kdtree.query(xk, k=k)
        neighbors = points[indices]
        neighbor_values = values[indices]

        current_idx = np.argmin(np.linalg.norm(points - xk, axis=1))
        current_value = values[current_idx]

        # Compute discrete gradient using weighted average method
        weights = 1 / (distances ** 2 + 1e-6)  # Avoid division by zero
        weights /= weights.sum()  # Normalize weights

        grad = np.sum(weights[:, None] * (neighbor_values[:, None] - current_value) * (neighbors - xk), axis=0)
        grad_magnitude = np.linalg.norm(grad)

        print(f"Iteration {iteration}: xk = {xk}, grad = {grad}, grad_magnitude = {grad_magnitude}, current_value = {current_value}")

        # Stopping criterion
        if grad_magnitude < epsilon:
            print("Stopping criterion met: Gradient magnitude too small.")
            break

        # Update step
        xk = xk - alpha * grad
        trajectory.append(xk)

    return trajectory

with open('_schwefelFB.json', 'r') as file:
    data_json = json.load(file)

landscape_data = []
for run in data_json:
    for individual in run['individualsWithFPF']:
        landscape_data.append((individual['x1'], individual['x2'], individual['fpfValue']))

result_trajectory = discrete_gradient_method(landscape_data)
print("Final Optimization trajectory:", result_trajectory)
