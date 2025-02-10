import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def discrete_gradient(f, x, h=1e-5):
    """
    Computes the discrete gradient approximation for a nonsmooth function.
    """
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = h
        f_plus = f(x + e_i)
        f_minus = f(x - e_i)
        g[i] = (f_plus - f_minus) / (2 * h)
    return g


def discrete_gradient_descent_knn(f, points, fpf_values, lr=0.1, max_iter=100, tol=1e-6, k=12):
    """
    Performs discrete gradient descent on a set of points using k-Nearest Neighbors.
    """
    tree = KDTree(points)
    optimized_points = np.copy(points)

    for i in range(max_iter):
        gradients = np.zeros_like(points)

        for j, x in enumerate(points):
            _, idxs = tree.query(x, k=k)
            knn_points = points[idxs]
            knn_fpf = fpf_values[idxs]
            avg_fpf = np.mean(knn_fpf)
            gradients[j] = discrete_gradient(lambda p: f(p, avg_fpf), x)

        if np.linalg.norm(gradients) < tol:
            break

        optimized_points -= lr * gradients

    return optimized_points


def nonsmooth_function(x, avg_fpf):
    return np.abs(x[0]) + np.abs(x[1]) + avg_fpf


with open('_schwefelFB.json', 'r') as file:
    data = json.load(file)

x1, x2, fpfValue = [], [], []
for entry in data:
    for individual in entry['individualsWithFPF']:
        x1.append(individual['x1'])
        x2.append(individual['x2'])
        fpfValue.append(individual['fpfValue'])

points = np.array(list(zip(x1, x2)))
fpf_values = np.array(fpfValue)

# Run Gradient Descent
opt_points = discrete_gradient_descent_knn(nonsmooth_function, points, fpf_values)

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, fpfValue, c=fpfValue, cmap='viridis', marker='o', label='Initial')
ax.scatter(opt_points[:, 0], opt_points[:, 1], fpf_values, c='red', marker='x', label='Optimized')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('fpfValue')
ax.legend()
plt.show()