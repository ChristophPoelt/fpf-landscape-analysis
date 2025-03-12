import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Schwefel-Funktion
def schwefel(x):
    return (418.9828872724339 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def h1(x1, x2):
    term1 = np.sin(x1 - x2 / 8) ** 2
    term2 = np.sin(x2 + x1 / 8) ** 2
    denominator = np.sqrt((x1 - 8.6998) ** 2 + (x2 - 6.7665) ** 2) + 1
    return (term1 + term2) / denominator

# Erzeuge eine zufällige Punktwolke zur k-NN-Gradientenberechnung
num_points = 25000  # Anzahl der zufällig generierten Punkte
dim = 2
X_samples = np.random.uniform(-500, 500, (num_points, dim))
y_samples = np.array([schwefel(x) for x in X_samples])

# KD-Tree zur schnellen k-NN Suche
tree = KDTree(X_samples)

# Diskrete Gradient Methode mit k-NN
def discrete_gradient_method(starting_point, k=12, learning_rate=5, max_iters=100, tolerance=0.05):
    x = np.array(starting_point, dtype=np.float64)
    path = [schwefel(x)]  # Speichert die Funktionswerte über Iterationen

    for step in range(max_iters):
        # Finde k nächste Nachbarn
        dists, indices = tree.query(x, k=k)
        neighbors = X_samples[indices]
        f_neighbors = y_samples[indices]

        # Berechnung der diskreten Gradienten
        gradients = np.array([(f_neighbors[i] - schwefel(x)) / (np.linalg.norm(neighbors[i] - x) ** 2) * (neighbors[i] - x)
                              for i in range(k)])

        # Gewichteter Mittelwert der Gradienten
        weights = 1 / (dists + 1e-8)  # Vermeidung von Division durch 0
        weights /= np.sum(weights)  # Normalisierung
        avg_gradient = np.sum(weights[:, None] * gradients, axis=0)

        # Gradient Descent Schritt
        x -= learning_rate * avg_gradient

        # Stelle sicher, dass x innerhalb [-500, 500] bleibt
        x = np.clip(x, -500, 500)

        path.append(schwefel(x))

        # Überprüfe Konvergenz
        if np.linalg.norm(avg_gradient) < tolerance:
            break

    return path, step + 1  # Rückgabe: Verlauf der Funktionswerte, Anzahl der Schritte

# 100 zufällige Startpunkte
num_runs = 100
start_points = np.random.uniform(-500, 500, (num_runs, dim))

# Führe DGM aus und speichere Ergebnisse
results = [discrete_gradient_method(start) for start in start_points]

# Extrahiere Werte
final_values = np.array([res[0][-1] for res in results])  # Letzter Funktionswert jeder Optimierung
num_steps = np.array([res[1] for res in results])  # Anzahl an Schritten bis zur Konvergenz

# Berechne Durchschnittswerte und Standardabweichung
avg_final_value = np.mean(final_values)
std_final_value = np.std(final_values)
avg_steps = np.mean(num_steps)
std_steps = np.std(num_steps)

print(f"Durchschnittlicher finaler Wert: {avg_final_value:.2f} ± {std_final_value:.2f}")
print(f"Durchschnittliche Anzahl an Schritten bis zur Konvergenz: {avg_steps:.2f} ± {std_steps:.2f}")

# Histogramm der finalen Werte
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(final_values, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("DGM results' 2D-Schwefel target value")
plt.ylabel("Frequency")
plt.title("Distribution of DGM results")

# Histogramm der Anzahl an Schritten
plt.subplot(1, 2, 2)
plt.hist(num_steps, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Number of steps to convergence")
plt.ylabel("Frequency")
plt.title("Distribution of steps to convergence")

plt.tight_layout()
plt.show()