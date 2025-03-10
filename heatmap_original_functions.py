import numpy as np
import matplotlib.pyplot as plt

def schwefel(x, y):
    return 418.9829 * 2 - (x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y))))

# Definiere den Bereich der Heatmap
x_min, x_max = -500, 500
y_min, y_max = -500, 500

# Erstelle ein Gitter von x- und y-Werten
resolution = 500  # Anzahl der Punkte pro Achse
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

# Berechne die Schwefel-2D-Werte
Z = schwefel(X, Y)

# Erstelle die Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
plt.colorbar(label='Schwefel target value')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Heatmap of the original Schwefel Function')
plt.show()