import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier # pip install scikit-learn

# https://www.w3schools.com/python/python_ml_knn.asp

# Coordenadas de los puntos verdes y rojos
green_points = np.array([
    [1, 12],
    [2, 11],
    [3, 10],
    [2, 9],
    [3.5, 8],
    [1, 7],
    [3, 6],
    [2, 5]
])

red_points = np.array([
    [1.5, 9],
    [5.625, 4],
    [3.7, 3],
    [5.1, 3],
    [3, 2],
    [4, 2],
    [7, 2],
    [6.2, 1]
])

# Valor asignado a K
k_value = 3

# Puntos a analizar
task_points = np.array([[2.5, 7], [5.5, 4.5]]) # Conjunto de puntos del problema
# task_points = np.array([[2.5, 7]]) # Punto 1 del problema
# task_points = np.array([[5.5, 4.5]]) # Punto 2 del problema

# Crear etiquetas para los puntos (0 para verde, 1 para rojo)
green_labels = np.zeros(green_points.shape[0])
red_labels = np.ones(red_points.shape[0])

# Combinar puntos y etiquetas
points = np.vstack((green_points, red_points))
labels = np.hstack((green_labels, red_labels))

# Crear y entrenar el modelo KNN con un valor K
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(points, labels)

# Representar en la gráfica los puntos del problema
plt.scatter(green_points[:, 0], green_points[:, 1], color='green', label='Verde')
plt.scatter(red_points[:, 0], red_points[:, 1], color='red', label='Rojo')
plt.scatter(task_points[:, 0], task_points[:, 1], color='blue', marker='x', s=100, label='Puntos del problema')

# Mostrar vecinos más cercanos
for point in task_points:
    distances, indices = knn.kneighbors([point])
    for index in indices[0]:
        plt.plot([point[0], points[index][0]], [point[1], points[index][1]], 'k--')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'KNN con k={k_value}')
plt.xlim(0, 14) # Ajustar el eje X para que llegue hasta 14, mostrando un gráfico cuadrado
plt.ylim(0, 14) # Ajustar el eje Y para que llegue hasta 14, mostrando un gráfico cuadrado
plt.show()
