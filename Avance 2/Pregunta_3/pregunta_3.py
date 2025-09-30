import numpy as np
import math
import matplotlib.pyplot as plt


def matrix(m=45, n=30):
    """
    Construye una matriz especial donde cada elemento A[i,j] = i² + j².

    Parámetros:
        m : int
            Número de filas de la matriz
        n : int
            Número de columnas de la matriz

    Retorna:
        numpy.ndarray: Matriz de dimensiones m×n con elementos i² + j²
    """
    A = np.zeros((m, n), dtype=float)
    for i in range(1, m + 1):  # Se inicia en (1,m+1) para que a la hora de acceder a A[0,0] se calcule como 1^2+1^2
        for j in range(1, n + 1):  # Aca es la misma cosa
            A[i - 1, j - 1] = i ** 2 + j ** 2
    return A


# Generar matriz de prueba
A = matrix()

# Inicialización para el método iterativo (Newton-Schulz)
A.T / (np.linalg.norm(A, 'fro') ** 2)  # inicializacion de X0


def pseudoinversa(A, p, tol=1e-5, max_iter=100000):
    """
    Calcula la pseudoinversa de una matriz usando un método iterativo basado en Newton-Schulz.

    Parámetros:
        A : numpy.ndarray
            Matriz para la cual calcular la pseudoinversa
        p : int
            Grado del polinomio de aproximación
        tol : float
            Tolerancia para el criterio de parada
        max_iter : int
            Número máximo de iteraciones permitidas

    Retorna:
        tuple: (pseudoinversa_aproximada, iteraciones_realizadas, error_final)
    """
    m, n = A.shape
    # inicialización (Newton-Schulz)
    X = A.T / (np.linalg.norm(A, 'fro') ** 2)

    # Iteraciones del método
    for k in range(max_iter):
        M = A @ X  # (m x m)

        # Sumatoria para la aproximación polinomial
        S = np.zeros_like(M)
        potencia = np.eye(M.shape[0])
        for q in range(1, p + 1):
            coeff = ((-1) ** (q - 1)) * math.comb(p, q)
            S += coeff * potencia
            potencia = potencia @ M  # siguiente potencia

        # actualización de la aproximación
        X_act = X @ S

        # error de parada: norma de A*X*A - A
        error = np.linalg.norm(A @ X_act @ A - A, 'fro')
        if error < tol:
            return X_act, k + 1, error

        X = X_act

    # Retornar mejor aproximación después de iteraciones máximas
    return X, max_iter, error


"""
# Ejemplo de uso:
X_aprox, iters, final_error = pseudoinversa(A, p=10)
print("Iteraciones:", iters)
print("Error final:", final_error)
"""

# Estudio de convergencia para diferentes valores de p
p_values = [1, 2, 3, 4, 5, 6, 7, 8, 10]
iters = []  # <- debería ser una lista de enteros, uno por cada p

# Probar diferentes grados del polinomio
for p in p_values:
    _, iteraciones, _ = pseudoinversa(A, p)
    print(f"p = {p}, iteraciones = {iteraciones}")
    iters.append(iteraciones)

print("Lista de iteraciones:", iters)

# Graficar resultados de convergencia
plt.plot(p_values, iters, marker='o')
plt.xlabel("Valor de p")
plt.ylabel("Número de iteraciones")
plt.title("Convergencia vs grado del método iterativo")
plt.grid(True)
plt.show()