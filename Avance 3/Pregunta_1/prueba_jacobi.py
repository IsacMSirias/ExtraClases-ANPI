import numpy as np


def angulo(A, i, j):
    """
    Calcula el ángulo de rotación θ necesario para anular el elemento A[i, j]
    en el metodo de Jacobi.

    Parámetros:
        A : numpy.ndarray
            Matriz simétrica cuadrada sobre la que se aplica el metodo de Jacobi.
        i : int
            Índice de la fila del elemento fuera de la diagonal que se desea eliminar.
        j : int
            Índice de la columna del elemento fuera de la diagonal que se desea eliminar.

    Retorna:
        float: Ángulo de rotación θ en radianes.
    """
    Aii = A[i, i]
    Aij = A[i, j]
    Ajj = A[j, j]

    if abs(Aij) > 1e-16:
        # Manejar el caso donde el denominador tiende a cero
        if abs(Aii - Ajj) < 1e-16:
            theta = np.pi / 4 if Aij > 0 else -np.pi / 4
        else:
            theta = (1 / 2) * np.arctan((2 * Aij) / (Aii - Ajj))
    else:
        theta = 0
    return theta


def matrix_rotation(i, j, n, theta):
    """
    Construye la matriz de rotación de Jacobi G(i, j, θ).

    Parámetros:
        i : int
            Índice de la primera fila/columna involucrada en la rotación.
        j : int
            Índice de la segunda fila/columna involucrada en la rotación.
        n : int
            Tamaño de la matriz cuadrada.
        theta : float
            Ángulo de rotación en radianes.

    Retorna:
        numpy.ndarray: Matriz de rotación ortogonal G de tamaño n x n.
    """
    In = np.eye(n)
    Z = np.zeros((n, n))

    # Ajustes en el subespacio (i, j)
    Z[i, i] = np.cos(theta) - 1
    Z[j, j] = np.cos(theta) - 1
    Z[i, j] = -np.sin(theta)
    Z[j, i] = np.sin(theta)

    # Matriz de rotación completa
    G = In + Z
    return G


def jacobi_valores_propios(A, iterMax, tol):
    """
    Aplica el metodo de Jacobi para aproximar los valores propios de una matriz simétrica.

    Parámetros:
        A : numpy.ndarray
            Matriz simétrica cuadrada de tamaño n x n.
        iterMax : int
            Número máximo de iteraciones permitidas.
        tol : float
            Tolerancia de convergencia. Si el cambio entre iteraciones consecutivas
            es menor que 'tol', el algoritmo se detiene.

    Retorna:
        tuple: (xK, k, erk) donde:
            xK  : numpy.ndarray
                Vector con las aproximaciones de los valores propios.
            k   : int
                Número de iteraciones realizadas.
            erk : float
                Error final (norma 2 de la diferencia entre iteraciones).
    """
    Ao = A
    Ak = Ao.copy()
    x0 = np.diag(A)
    xK = x0.copy()
    m = A.shape[0]

    for k in range(iterMax):
        Bk = Ak.copy()
        # Bucle doble sobre los elementos fuera de la diagonal superior
        for i in range(m - 1):
            for j in range(i + 1, m):
                theta = angulo(Bk, i, j)  # Calcular ángulo de rotación
                G = matrix_rotation(i, j, m, theta)  # Matriz de rotación
                Bk = G.T @ Bk @ G  # Transformación ortogonal de Jacobi

        Ak = Bk
        xNK = np.diag(Ak)  # Nuevos valores diagonales (aproximaciones)

        # Cálculo del error relativo entre iteraciones
        erk = np.linalg.norm(xNK - xK, 2)
        xK = xNK

        # Criterio de convergencia
        if erk < tol:
            return xNK, k + 1, erk

    # Si no converge en iterMax iteraciones, retorna la última aproximación
    return xK, k + 1, erk


# ---------------------------------------------------------------------------
# Bloque principal del programa
# ---------------------------------------------------------------------------

# Definición de la matriz de prueba (simétrica de 15x15)
A = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        A[i, j] = 0.5 * (i + 1 + j + 1)

# Cálculo de los valores propios exactos usando NumPy
valores_exactos = np.linalg.eigvals(A)

# Cálculo de los valores propios aproximados con el metodo de Jacobi
iterMax=1000
tol=1e-12
xk, iteraciones, error = jacobi_valores_propios(A, iterMax, tol)

# Resultados
print("Valores propios exactos:")
print(np.sort(valores_exactos))
print("Aproximaciones de los valores propios:")
print(np.sort(xk))
print(f"Iteraciones: {iteraciones}")
print(f"Error final: {error}")