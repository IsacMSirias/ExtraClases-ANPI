import numpy as np


def matrix(n):
    """
    Genera una matriz tridiagonal y un vector para sistemas lineales.

    La matriz tiene 5 en la diagonal principal y 1 en las diagonales superior e inferior.
    El vector tiene -14 en la mayoría de posiciones y -12 en los extremos.

    Parámetros:
        n : int
            Dimensión de la matriz y vector

    Retorna:
        tuple: (A, d) donde:
            A: Matriz tridiagonal n×n
            d: Vector de términos independientes de tamaño n
    """
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = 5  # diagonal principal
        if i < n - 1:
            A[i, i + 1] = 1  # diagonal superior
            A[i + 1, i] = 1  # diagonal inferior
    d = np.full(n, -14)
    d[0] = d[-1] = -12

    return A, d


# Generar matriz y vector para n=100
A, d = matrix(100)


def thomas(A, d):
    """
    Resuelve un sistema tridiagonal usando el algoritmo de Thomas (TDMA).

    El algoritmo de Thomas es una versión optimizada de la eliminación gaussiana
    para matrices tridiagonales con complejidad O(n).

    Parámetros:
        A : numpy.ndarray
            Matriz tridiagonal de tamaño n×n
        d : numpy.ndarray
            Vector de términos independientes de tamaño n

    Retorna:
        numpy.ndarray: Solución x del sistema A*x = d
    """
    n = len(d)

    # Extraer las tres diagonales de la matriz
    a = np.zeros(n - 1)  # diagonal inferior (subdiagonal)
    b = np.zeros(n)  # diagonal principal
    c = np.zeros(n - 1)  # diagonal superior (superdiagonal)

    for i in range(n):
        b[i] = A[i, i]
        if i < n - 1:
            c[i] = A[i, i + 1]
            a[i] = A[i + 1, i]

    # Crear copias de las diagonales y vector para no modificar los originales
    ac, bc, cc, dc = map(np.array, (a, b, c, d))

    # Fase de eliminación (sustitución hacia adelante)
    for i in range(1, n):
        m = ac[i - 1] / bc[i - 1]  # multiplicador
        bc[i] = bc[i] - m * cc[i - 1]  # actualizar diagonal principal
        dc[i] = dc[i] - m * dc[i - 1]  # actualizar vector derecho

    # Fase de sustitución hacia atrás
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]  # solución para la última variable
    for i in reversed(range(n - 1)):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]  # solución para variable i

    return x


# Resolver el sistema tridiagonal
x = thomas(A, d)
print(x)