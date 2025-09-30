import numpy as np


def sust_atras(A, b):
    """
    Resuelve un sistema triangular superior usando sustitución hacia atrás.

    Parámetros:
        A : numpy.ndarray
            Matriz triangular superior de tamaño n x n
        b : numpy.ndarray
            Vector de términos independientes de tamaño n

    Retorna:
        numpy.ndarray: Vector solución x del sistema A*x = b
    """
    n = len(b)
    x = np.zeros(n)

    # Recorremos desde la última ecuación hacia la primera
    for i in reversed(range(n)):
        # Suma de los términos conocidos (producto de coeficientes por variables ya calculadas)
        suma = sum(A[i, j] * x[j] for j in range(i + 1, n))
        # Despejamos x[i] (sustitución estándar para sistemas triangulares superiores)
        x[i] = (b[i] - suma) / A[i, i]
    return x


def triang_sup(A, b):
    """
    Transforma un sistema lineal en uno triangular superior usando eliminación gaussiana.

    Parámetros:
        A : numpy.ndarray
            Matriz de coeficientes de tamaño n x n
        b : numpy.ndarray
            Vector de términos independientes de tamaño n

    Retorna:
        tuple: (At, bt) donde:
            At: Matriz triangular superior
            bt: Vector modificado correspondiente

    Lanza:
        ValueError: Si aparece un pivote cero (sistema sin solución única)
    """
    n = len(b)
    At = A.copy().astype(float)  # Copia de A para no modificar la original
    bt = b.copy().astype(float)  # Copia de b para no modificar el original

    # Proceso de eliminación gaussiana
    for i in range(n - 1):
        # Verificar que el pivote no sea cero
        if At[i, i] == 0:
            raise ValueError("no hay pivote, por lo que el sistema no tiene solucion unica")

        # Eliminación de elementos debajo del pivote
        for j in range(i + 1, n):  # se itera sobre las filas debajo de la fila i
            fct = At[j, i] / At[i, i]  # factor para eliminar el elemento
            At[j] = At[j] - fct * At[i]  # se actualiza la fila j
            bt[j] = bt[j] - fct * bt[i]  # se actualiza el vector b
    return At, bt


def elimi_gauss(A, b):
    """
    Resuelve un sistema lineal usando eliminación gaussiana completa.

    Parámetros:
        A : numpy.ndarray
            Matriz de coeficientes del sistema
        b : numpy.ndarray
            Vector de términos independientes

    Retorna:
        numpy.ndarray: Vector solución x del sistema A*x = b
    """
    At, bt = triang_sup(A, b)  # Triangularizar el sistema
    x = sust_atras(At, bt)  # Resolver el sistema triangular

    return x


# Matriz del problema específico
A = np.array([
    [10, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 20, 2, 3, 4, 5, 6, 7, 8, 9],
    [2, 1, 30, 3, 4, 5, 6, 7, 8, 9],
    [3, 2, 1, 40, 4, 5, 6, 7, 8, 9],
    [4, 3, 2, 1, 50, 5, 6, 7, 8, 9],
    [5, 4, 3, 2, 1, 60, 6, 7, 8, 9],
    [6, 5, 4, 3, 2, 1, 70, 7, 8, 9],
    [7, 6, 5, 4, 3, 2, 1, 80, 8, 9],
    [8, 7, 6, 5, 4, 3, 2, 1, 90, 9],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 100]
], dtype=float)

# Vector de términos independientes
b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

# Resolver el sistema
x = elimi_gauss(A, b)
print("Solución x:", x)