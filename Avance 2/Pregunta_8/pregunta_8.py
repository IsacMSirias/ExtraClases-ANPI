import numpy as np


def matrix():
    """
    Construye una matriz especial de 1000×1000 donde:
    - Los elementos diagonales son 1001
    - Los elementos no diagonales son 1

    Retorna:
        numpy.ndarray: Matriz de 1000×1000 con el patrón especificado
    """
    A = np.zeros((1000, 1000))
    for i in range(1000):
        for j in range(1000):
            if i == j:
                A[i, j] = 1001  # Diagonal principal
            else:
                A[i, j] = 1  # Elementos fuera de la diagonal
    return A


def sust_atras(A, b):
    """
    Resuelve un sistema triangular superior usando sustitución hacia atrás.

    Parámetros:
        A : numpy.ndarray
            Matriz triangular superior n×n
        b : numpy.ndarray
            Vector de términos independientes de tamaño n

    Retorna:
        numpy.ndarray: Solución x del sistema A*x = b
    """
    n = len(b)
    x = np.zeros(n)

    # Recorrer desde la última ecuación hasta la primera
    for i in reversed(range(n)):
        # Suma de términos conocidos (variables ya calculadas)
        suma = sum(A[i, j] * x[j] for j in range(i + 1, n))
        # Despejar variable i-ésima
        x[i] = (b[i] - suma) / A[i, i]
    return x


def gauss_seidel(A, b, x0, tol, max_iter):
    """
    Implementa el método de Gauss-Seidel para resolver sistemas lineales.

    El método utiliza la descomposición A = L + D + U y resuelve iterativamente
    el sistema (D + U)x^(k+1) = b - Lx^k

    Parámetros:
        A : numpy.ndarray
            Matriz del sistema n×n
        b : numpy.ndarray
            Vector de términos independientes
        x0 : numpy.ndarray
            Aproximación inicial
        tol : float
            Tolerancia para el criterio de parada
        max_iter : int
            Número máximo de iteraciones

    Retorna:
        float: Error final de la aproximación ||A*x - b||
    """
    # Descomposición de la matriz: A = L + D + U
    D = np.diag(np.diag(A))  # Matriz diagonal
    L = np.tril(A, k=-1)  # Parte triangular inferior estricta
    U = np.triu(A, k=1)  # Parte triangular superior estricta

    # Matriz del sistema: M = D + U
    M = D + U
    # Término constante: d = M^(-1) * b
    d = sust_atras(M, b)

    xk = x0  # Aproximación inicial

    # Iteraciones del método
    for k in range(max_iter):
        # Calcular y^k = -L * x^k
        yk = -L @ xk
        # Resolver M * z^k = y^k para obtener z^k
        zk = sust_atras(M, yk)
        # Actualizar aproximación: x^(k+1) = z^k + d
        xk = zk + d
        # Calcular error residual
        erk = np.linalg.norm(A @ xk - b)

        # Verificar criterio de convergencia
        if erk < tol:
            return erk  # Retornar error cuando se cumple tolerancia

    # Retornar error después de máximo de iteraciones
    return erk


# Configuración del problema
A1 = matrix()  # Generar matriz
b = np.ones(1000)  # Vector de términos independientes
x0 = np.zeros(1000)  # Vector inicial

# Resolver sistema usando Gauss-Seidel
erk = gauss_seidel(A1, b, x0, 1e-10, 1000)
print(erk)