import numpy as np
from scipy.linalg import solve

def pentadiagonal(m, a, b, c, d, e):
    """
    Construye una matriz pentadiagonal de dimensiones m×m a partir de vectores que representan sus diagonales.

    Una matriz pentadiagonal tiene cinco diagonales distintas de cero: la principal, dos subdiagonales
    y dos superdiagonales. Esta función organiza los vectores de entrada en estas diagonales.

    Parámetros:
        m : int
            Dimensión de la matriz (m >= 5)
        a : array_like, shape (m,1)
            Vector para la diagonal principal
        b : array_like, shape (m-1,1)
            Vector para la primera superdiagonal (justo arriba de la principal)
        c : array_like, shape (m-1,1)
            Vector para la primera subdiagonal (justo debajo de la principal)
        d : array_like, shape (m-2,1)
            Vector para la segunda superdiagonal
        e : array_like, shape (m-2,1)
            Vector para la segunda subdiagonal

    Retorna:
        numpy.ndarray: Matriz pentadiagonal de dimensiones m×m con las diagonales especificadas
        None: Si m < 5 (no cumple con el requisito mínimo)
    """
    
    if m < 5:
        print("El valor de m debe ser mayor o igual a 5")
        return None
    else:
        A = np.zeros((m, m)) # se crea una matriz nula como base
        A[np.arange(m), np.arange(m)] = a # vec  m × 1
        A[np.arange(m-1), np.arange(1, m)] = b # vec  (m − 1) × 1
        A[np.arange(1, m), np.arange(m-1)] = c # Igual que el anterir
        A[np.arange(m-2), np.arange(2, m)] = d  # vec (m − 2) × 1.
        A[np.arange(2, m), np.arange(m-2)] = e # igual que el anterior

    
        return A

# Configuración del problema específico

m = 2500  # Dimensión de la matriz

# Construcción de los vectores para las diagonales según las fórmulas dadas
a = np.array([2*(i+1) for i in range(m)], dtype=float)  # 2(i+1),para i=0,1,...,m−1
b = np.array([(i+1)/3 for i in range(m-1)], dtype=float) # i+1/3​,i=0,1,...,m−2
c = np.array([i/3 for i in range(m-1)], dtype=float) # i/3​,i=0,1,...,m−2
d = np.array([(i+2)/4 for i in range(m-2)], dtype=float) # (i+2)/4​,i=0,1,...,m−3
e = np.array([i/4 for i in range(m-2)], dtype=float) # i/4​,i=0,1,...,m−3

# Construcción del vector del lado derecho del sistema lineal
h = np.array([2*i for i in range(m)], dtype=float) # 2i,i=0,1,...,m−1

#  Contruccion de la matriz y su sol
A = pentadiagonal(m, a, b, c, d, e)  # Crear matriz pentadiagonal
x = solve(A, h)  # Resolver el sistema lineal A*x = h

# Cálculo del error de la solución
error = np.linalg.norm(A @ x - h, 2)
print("Error:", error)