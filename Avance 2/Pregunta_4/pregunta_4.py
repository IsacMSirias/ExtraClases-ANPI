import numpy as np

"""
Entradas:
A: Matriz triangular superior de tamaño m x m
b: Vector de tamaño m

Salidas:
x: Vector solución de tamaño m

"""

def sust_atras(A, b):
    n= len(b)
    x = np.zeros(n)
    
    for i in reversed(range(n)): # recorremos desde la ultima ec hacia la primera
       suma =  sum(A[i,j] * x[j] for j in range(i+1, n))  # suma de los terminos conocidos 
       x[i] = (b[i] - suma) / A[i,i]  # despejamos x[i] (una sustitucion de ec comun y correinte de toda la laif)
    return x
"""
Entradas:
A: Matriz de tamaño m x m
b: Vector de tamaño m

Salidas:
At = matriz triangular superior de tamaño m x m
bt = vector modificado de tamaño m
"""

def triang_sup(A, b):
    
    n = len(b)
    At = A.copy().astype(float)  # Copia de A para no modificar la original
    bt = b.copy().astype(float)  # Copia de b para no modificar el original
    
    for i in range(n-1):
        if At[i,i] == 0:
            raise ValueError("no hay pivote, por lo que el sistema no tiene solucion unica")
        
        for j in range(i+1, n): # se itera sobre las filas debajo de la fila i
            fct = At[j,i] / At[i,i]  # factor para eliminar el elemento
            At[j] = At[j] - fct * At[i]  # se actualiza la fila j
            bt[j] = bt[j] - fct * bt[i]  # se actualiza el vector b
    return At, bt



def elimi_gauss(A, b):
    At, bt = triang_sup(A, b)   #triangularizar
    x = sust_atras(At, bt) # resolver

    return x


# Matriz del problema
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

b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

x = elimi_gauss(A, b)
print("Solución x:",x)