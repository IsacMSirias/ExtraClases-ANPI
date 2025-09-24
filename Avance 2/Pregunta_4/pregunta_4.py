import numpy as np

# En base a los metodos de eliminacion Gaussiana se tiene lo siguiente:

# Como hay que implementar sustitucion hacia atras, se asume que la matriz A es triangular superior

# Matriz triangular superior A
A1 = np.array([[2, -1,  3,  4,  5],[0,  1, -2,  3,  1],[0,  0,  4, -1,  2],[0,  0,  0,  3, -2],[0,  0,  0,  0,  6]], dtype=float)

# Vector b
b1 = np.array([20, 10, 5, 6, 12], dtype=float)

"""
Entradas:
A: Matriz triangular superior de tamaño m x m
b: Vector de tamaño m

Salidas:
x: Vector solución de tamaño m

"""

def sustitucion_atras(A, b):
    n= len(b)
    x = np.zeros(n)
    
    for i in reversed(range(n)): # recorremos desde la ultima ec hacia la primera
       suma =  sum(A[i,j] * x[j] for j in range(i+1, n))  # suma de los terminos conocidos 
       x[i] = (b[i] - suma) / A[i,i]  # despejamos x[i] (una sustitucion de ec comun y correinte de toda la laif)
    return x

#x = sustitucion_atras(A1, b1)
#print(x)

#matriz para convertir en triangulas superior

A2 = np.array([
[2, 1, 0, 3, 4],
    [1, 2, 1, 1, 0],
    [0, 1, 3, 2, 1],
    [3, 0, 2, 4, 1],
    [1, 2, 1, 0, 3]
], dtype=float)
# la matriz A es de 5x5, y verifique que su det no sea cero para que no reviente
# y que tenga solucion unica

b2 = np.array([10, 7, 8, 15, 9], dtype=float)

"""
Entradas:
A: Matriz de tamaño m x m
b: Vector de tamaño m

Salidas:
At = matriz triangular superior de tamaño m x m
bt = vector modificado de tamaño m
"""

def triangu_sup(A, b):
    
    n = len(b)
    At = A.copy().astype(float)  # Copia de A para no modificar la original
    bt = b.copy().astype(float)  # Copia de b para no modificar el original
    
    for i in range(n-1): # se itera sobre cada fila
        if At[i,i] == 0:
            raise ValueError("no hay pivote, por lo que el sistema no tiene solucion unica")
        
        for j in range(i+1, n): # se itera sobre las filas debajo de la fila i
            fct = At[j,i] / At[i,i]  # factor para eliminar el elemento
            At[j] = At[j] - fct * At[i]  # se actualiza la fila j
            bt[j] = bt[j] - fct * bt[i]  # se actualiza el vector b
    return At, bt

#At, bt = triangu_sup(A2, b2)

#print(At)


def elimi_gauss(A, b):
    At, bt = triangu_sup(A, b)   #triangularizar
    x = sustitucion_atras(At, bt) # resolver 

    return x

#x = elimi_gauss(A2, b2)
#print(x)


# Segun las instrucciones teengo que usar la matriz:


A = np.array([[10, 1, 2, 3, 4, 5, 6, 7, 8, 9],[1, 20, 2, 3, 4, 5, 6, 7, 8, 9],[2, 1, 30, 3, 4, 5, 6, 7, 8, 9],[3, 2, 1, 40, 4, 5, 6, 7, 8, 9],
              [4, 3, 2, 1, 50, 5, 6, 7, 8, 9],[5, 4, 3, 2, 1, 60, 6, 7, 8, 9],[6, 5, 4, 3, 2, 1, 70, 7, 8, 9],[7, 6, 5, 4, 3, 2, 1, 80, 8, 9],
              [8, 7, 6, 5, 4, 3, 2, 1, 90, 9],[9, 8, 7, 6, 5, 4, 3, 2, 1, 100]], dtype=float)

b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

x = elimi_gauss(A, b)
print(x)