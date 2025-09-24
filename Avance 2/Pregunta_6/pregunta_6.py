import numpy as np

#Antes de hacerlo a pata, hice una funcion para generar la matriz y el vector
def matrix(n):
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = 5 # diagonal principal
        if i < n-1:
            A[i, i+1] = 1   # diagonal superior
            A[i+1, i] = 1   # diagonal inferior
    d = np.full(n, -14)
    d[0] = d[-1] = -12
    
    return A, d

A,d = matrix(100)

#print(A)
#print(d)

"""
Entradas:
A: matriz tridiagonal (100 x 100
d: vector (100)
    
Retorna:
x: solución del sistema (n)
"""

def thomas(A,d):
    
    n = len(d)
    
    # extremos de las diagonales
    a = np.zeros(n-1)  # diagonal inferior
    b = np.zeros(n)    # diagonal principal
    c = np.zeros(n-1)  # diagonal superior
    for i in range(n):
        b[i] = A[i,i]
        if i < n-1:
            c[i] = A[i,i+1]
            a[i] = A[i+1,i]
            
    # defino unas cuantas variables temporales
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) #diagonales y vector d
    
    #sustitucion hacia adelante
    for i in range(1, n):
        m = ac[i-1] / bc[i-1]
        bc[i] = bc[i] - m * cc[i-1]
        dc[i] = dc[i] - m * dc[i-1]
    
    #sustitucion hacia atras
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in reversed(range(n-1)):
        x[i] = (dc[i] - cc[i] * x[i+1]) / bc[i]

    return x

x = thomas(A,d)
print(x)