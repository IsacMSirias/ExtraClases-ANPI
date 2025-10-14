import numpy as np
import sympy as sp

def angulo(A,i ,j):
    Aii = A[i,i]
    Aij = A[i,j]
    Ajj = A[j,j]
    if abs(Aij) > 1e-16:
        # Manejar caso cuando denominador es cero
        if abs(Aii - Ajj) < 1e-16:
            theta = np.pi / 4 if Aij > 0 else -np.pi / 4
        else:
            theta = (1 / 2) * np.arctan((2 * Aij) / (Aii - Ajj))
    else:
        theta = 0
    return theta

def matrix_rotation(i, j, n, theta):
    In = np.eye(n)

    Z = np.zeros((n, n))

    Z[i, i] = np.cos(theta) - 1
    Z[j,j] = np.cos(theta) - 1

    Z[i, j] = -np.sin(theta)
    Z[j, i] = np.sin(theta)

    G = In + Z

    return G

def jacobi_valores_propios (A, iterMax, tol):
    Ao = A
    Ak= Ao.copy()
    x0= np.diag(A)
    xK=x0.copy()
    m=A.shape[0]

    for k in range(iterMax):
        Bk = Ak.copy()
        for i in range(m-1):
            for j in range(i+1,m):
                theta = angulo(Bk,i,j)
                G = matrix_rotation(i,j,m,theta)
                Bk = G.T @ Bk @ G

        Ak = Bk
        xNK= np.diag(Ak)

        erk = np.linalg.norm(xNK-xK, 2)
        xK = xNK
        if erk < tol:
            return xNK, k+1, erk
    return xK, k+1, erk

A = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        A[i, j] = 0.5 * (i + 1 + j + 1)

# Calcular valores exactos
valores_exactos = np.linalg.eigvals(A)

# Aproximar con Jacobi
xk, iteraciones, error = jacobi_valores_propios(A, iterMax=1000, tol=1e-12)

print(np.sort(valores_exactos))
print("Aproximaciones de los valores propios:")
print(np.sort(xk))
print(f"Iteraciones: {iteraciones}")
print(f"Error final: {error}")