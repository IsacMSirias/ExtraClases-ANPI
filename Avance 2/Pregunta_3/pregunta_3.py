import numpy as np
import math
import matplotlib.pyplot as plt
def matrix(m=45, n=30):
    A = np.zeros((m, n), dtype=float)
    for i in range(1, m+1): #Se inicia en (1,m+1) para que a la hora de acceder a A[0,0] se calcule como 1^2+1^2
        for j in range(1, n+1): # Aca es la misma cosa
            A[i-1, j-1] = i**2 + j**2
    return A
A = matrix()


A.T / (np.linalg.norm(A, 'fro')**2) # inicializacion de X0

def pseudoinversa(A, p, tol=1e-5, max_iter=100000):
    m, n = A.shape
    # inicialización (Newton-Schulz)
    X = A.T / (np.linalg.norm(A, 'fro')**2)
    
    for k in range(max_iter):
        M = A @ X  # (m x m)
        # sumatoria
        S = np.zeros_like(M)
        potencia = np.eye(M.shape[0])
        for q in range(1, p+1):
            coeff = ((-1)**(q-1)) * math.comb(p, q)
            S += coeff * potencia
            potencia = potencia @ M  # siguiente potencia
        
        # actualización
        X_act = X @ S
        
        # error de parada
        error = np.linalg.norm(A @ X_act @ A - A, 'fro')
        if error < tol:
            return X_act, k+1, error
        
        X = X_act
    
    return X, max_iter, error

"""
X_aprox, iters, final_error = pseudoinversa(A, p=10)
print("Iteraciones:", iters)
print("Error final:", final_error)
"""


p_values = [1,2,3,4,5,6,7,8,10]
iters = []   # <- debería ser una lista de enteros, uno por cada p

for p in p_values:
    _, iteraciones, _ = pseudoinversa(A, p)
    print(f"p = {p}, iteraciones = {iteraciones}")
    iters.append(iteraciones)

print("Lista de iteraciones:", iters)

plt.plot(p_values, iters, marker='o')
plt.xlabel("Valor de p")
plt.ylabel("Número de iteraciones")
plt.title("Convergencia vs grado del método iterativo")
plt.grid(True)
plt.show()