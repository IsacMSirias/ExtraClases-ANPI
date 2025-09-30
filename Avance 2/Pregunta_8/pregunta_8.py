import numpy as np

# Crear la matriz A y el vector b
A = np.zeros((1000,1000))
b = np.ones(1000)
def matrix():
    for i in range(1000):
        for j in range(1000):
            if i == j:
                A[i,j] = 1001
            else:
                A[i,j] = 1
    return A

def sust_atras(A, b):
    n= len(b)
    x = np.zeros(n)
    
    for i in reversed(range(n)): # recorremos desde la ultima ec hacia la primera
       suma =  sum(A[i,j] * x[j] for j in range(i+1, n))  # suma de los terminos conocidos 
       x[i] = (b[i] - suma) / A[i,i]  # despejamos x[i] (una sustitucion de ec comun y correinte de toda la laif)
    return x


def gauss_seidel(A, b, x0, tol, max_iter): 
    # L D U                                              
    D = np.diag(np.diag(A))
    L= np.tril(A, k=-1)
    U= np.triu(A, k=1)
    # M
    M= D+U
    # d
    d= sust_atras(M, b)
    xk= x0
    
    # ahora si se viene la iteacion
    
    for k in range(max_iter): #para k = 1, 2, ..., max_iter 
        yk = -L @ xk 
        zk = sust_atras(M,yk)  # calculo de z^k = M^-1 y^k
        xk = zk + d # calculo de x^(k+1) = z^k + d
        erk = np.linalg.norm(A @ xk - b) # erk
        if erk < tol:
            return erk # retornamos el error como lo pide la instruccion
    return erk



A1 = matrix()
x0 = np.zeros(1000)

erk = gauss_seidel(A1, b, x0, 1e-10,1000)# el ejercicio no indica la tolerancia ni la iteracion maxima, 
print(erk)                                              #pero pongo estas ya porque en otros ejericios se usaron estos datos