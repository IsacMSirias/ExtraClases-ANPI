
import numpy as np

def fun(x):
    if x == 0:
        return float('inf')  # si hay un valor muy grande, no revienta
    return np.log(x**2 +1) + np.sin(x) - np.pi

def metodo_iterativo(fun, x0 = 4, tol=10e-12, iter_max=10000):
    xk = x0
    
    for k in range(iter_max):
        fxk = fun(xk)
        fxk_menos = fun(xk - fxk)
        fxk_mas = fun(xk + fxk)
        denom = fxk_mas - fxk_menos
   
        
        zk = xk + 2 * (fxk**2) / denom
        xk1 = xk - (2 * fxk * (fun (zk) - fxk)) / denom
        
        erk = max(abs(fun(xk1)), abs(xk1 -xk)/abs(xk1))
        
        if erk < tol:
            return xk1, k, erk
        
        xk = xk1
        
    return xk1, iter_max, erk 


raiz,k,error = metodo_iterativo(fun)

print(f"Raíz aproximada: {raiz}")
print(f"Número de iteraciones: {k}")
print(f"Error aproximado: {error}")


def f(x):
    return x**2 - np.exp(-x) - 3*x + 2

def df(x):
    return 2*x + np.exp(-x) - 3

def H(x, z):
    return (df(x) - df(z)) / (3*df(z) - df(x))

def HNO(f, df, x0=0, tol=1e-10, iter_max=1000):
    xk = x0
    
    for k in range(iter_max):
        fxk = f(xk)
        dfxk = df(xk)
        
        
        zk = xk - fxk /dfxk
        xk1 = zk - H(xk, zk) * (fxk/dfxk)
        
        error = abs(xk1 - xk)/abs(xk1)
        
        if error < tol:
            return xk1, k, error
        xk = xk1
        
    return xk1, iter_max, error

raiz,k,error = HNO(f, df)
 
 
print(f"Raíz aproximada: {raiz}")
print(f"Número de iteraciones: {k}")
print(f"Error aproximado: {error}")



def bolzano(f, a, b):
    return f(a) * f(b) < 0

