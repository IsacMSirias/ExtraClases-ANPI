from math import *
import numpy as np
import time 

# Pregunta 4, inciso a

# Ecuación a resolver f(x) = x*e**−x − 5 − (cos(x)/x)
# Yo se que este ejericio lo hicimos en clase, pero queria ver si podia hacerlo de una manera en la cual pudiera usar funciones auxiliares para tener un 
# codigo mas limpio y entendible (al menos para mi )

def fun(x):
    if x == 0:
        return float('inf')  # o algún valor muy grande para que no rompa
    return x * np.exp(-x) - 5 - (np.cos(x) / x)

# de manera teorica se vio en clase el teorema de Bolzano, que dice que si f(a) y f(b) tienen signos distintos
# entonces existe al menos una raiz en (a,b). Por lo tanto, se puede hacer un barrido en un intervalo grande
# para encontrar un intervalo donde se cumpla esto y luego aplicar el metodo de biseccion:

def bolzano(f, a, b):
    return f(a) * f(b) < 0

def biseccion(fun, a, b, tol, max_iter):
    
    for k in range(max_iter):
        xk = (a + b) / 2 # calcula el punto medio

        if bolzano(fun, a, xk): # verifica en que subintervalo esta la raiz
            b = xk # asigna el nuevo extremo derecho
        else:
            a = xk # asigna el nuevo extremo izquierdo

        erk = abs(fun(xk)) # calcula el error absoluto
        if erk < tol or (b - a)/2 < tol: # criterio de parada
            return xk, k+1, erk 

    return xk, k+1, erk 

inicio = time.perf_counter()
raiz, iteraciones, error = biseccion(fun, -0.2, -0.1, 1e-8, 10000)
final = time.perf_counter()
print(f"Raíz aproximada: {raiz:.15f}")
print(f"Iteraciones: {iteraciones}")
print(f"|f(xk)| = {error:.15f}")
print(f"Tiempo total: {final - inicio:.15f} segundos")


# Pregunta 4, inciso b

