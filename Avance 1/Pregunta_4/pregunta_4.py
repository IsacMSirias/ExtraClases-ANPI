from math import *
import numpy as np
import time 
import sympy as sp

# Pregunta 4, inciso a

# Ecuación a resolver f(x) = x*e**−x − 5 − (cos(x)/x)
# Yo se que este ejericio lo hicimos en clase, pero queria ver si podia hacerlo de una manera en la cual pudiera usar funciones auxiliares para tener un 
# codigo mas limpio y entendible (al menos para mi )

def fun(x):
    if x == 0:
        return float('inf')  # si hay un valor muy grande, no revienta
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
        if erk < tol : # criterio de parada
        #or (b - a)/2 < tol: esto lo hace muchísimo mas exacto 
            return xk, k+1, erk 

    return xk, k+1, erk 

inicio = time.perf_counter()
raiz, iteraciones, error = biseccion(fun, -0.3, -0.1, 1e-8, 10000)  
final = time.perf_counter()
print ("Metodo de Biseccion")
print(f"sol aproximada: {raiz:.15f}")
print(f"Iteraciones: {iteraciones}")
print(f"|f(xk)| = {error:.15f}")
print(f"Tiempo total: {final - inicio:.15f} segundos")
print("---------------------------------------------------")


# Pregunta 4, inciso b Metodo de Newton-Raphson
# retorna [xk, k, erk]
def newton_raphson(fun, x0, tol, max_iter):
    
    xk = x0
    x = sp.symbols('x')  # Variable simbólica
    f_sym = x * sp.exp(-x) - 5 - (sp.cos(x) / x) # Función simbólica para derivar

    # Derivada simbólica y luego convertirla en función numérica
    dfs = sp.diff(f_sym, x)    
    dfn = sp.lambdify(x, dfs, 'numpy')  

    for k in range(max_iter):
        if dfn(xk) == 0:  # Evitar división por cero
            print("Derivada nula.")
            return None, k, None
        else:
            xk = xk - fun(xk) / dfn(xk) # Actualiza xk usando la fórmula de Newton-Raphson
            erk = abs(fun(xk)) # Calcula el error absoluto
            if erk < tol:  # Criterio de parada
                k = k + 1
                break
    return xk, k + 1, erk # Retorna la aproximación, número de iteraciones y el error


inicio = time.perf_counter()
raiz, iteraciones, error = newton_raphson(fun, -0.1, 1e-8, 10000)
final = time.perf_counter()
print ("Metodo de Newton-Raphson")
print(f"sol aproximada: {raiz:.15f}")
print(f"Iteraciones: {iteraciones}")
print(f"|f(xk)| = {error:.15f}")
print(f"Tiempo total: {final - inicio:.15f} segundos")
print("---------------------------------------------------")

#Pregunta 4, inciso c Metodo de Steffensen

def steffensen(fun, x0, tol, max_iter):
    return None, None, None


# Pregunta 4, inciso d Metodo de la Secante
#En la buena y sana teoria, este metodo es pareciod Newton-Raphson, lo unico distinto serian como las valicaciones, que no hay derivada, y el calculo
# retorna [xk, k, erk]

def stiffensen(fun, x0, tol, max_iter):
    xk = x0
    for k in range(max_iter):
        if abs(fun(xk)-fun(xk-1)) < 1e-15:  # evitar división por cero
            print("Diferencia en f(x) nula.")
            return None, k, None
        else:
            xk = xk-(fun(xk)**2)/(fun(xk+fun(xk))-fun(xk)) # formula de Steffensen
            erk = abs(fun(xk)) # error absoluto
            if erk < tol:  # Criterio de parada
                k = k + 1
                break
    return xk, k + 1, erk 

inicio = time.perf_counter()
raiz, iteraciones, error = stiffensen(fun, -1, 1e-8, 10000)
final = time.perf_counter()
print ("Metodo de Steffensen")
print(f"sol aproximada: {raiz:.15f}")
print(f"Iteraciones: {iteraciones}")
print(f"|f(xk)| = {error:.15f}")
print(f"Tiempo total: {final - inicio:.15f} segundos")
print("---------------------------------------------------")

def secante(fun, x0, x1, tol, max_iter):

    for k in range(max_iter):
        if abs(fun(x1) - fun(x0)) < 1e-15:  # Evitar división por cero
            print("Diferencia en f(x) nula.")
            return None, k, None
        else:
            xk = x1-fun(x1)*(x1-x0)/(fun(x1)-fun(x0)) # Actualiza xk usando la fórmula de la secante
            x0,x1 = x1,xk  # Actualiza los puntos para la siguiente iteración
            erk = abs(fun(xk)) # Calcula el error absoluto
            if erk < tol:  # Criterio de parada
                k = k + 1
                break
    return xk, k + 1, erk # Retorna la aproximación, número de iteraciones y el error


inicio = time.perf_counter()
raiz, iteraciones, error = secante(fun, -0.3, -0.1, 1e-8, 10000)
final = time.perf_counter()
print ("Metodo de la Secante")
print(f"sol aproximada: {raiz:.15f}")
print(f"Iteraciones: {iteraciones}")
print(f"|f(xk)| = {error:.15f}")
print(f"Tiempo total: {final - inicio:.15f} segundos")
print("---------------------------------------------------")
