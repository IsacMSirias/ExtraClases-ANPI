from math import *
import numpy as np
import time 
import sympy as sp
import pandas as pd

np.seterr(over='ignore')

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

biseccion_inicio = time.perf_counter()
biseccion_raiz, biseccion_iteraciones, biseccion_error = biseccion(fun, -0.3, -0.1, 1e-8, 10000)  
biseccion_final = time.perf_counter()



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
            return None, k, None
        else:
            xk = xk - fun(xk) / dfn(xk) # Actualiza xk usando la fórmula de Newton-Raphson
            erk = abs(fun(xk)) # Calcula el error absoluto
            if erk < tol:  # Criterio de parada
                k = k + 1
                break
    return xk, k + 1, erk # Retorna la aproximación, número de iteraciones y el error


nr_inicio = time.perf_counter()
nr_raiz, nr_iteraciones, nr_error = newton_raphson(fun, -0.1, 1e-8, 10000)
nr_final = time.perf_counter()


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
            return None, k, None
        else:
            xk = xk-(fun(xk)**2)/(fun(xk+fun(xk))-fun(xk)) # formula de Steffensen
            erk = abs(fun(xk)) # error absoluto
            if erk < tol:  # Criterio de parada
                k = k + 1
                break
    return xk, k + 1, erk 

st_inicio = time.perf_counter()
st_raiz, st_iteraciones, st_error = stiffensen(fun, -0.1, 1e-8, 10000) 
st_final = time.perf_counter()


# Pregunta 4, inciso d Metodo de la Secante
# retorna [xk, k, erk]
def secante(fun, x0, x1, tol, max_iter):

    for k in range(max_iter):
        if abs(fun(x1) - fun(x0)) < 1e-15:  # Evitar división por cero
            return None, k, None
        else:
            xk = x1-fun(x1)*(x1-x0)/(fun(x1)-fun(x0)) # Actualiza xk usando la fórmula de la secante
            np.seterr(over='ignore')
            x0,x1 = x1,xk 
            erk = abs(fun(xk)) # Calcula el error absoluto
            if erk < tol:  # Criterio de parada
                k = k + 1
                break
    return xk, k + 1, erk 


sec_inicio = time.perf_counter()
sec_raiz, sec_iteraciones, sec_error = secante(fun, -0.3, -0.1, 1e-8, 10000)
sec_final = time.perf_counter()


# Pregunta 4, inciso d Metodo de la falsa posicion
# retorna [xk, k, erk]

def falsa_posicion(fun, a, b, tol, max_iter):
    if fun(a) * fun(b) > 0:
        return None, 0, None
    for k in range(max_iter):
        xk = a - (fun(a) * (a - b)) / (fun(a) - fun(b)) # calcula la formula de la secante

        if bolzano(fun, a, xk):  #  valida que se cumpla bolzano
            b = xk  
        else:
            a = xk  

        erk = abs(fun(xk))  # erk
        if erk < tol:  # Criterio de parada
            return xk, k + 1, erk

    return xk, k + 1, erk

#prueba del metodo
fp_inicio = time.perf_counter()
fp_raiz, fp_iteraciones, fp_error = falsa_posicion(fun, -0.3, -0.1, 1e-8, 10000)
fp_final = time.perf_counter()

#Inciso f) de Pregunta 4: Newton-Hermite-Ostrowski
def nho (fun, x0, tol, max_iter):
    xk = x0
    x = sp.symbols('x')  # Variable simbólica
    f_sym = x * sp.exp(-x) - 5 - (sp.cos(x) / x)  # Función simbólica para derivar

    # Derivada simbólica
    dfs = sp.diff(f_sym, x)
    dfn = sp.lambdify(x, dfs, 'numpy')

    for k in range(max_iter):
        fpx = dfn(xk)

        # Evitar división por cero con tolerancia
        if abs(fpx) < 1e-15:
            return None, k, None

        #Newton-Raphson
        zn = xk - fun(xk) / fpx
        fz = fun(zn)
        fpz = dfn (zn)

        # Evitar división por cero en H
        if abs(3 * fpz - fpx) < 1e-15:
            return None, k, None

        H = (fpx - fpz) / (3 * fpz - fpx)

        # Fórmula NHO
        xnew = zn - H * (fz / fpx)
        erk = abs(fun(xnew))

        # Criterio de parada
        if erk < tol:
            return xnew, k + 1, erk
        xk = xnew
    return xk, max_iter, erk

nho_inicio = time.perf_counter()
nho_raiz, nho_iteraciones, nho_error = nho (fun, -0.1, 1e-8, 10000)
nho_final = time.perf_counter()



# Tabla de resultados

datos = {
    "Metodo": ["Biseccion", "Newton-Raphson", "Steffensen", "Secante", "Falsa Posicion", "NHO"],
    "R(Xk)": [biseccion_raiz, nr_raiz, st_raiz, sec_raiz, fp_raiz, nho_raiz],
    "k": [biseccion_iteraciones, nr_iteraciones, st_iteraciones, sec_iteraciones, fp_iteraciones, nho_iteraciones],
    "|f(Xk)|": [biseccion_error, nr_error, st_error, sec_error, fp_error, nho_error],
    "Tiempo (s)": [biseccion_final - biseccion_inicio, nr_final - nr_inicio,
                   st_final - st_inicio, sec_final - sec_inicio, fp_final - fp_inicio, nho_final - nho_inicio]
}

df = pd.DataFrame(datos)
df["R(Xk)"] = df["R(Xk)"].map(lambda x: f"{x:.15f}")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
print(df)