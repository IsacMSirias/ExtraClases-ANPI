import math
import numpy as np
import sympy as sp

def fun(x):
    return (1 + x) * np.sin(x) - 1

def disc (a, b, c):
    return math.sqrt(b**2 - 4*a*c)


def muller (f,x0, x1, x2, tol, iterMax):
    f0 = f(x0)
    f1 = f(x1)
    f2 = f(x2)

    for k in range(iterMax):
        # Calcular las constantes a, b, c del polinomio cuadrático
        # usando las fórmulas del metodo de Müller

        h0 = x0 - x2
        h1 = x1 - x2

        # Sistema de ecuaciones para determinar a, b, c
        # f0 = a*h0² + b*h0 + c
        # f1 = a*h1² + b*h1 + c
        # f2 = c

        # Resolver para a y b
        denom = h0 * h1 * (h0 - h1)

        if abs(denom) < 1e-15:   # Evitar división por cero
            return x2, k+1, abs(f2)

        a = (h1 * (f0 - f2) - h0 * (f1 - f2)) / denom
        b = (h0 ** 2 * (f1 - f2) - h1 ** 2 * (f0 - f2)) / denom
        c = f2

        discriminante = disc(a,b,c)

        # Elegir el denominador más grande (para estabilidad numérica)
        if abs(b + discriminante) > abs(b - discriminante):
            denominador = b + discriminante
        else:
            denominador = b - discriminante

        # Calcular la nueva aproximación
        if abs(denominador) < 1e-15:  # Evitar división por cero
            x3 = x2
        else:
            x3 = x2 - (2 * c) / denominador

        # Evaluar la función en la nueva aproximación
        f3 = f(x3)

        # Verificar criterio de parada
        if abs(f3) < tol:
            return x3, k + 1, abs(f3)

        # Preparar para la siguiente iteración
        # Mantener los tres puntos más cercanos a la raíz
        x0, x1, x2 = x1, x2, x3
        f0, f1, f2 = f1, f2, f3

    return x2, iterMax, abs(f2)

def main():
    #Parametros
    x0 = 2.5
    x1 = 2.75
    x2 = 3.0
    tol = 1e-10
    iterMax = 1000

    return muller(fun, x0, x1, x2, tol, iterMax)

# Ejecutar el metodo y obtener resultados
raiz, iteraciones, error = main()

# Mostrar resultados
print("=== MÉTODO DE MÜLLER ===")
print(f"Función: (1 + x)sin(x) - 1 = 0")
print("=" * 40)
print(f"Aproximación de la raíz: x = {raiz}")
print(f"Valor de |f(x)|: {error}")
print(f"Número de iteraciones: {iteraciones}")