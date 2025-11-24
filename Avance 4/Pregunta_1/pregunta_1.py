import numpy as np
import sympy as sp
from matplotlib import pyplot as plt


def f(x):
    """
    Función matemática a interpolar: f(x) = ln(arcsin(x)) / ln(x)

    Parámetros:
        x : float o array
            Punto(s) donde evaluar la función

    Retorna:
        float o array: Valor(es) de la función evaluada en x
    """
    return np.log(np.arcsin(x)) / np.log(x)


def newtonSimbolico(x, y, n):
    """
    Construye el polinomio de interpolación de Newton usando diferencias divididas.

    Parámetros:
        x : numpy.ndarray
            Vector de nodos de interpolación (puntos x)
        y : numpy.ndarray
            Vector de valores de la función en los nodos (puntos y)
        n : int
            Grado del polinomio de interpolación (número de puntos - 1)

    Retorna:
        sympy.Expr: Polinomio de interpolación de Newton en forma simbólica
    """
    # Crear tabla de diferencias divididas
    tabla = np.zeros([(n + 1), (n + 1)], dtype=float)

    # Paso 1: Llenar la primera columna con los valores de y
    for i in range(n + 1):
        tabla[i, 0] = y[i]

    # Calcular diferencias divididas recursivamente
    for j in range(1, n + 1):
        for i in range((n + 1) - j):
            tabla[i, j] = (tabla[i + 1, j - 1] - tabla[i, j - 1]) / (x[i + j] - x[i])

    # Paso 2: Construir el polinomio de Newton
    X = sp.symbols('X')
    polinomio = tabla[0, 0]  # Término constante
    termino = 1  # Productorio de (X - x_i)

    # Sumar términos sucesivos del polinomio
    for k in range(1, n + 1):
        termino = termino * (X - x[k - 1])  # Actualizar productorio
        polinomio = polinomio + tabla[0, k] * termino  # Sumar término actual

    # Expandir el polinomio para forma canónica
    polinomio = sp.expand(polinomio)

    return polinomio


def main():
    """
    Función principal que ejecuta la interpolación de Newton.

    Retorna:
        tuple: (xv, yv, polinomio) donde:
            xv : array de nodos de interpolación
            yv : array de valores en los nodos
            polinomio : polinomio de interpolación simbólico
    """
    # Definir nodos de interpolación equidistantes
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # Evaluar función en los nodos
    y = np.array([f(0.1), f(0.2), f(0.3), f(0.4), f(0.5), f(0.6), f(0.7), f(0.8)])
    n = 7  # Grado del polinomio

    return x, y, newtonSimbolico(x, y, n)


# Ejecutar interpolación
xv, yv, p = main()
print("Polinomio de interpolación:")
print(p)

# Convertir polinomio simbólico a función numérica
X = sp.symbols('X')
P = sp.lambdify(X, p, 'numpy')
print("Valores del polinomio en los nodos:", P(xv))

print("Puntos x:", xv)
print("Puntos y:", yv)

# Crear gráfica comparativa
x_exact = np.linspace(0.1, 0.8, 1000)  # Puntos densos para gráfica suave
y_exact = f(x_exact)  # Valores exactos de la función
y_polinomio = P(x_exact)  # Valores del polinomio interpolante

# Configurar gráfica
plt.figure(figsize=(10, 6))

# Graficar función exacta
plt.plot(x_exact, y_exact, 'b', label='Función f(x)', linewidth=2)
# Graficar polinomio interpolante
plt.plot(x_exact, y_polinomio, 'g', label='Polinomio de Newton', linewidth=2)
# Marcar nodos de interpolación
plt.scatter(xv, yv, color='red', label='Puntos de interpolación')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolación por Diferencias Divididas de Newton')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
