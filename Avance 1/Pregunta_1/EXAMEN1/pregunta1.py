import numpy as np
import sympy as sp

def metodo_iterativo(fun, x0, tol, max_iter):
    """
    Método iterativo para aproximar un cero de la función:
    f(x) = ln(x^2 + 1) + sin(x) - π

    Según la sucesión:
    x_{k+1} = x_k - (2 f(x_k) [f(z_k) - f(x_k)]) / (f(x_k + f(x_k)) - f(x_k - f(x_k)))
    con
    z_k = x_k + (2 f(x_k)^2) / (f(x_k + f(x_k)) - f(x_k - f(x_k)))
    """

    xk = x0

    for k in range(max_iter):
        fxk = fun(xk)
        fxk_menos = fun(xk - fxk)
        fxk_mas = fun(xk + fxk)

        # Calcular z_k
        zk = xk + (2 * (fxk**2)) / (fxk_mas - fxk_menos)
        fzk = fun(zk)

        # Calcular x_{k+1}
        xk1 = xk - (2 * fxk * (fzk - fxk)) / (fxk_mas - fxk_menos)

        # Criterio de parada
        err = max(abs(fun(xk1)), abs((xk1 - xk) / xk1))
        if err < tol:
            return xk1, k + 1, err

        xk = xk1

    # Si no converge dentro de max_iter
    return xk, max_iter, err


# -------------------------------------------------------------
# Definición de la función simbólica y numérica
# -------------------------------------------------------------
x = sp.symbols('x')
f_sym = sp.log(x**2 + 1) + sp.sin(x) - sp.pi
f_num = sp.lambdify(x, f_sym, 'numpy')

# -------------------------------------------------------------
# Ejecución del método
# -------------------------------------------------------------
raiz, iteraciones, error = metodo_iterativo(f_num, x0=4, tol=1e-12, max_iter=1000)

print(f"\nAproximacion del cero: x ~ {raiz:.15f}")
print(f"Iteraciones realizadas: {iteraciones}")
print(f"Error obtenido: {error:.3e}")
print(f"f(x) = {f_num(raiz):.15e}")
