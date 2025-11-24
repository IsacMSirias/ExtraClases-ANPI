import sympy as sp
import scipy.optimize as opt


def cota_simpson_puntos(f, a, b, tol):
    """
    Calcula el número mínimo de subintervalos n necesario para que el error
    de la regla de Simpson compuesta sea menor que una tolerancia dada.

    Parámetros:
        f : sympy.Expr
            Función a integrar en forma simbólica
        a : float
            Límite inferior de integración
        b : float
            Límite superior de integración
        tol : float
            Tolerancia máxima permitida para el error

    Retorna:
        float: Número mínimo de subintervalos n requeridos
    """
    # Definir variables simbólicas
    x, n = sp.symbols('x, n')
    # Tamaño del paso en función de n
    h = (b - a) / n
    # Expresión simbólica de la función
    fs = f(x)

    # Calcular la cuarta derivada (necesaria para error de Simpson)
    fds = sp.diff(fs, x, 4)
    # Convertir a función numérica para evaluación
    fdn = sp.lambdify(x, fds, 'numpy')

    # Paso 1: Encontrar el máximo absoluto de |f⁽⁴⁾(x)| en [a, b]
    def newfun(x):
        """
        Función auxiliar para maximización (minimizar el negativo del valor absoluto)

        Parámetros:
            x : float
                Punto donde evaluar

        Retorna:
            float: Negativo del valor absoluto de la cuarta derivada
        """
        return -abs(fdn(x))

    # Encontrar punto donde |f⁽⁴⁾(x)| es máximo usando búsqueda acotada
    xmax = opt.fminbound(newfun, a, b)
    # Valor máximo de |f⁽⁴⁾(x)| en el intervalo
    alphamax = abs(fdn(xmax))

    # Paso 2: Resolver ecuación de error para n
    # Fórmula del error para regla de Simpson compuesta
    num = ((b-a) * (h**4))
    # Ecuación: error ≤ tolerancia
    ecuacion = sp.Eq((num / 2880) * alphamax, tol)
    # Resolver ecuación para n
    soluciones = sp.solve(ecuacion, n)

    # Filtrar soluciones reales positivas
    sol = [float(s) for s in soluciones if s.is_real and s > 0]
    n = sol[0]  # Tomar la primera solución válida

    return n


def f(x):
    """
    Función a integrar: f(x) = eˣ * (26 - 10x + x²)

    Parámetros:
        x : sympy.Symbol
            Variable simbólica

    Retorna:
        sympy.Expr: Expresión simbólica de la función
    """
    return sp.exp(x)*(26-10*x+x**2)


# Parámetros de integración
a = 5  # Límite inferior
b = 5.55  # Límite superior
tol = 1e-8  # Tolerancia de error

# Calcular n
n = cota_simpson_puntos(f, a, b, tol)

print("Valor de n:")
print(n)