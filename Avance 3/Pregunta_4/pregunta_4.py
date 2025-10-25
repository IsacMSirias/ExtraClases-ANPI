import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def y_exacta(x):
    """
    Calcula la solución analítica exacta del problema de EDO propuesto.

    Parámetros:
        x : numpy.ndarray o float
            Valor o conjunto de valores donde evaluar la función exacta.

    Retorna:
        numpy.ndarray o float: Valor de la solución exacta en los puntos dados.

    Descripción:
        La función corresponde a la solución exacta del problema de frontera:
            y'' + p(x)*y' + q(x)*y = r(x)
        donde p, q y r son funciones dadas. Se usa como referencia
        para comparar con la solución numérica obtenida.
    """
    return np.sin(6 - x) / (np.sin(5) * np.sqrt(x))


def thomas(a, b, c, d):
    """
    Resuelve un sistema lineal tridiagonal mediante el método de Thomas.

    Parámetros:
        a : numpy.ndarray
            Subdiagonal inferior del sistema (longitud n-1).
        b : numpy.ndarray
            Diagonal principal del sistema (longitud n).
        c : numpy.ndarray
            Superdiagonal superior del sistema (longitud n-1).
        d : numpy.ndarray
            Vector del lado derecho del sistema (longitud n).

    Retorna:
        numpy.ndarray: Vector solución del sistema tridiagonal.

    """
    n = len(b)

    # Eliminación hacia adelante
    for i in range(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    # Sustitución hacia atrás
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def edo2(p, q, r, h, a, b, y0, yn):
    """
    Resuelve una ecuación diferencial ordinaria de segundo orden
    mediante diferencias finitas y el metodo de Thomas.

    Parámetros:
        p, q, r : funciones
            Coeficientes de la EDO en la forma:
                y'' + p(x)y' + q(x)y = r(x)
        h : float
            Tamaño de paso (espaciado entre nodos).
        a : float
            Extremo izquierdo del dominio (x = a).
        b : float
            Extremo derecho del dominio (x = b).
        y0 : float
            Condición de frontera en x = a.
        yn : float
            Condición de frontera en x = b.

    Retorna:
        tuple: (x, y)
            x : numpy.ndarray
                Vector de posiciones espaciales.
            y : numpy.ndarray
                Vector de soluciones aproximadas.

    """
    n_float = (b - a) / h
    n = int(round(n_float))

    # Validación de consistencia del tamaño de paso
    if abs(n - n_float) > 1e-12:
        raise print("ekisde")

    x = np.linspace(a, b, n + 1)
    m = n - 1  # Número de incógnitas internas

    # Caso trivial: solo extremos
    if m == 0:
        return x, np.array([y0, yn])

    # Inicialización de coeficientes del sistema tridiagonal
    A = np.zeros(m - 1)
    B = np.zeros(m)
    C = np.zeros(m - 1)
    D = np.zeros(m)

    inv_h2 = 1.0 / (h * h)
    inv_2h = 1.0 / (2 * h)

    # Construcción del sistema
    for k in range(1, n):
        xi = x[k]
        pk = p(xi)
        qk = q(xi)
        rk = r(xi)

        # Coeficientes del esquema de diferencias finitas
        a_k = inv_h2 - pk * inv_2h
        b_k = -2 * inv_h2 + qk
        c_k = inv_h2 + pk * inv_2h

        i = k - 1
        B[i] = b_k
        D[i] = rk
        if i - 1 >= 0:
            A[i - 1] = a_k
        if i + 1 <= m - 1:
            C[i] = c_k

    # Ajuste por condiciones de frontera
    D[0] -= (inv_h2 - p(x[1]) * inv_2h) * y0
    D[-1] -= (inv_h2 + p(x[n - 1]) * inv_2h) * yn

    # Resolución del sistema tridiagonal
    y_interior = thomas(A.copy(), B.copy(), C.copy(), D.copy())

    # Reconstrucción del vector completo (incluye fronteras)
    y = np.zeros(n + 1)
    y[0], y[-1] = y0, yn
    y[1:n] = y_interior

    return x, y


# ---------------------------------------------------------------------------
# Definición simbólica de las funciones p(x), q(x), r(x)
# ---------------------------------------------------------------------------

x_sym = sp.symbols('x')

# Coeficientes simbólicos de la EDO
p_sym = -1 / x_sym
q_sym = (1 / (4 * x_sym ** 2)) - 1
r_sym = 0

# Conversión simbólica → funciones numéricas compatibles con numpy
p = sp.lambdify(x_sym, p_sym, 'numpy')
q = sp.lambdify(x_sym, q_sym, 'numpy')
r = sp.lambdify(x_sym, r_sym, 'numpy')

# Condiciones de frontera y dominio
a, b = 1.0, 6.0
y0, yn = 1.0, 0.0

# ---------------------------------------------------------------------------
# Evaluación numérica para distintos tamaños de paso
# ---------------------------------------------------------------------------

h_values = [1, 0.5, 0.2, 0.1, 0.01]

plt.figure()

# Resolver y graficar para distintos tamaños de paso
for h in h_values:
    x, y = edo2(p, q, r, h, a, b, y0, yn)
    plt.plot(x, y, linewidth=1.2, label=f'h={h}')

# ---------------------------------------------------------------------------
# Comparación con la solución exacta
# ---------------------------------------------------------------------------
x_ex = np.linspace(1, 6, 300)
y_ex = y_exacta(x_ex)
plt.plot(x_ex, y_ex, 'k', linewidth=2, label='Exacta')

plt.title("Aproximaciones con distintos tamaños de paso")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True)
plt.show()