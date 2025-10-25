import numpy as np
import sympy as sp


def sturm_polinomio(T):
    """
    Calcula el polinomio característico de una matriz tridiagonal simétrica
    utilizando la recurrencia de Sturm.

    Parámetros:
        T : numpy.ndarray
            Matriz tridiagonal simétrica de tamaño n x n.

    Retorna:
        tuple: (polinomio, x)
            polinomio : sympy.Expr
                Polinomio característico expresado simbólicamente.
            x : sympy.Symbol
                Variable simbólica utilizada en el polinomio.

    Descripción:
        La secuencia de Sturm permite calcular recursivamente el polinomio
        característico sin necesidad de determinantes explícitos. Se define:
            p₀ = 1
            p₁ = a₁ - x
            pₖ = (aₖ - x)pₖ₋₁ - bₖ₋₁² pₖ₋₂
        donde a son los elementos de la diagonal principal
        y b los de la sub/superdiagonal.
    """
    x = sp.Symbol('x')
    a = np.diag(T)       # Diagonal principal
    b = np.diag(T, 1)    # Subdiagonal superior
    m = len(a)

    # Condiciones iniciales de la recurrencia de Sturm
    p0 = 1
    p1 = a[0] - x

    # Recurrencia para construir el polinomio característico
    for k in range(1, m):
        pk = (a[k] - x) * p1 - (b[k - 1]) ** 2 * p0
        p0 = p1
        p1 = pk

    # Expansión simbólica final
    polinomio = sp.expand(p1)
    return polinomio, x


def gershgorin_intervalos(T):
    """
    Calcula los intervalos de Gershgorin para una matriz tridiagonal simétrica.

    Parámetros:
        T : numpy.ndarray
            Matriz tridiagonal simétrica de tamaño n x n.

    Retorna:
        tuple: (Ints, intervalo_global)
            Ints : numpy.ndarray
                Intervalos individuales [centro - radio, centro + radio]
                para cada fila (discos de Gershgorin).
            intervalo_global : list[float]
                Intervalo total que contiene todos los valores propios.

    Descripción:
        Los intervalos de Gershgorin estiman la localización de los valores
        propios de una matriz. Cada valor propio se encuentra dentro de al menos
        uno de los discos centrados en A[ii] con radio igual a la suma de las
        magnitudes de los elementos fuera de la diagonal de la fila i.
    """
    m = T.shape[0]
    Ints = np.zeros((m, 2))

    # Primer disco de Gershgorin (fila 1)
    R1 = abs(T[0, 1])
    Ints[0, 0] = T[0, 0] - R1
    Ints[0, 1] = T[0, 0] + R1

    # Último disco (fila m)
    Rm = abs(T[m - 1, m - 2])
    Ints[m - 1, 0] = T[m - 1, m - 1] - Rm
    Ints[m - 1, 1] = T[m - 1, m - 1] + Rm

    # Discos intermedios (filas 2 a m-1)
    for k in range(1, m - 1):
        Rk = abs(T[k, k - 1]) + abs(T[k, k + 1])
        Ints[k, 0] = T[k, k] - Rk
        Ints[k, 1] = T[k, k] + Rk

    # Intervalo global (mínimo a máximo)
    intervalo_global = [np.min(Ints[:, 0]), np.max(Ints[:, 1])]

    return Ints, intervalo_global


def bolzano(f, a, b):
    """
    Verifica si una función f cumple el teorema de Bolzano en un intervalo [a, b].

    Parámetros:
        f : función
            Función continua en el intervalo.
        a : float
            Extremo izquierdo.
        b : float
            Extremo derecho.

    Retorna:
        bool: True si f(a) y f(b) tienen signos opuestos, False en caso contrario.
    """
    return f(a) * f(b) < 0


def falsa_posicion(fun, a, b, tol, max_iter):
    """
    Encuentra una raíz de una función usando el método de la falsa posición (regla falsa).

    Parámetros:
        fun : función
            Función cuya raíz se desea aproximar.
        a : float
            Extremo izquierdo del intervalo inicial.
        b : float
            Extremo derecho del intervalo inicial.
        tol : float
            Tolerancia de convergencia (criterio de parada).
        max_iter : int
            Número máximo de iteraciones permitidas.

    Retorna:
        tuple: (xk, iteraciones, error)
            xk : float
                Aproximación de la raíz.
            iteraciones : int
                Número de iteraciones ejecutadas.
            error : float
                Error absoluto final (|f(xk)|).

    Notas:
        - Requiere que f(a) y f(b) tengan signos opuestos (Bolzano).
        - Usa interpolación lineal para estimar la raíz en cada paso.
    """
    if not bolzano(fun, a, b):
        return None, 0, None

    for k in range(max_iter):
        # Fórmula de la falsa posición (interpolación lineal)
        xk = a - (fun(a) * (a - b)) / (fun(a) - fun(b))

        # Verificar convergencia
        erk = abs(fun(xk))
        if erk < tol:
            return xk, k + 1, erk

        # Actualizar el intervalo según el signo
        if bolzano(fun, a, xk):
            b = xk
        else:
            a = xk

    # Si no converge, retorna la última aproximación
    return xk, max_iter, erk


def calcular_valores_propios(T, h):
    """
    Calcula los valores propios de una matriz tridiagonal simétrica combinando:
        - Método de Sturm (polinomio característico)
        - Teorema de Gershgorin (acotación del espectro)
        - Método de Falsa Posición (búsqueda numérica de raíces)

    Parámetros:
        T : numpy.ndarray
            Matriz tridiagonal simétrica de tamaño n x n.
        h : float
            Paso de discretización usado para muestrear el intervalo global.

    Retorna:
        tuple: (vect_val_pro, polinomio)
            vect_val_pro : list[float]
                Lista con los valores propios encontrados.
            polinomio : sympy.Expr
                Polinomio característico simbólico.

    Proceso general:
        1. Se obtiene el polinomio característico P(x) con Sturm.
        2. Se determinan intervalos de Gershgorin que acotan las raíces.
        3. Se discretiza el intervalo global y se aplica la falsa posición
           en los subintervalos donde P(x) cambia de signo.
    """
    print("=" * 60)
    print("CÁLCULO DE VALORES PROPIOS - MÉTODO INTEGRADO")
    print("=" * 60)

    # ---- 1. Polinomio característico mediante Sturm ----
    print("\n1. POLINOMIO CARACTERÍSTICO (Sturm):")
    polinomio, x_sym = sturm_polinomio(T)
    print(f"P(x) = {polinomio}")

    # Conversión simbólica → función numérica
    p_func = sp.lambdify(x_sym, polinomio, 'numpy')

    # ---- 2. Intervalos de Gershgorin ----
    print("\n2. INTERVALOS DE GERSHGORIN:")
    Ints, intervalo_global = gershgorin_intervalos(T)

    for i in range(len(Ints)):
        print(f"   Disco {i + 1}: [{Ints[i, 0]:.4f}, {Ints[i, 1]:.4f}]")

    print(f"\n   Intervalo global: [{intervalo_global[0]:.4f}, {intervalo_global[1]:.4f}]")

    # ---- 3. Cálculo de valores propios mediante Falsa Posición ----
    print("\n3. CÁLCULO DE VALORES PROPIOS (Falsa Posición):")
    a_global, b_global = intervalo_global

    # Discretización del intervalo global con paso h
    x_val = np.arange(a_global, b_global + h, h)
    vect_val_pro = []

    print(f"   Buscando raíces en intervalo [{a_global:.4f}, {b_global:.4f}] con h = {h}")

    # Búsqueda de raíces donde ocurre cambio de signo
    for i in range(len(x_val) - 1):
        xi = x_val[i]
        xim1 = x_val[i + 1]

        # Detectar cambio de signo (posible raíz)
        if p_func(xi) * p_func(xim1) < 0:
            val_prop, iteraciones, error = falsa_posicion(p_func, xi, xim1, 1e-12, 1000)
            if val_prop is not None:
                vect_val_pro.append(val_prop)
                print(f"   ✓ Raíz encontrada en [{xi:.4f}, {xim1:.4f}] → {val_prop:.8f}")

    # Ordenar los valores propios de menor a mayor
    vect_val_pro.sort()

    return vect_val_pro, polinomio


def construir_matriz_T():
    """
    Construye una matriz tridiagonal simétrica T de tamaño 12x12.

    Retorna:
        numpy.ndarray: Matriz tridiagonal simétrica T.

    Descripción:
        - La diagonal principal y la subdiagonal se definen manualmente.
        - Los valores se asignan de modo que T sea simétrica.
    """
    T = np.zeros((12, 12))

    # Diagonal principal
    diagonal = [-1, -3, 3, -2, -1, 0, 0, -3, -2, 0, 0, 1]

    # Sub/superdiagonal
    subdiagonal = [4, 1, -2, 3, -3, 2, -2, -2, 1, 3, -1]

    # Asignar la diagonal principal
    for i in range(12):
        T[i, i] = diagonal[i]

    # Asignar sub y superdiagonal (simetría)
    for i in range(11):
        T[i, i + 1] = subdiagonal[i]
        T[i + 1, i] = subdiagonal[i]

    return T


def main():
    """
    Función principal del programa.

    Ejecuta la construcción de la matriz tridiagonal T,
    calcula sus valores propios con el método integrado
    (Sturm + Gershgorin + Falsa Posición),
    y retorna los resultados principales.
    """
    T = construir_matriz_T()
    h = 0.1  # Paso de discretización
    valores_propios, polinomio = calcular_valores_propios(T, h)

    return valores_propios, polinomio, T, h


# ---------------------------------------------------------------------------
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ---------------------------------------------------------------------------
valores_propios, polinomio, T, h = main()

print("\n" + "=" * 60)
print("RESULTADOS FINALES")
print("=" * 60)
print("Valores propios calculados:")
for i, val in enumerate(valores_propios):
    print(f"  λ_{i + 1:2d}: {val:.10f}")

# ---- Verificación con NumPy ----
print("\nVERIFICACIÓN CON NUMPY:")
valores_exactos = np.linalg.eigvals(T)
valores_exactos.sort()

for i, val in enumerate(valores_exactos):
    print(f"  λ_{i + 1:2d}: {val:.10f}")

# Diferencia máxima entre métodos
print(f"\nDiferencia máxima: {np.max(np.abs(valores_propios - valores_exactos)):.2e}")
