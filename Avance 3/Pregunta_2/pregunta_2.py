import numpy as np
import sympy as sp


def sturm_polinomio(T):
    """
    Calcula el polinomio característico usando la recurrencia de Sturm
    para una matriz tridiagonal simétrica
    """
    x = sp.Symbol('x')
    a = np.diag(T)
    b = np.diag(T, 1)
    m = len(a)

    p0 = 1
    p1 = a[0] - x

    for k in range(1, m):
        pk = (a[k] - x) * p1 - (b[k - 1]) ** 2 * p0
        p0 = p1
        p1 = pk

    polinomio = sp.expand(p1)
    return polinomio, x


def gershgorin_intervalos(T):
    """
    Calcula los intervalos de Gershgorin para una matriz tridiagonal
    """
    m = T.shape[0]
    Ints = np.zeros((m, 2))

    # Primer fila
    R1 = abs(T[0, 1])
    Ints[0, 0] = T[0, 0] - R1
    Ints[0, 1] = T[0, 0] + R1

    # Última fila
    Rm = abs(T[m - 1, m - 2])
    Ints[m - 1, 0] = T[m - 1, m - 1] - Rm
    Ints[m - 1, 1] = T[m - 1, m - 1] + Rm

    # Filas intermedias
    for k in range(1, m - 1):
        Rk = abs(T[k, k - 1]) + abs(T[k, k + 1])
        Ints[k, 0] = T[k, k] - Rk
        Ints[k, 1] = T[k, k] + Rk

    intervalo_global = [np.min(Ints[:, 0]), np.max(Ints[:, 1])]

    return Ints, intervalo_global

def bolzano(f, a, b):
    """
    Verifica si se cumple el teorema de Bolzano para una función en un intervalo.
    """
    return f(a) * f(b) < 0


def falsa_posicion(fun, a, b, tol, max_iter):
    """
    Encuentra una raíz de la función utilizando el método de la falsa posición.

    Parámetros:
        fun : función
            Función cuya raíz se busca
        a : float
            Extremo izquierdo del intervalo inicial
        b : float
            Extremo derecho del intervalo inicial
        tol : float
            Tolerancia para el criterio de parada
        max_iter : int
            Número máximo de iteraciones permitidas

    Retorna:
        tuple: (Aproximación de la raíz, número de iteraciones, error absoluto)
    """
    if not bolzano(fun, a, b):
        return None, 0, None

    for k in range(max_iter):
        # Fórmula de la falsa posición
        xk = a - (fun(a) * (a - b)) / (fun(a) - fun(b))

        # Verificar convergencia
        erk = abs(fun(xk))
        if erk < tol:
            return xk, k + 1, erk

        # Actualizar intervalo
        if bolzano(fun, a, xk):
            b = xk
        else:
            a = xk

    return xk, max_iter, erk


def calcular_valores_propios(T, h):
    """
    Calcula todos los valores propios de una matriz tridiagonal
    integrando Sturm, Gershgorin y Bisección
    """
    print("=" * 60)
    print("CÁLCULO DE VALORES PROPIOS - MÉTODO INTEGRADO")
    print("=" * 60)

    # 1. Obtener polinomio característico (Sturm)
    print("\n1. POLINOMIO CARACTERÍSTICO (Sturm):")
    polinomio, x_sym = sturm_polinomio(T)
    print(f"P(x) = {polinomio}")

    # Convertir a función numérica
    p_func = sp.lambdify(x_sym, polinomio, 'numpy')

    # 2. Obtener intervalos (Gershgorin)
    print("\n2. INTERVALOS DE GERsHGORIN:")
    Ints, intervalo_global = gershgorin_intervalos(T)

    for i in range(len(Ints)):
        print(f"   Disco {i + 1}: [{Ints[i, 0]:.4f}, {Ints[i, 1]:.4f}]")

    print(f"\n   Intervalo global: [{intervalo_global[0]:.4f}, {intervalo_global[1]:.4f}]")

    # 3. Calcular valores propios (Falsa Posicion)
    print("\n3. CÁLCULO DE VALORES PROPIOS (Falsa Posición):")
    a_global, b_global = intervalo_global

    # Discretizar el intervalo global
    x_val = np.arange(a_global, b_global + h, h)
    vect_val_pro = []

    print(f"   Buscando raíces en intervalo [{a_global:.4f}, {b_global:.4f}] con h = {h}")

    for i in range(len(x_val) - 1):
        xi = x_val[i]
        xim1 = x_val[i + 1]

        if p_func(xi) * p_func(xim1) < 0:
            val_prop, iteraciones, error = falsa_posicion(p_func, xi, xim1,1e-12,1000)
            if val_prop is not None:
                vect_val_pro.append(val_prop)
                print(f"   ✓ Raíz encontrada en [{xi:.4f}, {xim1:.4f}] → {val_prop:.8f}")

    # Ordenar los valores propios
    vect_val_pro.sort()

    return vect_val_pro, polinomio


def construir_matriz_T():
    """
    Construye la matriz tridiagonal simétrica T de 12x12
    """
    T = np.zeros((12, 12))

    # Diagonal principal
    diagonal = [-1, -3, 3, -2, -1, 0, 0, -3, -2, 0, 0, 0]

    # Sub/super diagonal
    subdiagonal = [4, 1, -2, 3, -3, 2, -2, -2, 1, 3,-1]

    # Llenar la matriz
    for i in range(12):
        T[i, i] = diagonal[i]

    for i in range(11):
        T[i, i + 1] = subdiagonal[i]
        T[i + 1, i] = subdiagonal[i]  # Simétrica

    return T

def main():
    T = construir_matriz_T()
    h = 0.1
    valores_propios, polinomio = calcular_valores_propios(T, h)

    return valores_propios, polinomio, T, h

valores_propios, polinomio, T, h = main()
print("\n" + "=" * 60)
print("RESULTADOS FINALES")
print("=" * 60)
print("Valores propios calculados:")
for i, val in enumerate(valores_propios):
    print(f"  λ_{i + 1:2d}: {val:.10f}")

# Verificación con numpy
print("\nVERIFICACIÓN CON NUMPY:")
valores_exactos = np.linalg.eigvals(T)
valores_exactos.sort()

for i, val in enumerate(valores_exactos):
    print(f"  λ_{i + 1:2d}: {val:.10f}")

print(f"\nDiferencia máxima: {np.max(np.abs(valores_propios - valores_exactos)):.2e}")