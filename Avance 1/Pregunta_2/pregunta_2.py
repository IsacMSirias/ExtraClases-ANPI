import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def analisis_completo(f):
    """
    Realiza un análisis completo de una función racional.

    Parámetros:
        f : expresión simbólica de SymPy
            Función a analizar de la forma P(x)/Q(x)

    Retorna:
        Diccionario con todos los resultados del análisis que incluye:
            - función simplificada
            - dominio
            - intersecciones con ejes
            - asíntotas
            - derivadas
            - intervalos de monotonia
            - intervalos de concavidad
    """

    f_simp = sp.simplify(f)

    num, den = sp.fraction(sp.together(f_simp))
    numfactor, denfactor = sp.factor(num), sp.factor(den)


    dominio = dominio_funcion(f)
    x_int, y_int = intersecciones_funcion(f, numfactor, denfactor, dominio)
    asint_vert, asint_horiz, asint_oblicua = asintotas_funcion(numfactor, denfactor)
    f_prima, f_segunda = derivadas_funcion(f)

    puntos_indefinidos = asint_vert

    # Análisis de monotonia y concavidad
    creciente, decreciente = analizar_monotonia(f_prima, puntos_indefinidos)
    concava_arriba, concava_abajo = analizar_concavidad(f_segunda, puntos_indefinidos)

    return {
        'funcion': f_expr,
        'dominio': dominio,
        'x_intersecciones': x_int,
        'y_interseccion': y_int,
        'asint_verticales': asint_vert,
        'asint_horizontal': asint_horiz,
        'asint_oblicua': asint_oblicua,
        'f_prima': f_prima,
        'f_segunda': f_segunda,
        'creciente': creciente,
        'decreciente': decreciente,
        'concava_arriba': concava_arriba,
        'concava_abajo': concava_abajo,
    }


def dominio_funcion(f):
    """
    Determina el dominio de una función.

    Parámetros:
        f : expresión simbólica de SymPy
            Función cuyo dominio se quiere encontrar

    Retorna:
        Conjunto de números reales donde la función es continua
    """
    return sp.calculus.util.continuous_domain(f, x, sp.S.Reals)

def intersecciones_funcion(f, num, den, dominio):
    """
    Encuentra las intersecciones de la función con los ejes coordenados.

    Parámetros:
         f : expresión simbólica
            Función a analizar
         num : expresión simbólica
             Numerador de la función simplificada
         den : expresión simbólica
             Denominador de la función simplificada
         dominio : set
             Dominio de la función

    Retorna:
        tuple: (intersecciones_x, interseccion_y) donde:
        intersecciones_x: lista de puntos de corte con el eje X
        interseccion_y: punto de corte con el eje Y o None si no existe
    """

    x_intersecciones = []

    try:
        raices_num = sp.solve(sp.Eq(num, 0), x)
        for raiz in raices_num:
            if den.subs(x, raiz) != 0:
                x_intersecciones.append(raiz)
    except:
        pass
    y_interseccion = None
    try:
        if 0 in dominio:
            y_interseccion= f.subs(x, 0)
    except:
        pass
    return x_intersecciones, y_interseccion

def asintotas_funcion(num, den):
    """
    Determina las asíntotas de una función racional.

    Parámetros:
        num : expresión simbólica
            Numerador de la función
        den : expresión simbólica
            Denominador de la función

    Retorna:
        tuple: (asint_verticales, asint_horizontal, asint_oblicua) donde:
            asint_verticales: lista de asíntotas verticales
            asint_horizontal: asíntota horizontal o None si no existe
            asint_oblicua: asíntota oblicua o None si no existe
    """

    asint_verticales = []
    try:
        asint_verticales = sp.solve(sp.Eq(den, 0), x)
    except:
        pass

    asint_oblicua = None
    asint_horizontal= None

    try:
        grado_num = sp.degree(num, x)
        grado_den = sp.degree(den, x)

        if grado_num < grado_den:
            asint_horizontal = 0
        elif grado_num == grado_den:
            asint_horizontal = sp.LC(num, x) / sp.LC(den, x)
        else:
            asint_oblicua = sp.div(num, den)[0]  # Cociente de la división
    except:
        pass

    return asint_verticales, asint_horizontal, asint_oblicua


def derivadas_funcion(f):
    """
    Calcula la primera y segunda derivada de una función.

    Parámetros:
        f : expresión simbólica
            Función a derivar

    Retorna:
        tuple: (f_prima, f_segunda) donde:
            f_prima: primera derivada de f
            f_segunda: segunda derivada de f
    """
    f_prima = sp.simplify(sp.diff(f, x))
    f_segunda = sp.simplify(sp.diff(f_prima, x))
    return f_prima, f_segunda

def analizar_signo(f,puntos_discontinuidad):
    """
    Analiza el signo de una función en intervalos definidos por puntos críticos.

    Parámetros:
        f : expresión simbólica
            Función cuyo signo se quiere analizar
        puntos_discontinuidad : list
            Puntos donde la función no está definida

    Retorna:
        tuple: (intervalos_positivos, intervalos_negativos) donde:
            intervalos_positivos: lista de tuplas (a, b) donde f(x) > 0
            intervalos_negativos: lista de tuplas (a, b) donde f(x) < 0
    """

    puntos_cero = []
    try:
        numerador = sp.numer(sp.together(f))
        puntos_cero = sp.solve (sp.Eq(numerador, 0),x)
    except:
        pass

    # Combinar todos los puntos importantes
    puntos_importantes = set(puntos_cero + puntos_discontinuidad)
    puntos_reales = sorted([p for p in puntos_importantes if p.is_real])

    # Crear intervalos
    if not puntos_reales:
        return [(-sp.oo, sp.oo)], []

    intervalos = [(-sp.oo, puntos_reales[0])]
    for i in range(len(puntos_reales) - 1):
        intervalos.append((puntos_reales[i], puntos_reales[i + 1]))
    intervalos.append((puntos_reales[-1], sp.oo))

    # Analizar signo en cada intervalo
    creciente = []
    decreciente = []

    for a, b in intervalos:
        # Elegir punto de prueba en el intervalo
        if a == -np.inf and b == np.inf:
            punto_prueba = 0
        elif a == -np.inf:
            punto_prueba = b - 1
        elif b == np.inf:
            punto_prueba = a + 1
        else:
            punto_prueba = (a + b) / 2

        try:
            valor = f.subs(x, punto_prueba)
            if valor > 0:
                creciente.append((a, b))
            elif valor < 0:
                decreciente.append((a, b))
        except:
            continue

    return creciente, decreciente

def analizar_monotonia(f_prima, puntos_indefinidos):
    """
    Analiza los intervalos donde la función es creciente o decreciente.

    Parámetros:
        f_prima : expresión simbólica
            Primera derivada de la función
        puntos_indefinidos : list
            Puntos donde la función original no está definida

    Retorna:
        tuple: (intervalos_creciente, intervalos_decreciente) donde:
            intervalos_creciente: intervalos donde f'(x) > 0
            intervalos_decreciente: intervalos donde f'(x) < 0
    """
    return analizar_signo(f_prima, puntos_indefinidos)

def analizar_concavidad(f_segunda, puntos_indefinidos):
    """
    Analiza la concavidad de la función.

    Parámetros:
        f_segunda : expresión simbólica
            Segunda derivada de la función
        puntos_indefinidos : list
            Puntos donde la función original no está definida

    Retorna:
        tuple: (intervalos_concava_arriba, intervalos_concava_abajo) donde:
            intervalos_concava_arriba: intervalos donde f''(x) > 0
            intervalos_concava_abajo: intervalos donde f''(x) < 0
    """

    return analizar_signo(f_segunda, puntos_indefinidos)


# Definir variable y función
x = sp.symbols('x')
f_expr = (x ** 3 - 3 * x ** 2 + 3 * x - 1) / (x ** 2 - 2 * x)

# Realizar análisis completo
resultados = analisis_completo(f_expr)

print("=" * 60)
print("ANÁLISIS DE LA FUNCIÓN")
print("=" * 60)
print(f"f(x) = {sp.pretty(f_expr)}")
print()



# Mostrar resultados
print("(a) DOMINIO:")
print(f"   {resultados['dominio']}")
print()

print("(b) INTERSECCIONES:")
print(f"   Con eje x: {resultados['x_intersecciones']}")
print(f"   Con eje y: {resultados['y_interseccion'] if resultados['y_interseccion'] is not None else 'No existe'}")
print()

print("(c) ASÍNTOTAS:")
print(f"   Verticales: {resultados['asint_verticales']}")
print(f"   Horizontal: {resultados['asint_horizontal']}")
print(f"   Oblicua: {resultados['asint_oblicua']}")
print()

print("(d) DERIVADAS:")
print(f"   f'(x) = {sp.pretty(resultados['f_prima'])}")
print(f"   f''(x) = {sp.pretty(resultados['f_segunda'])}")
print()

print("(f) MONOTONÍA:")
print(f"   Creciente en: {resultados['creciente']}")
print(f"   Decreciente en: {resultados['decreciente']}")
print()

print("(g) CONCAVIDAD:")
print(f"   Cóncava hacia arriba en: {resultados['concava_arriba']}")
print(f"   Cóncava hacia abajo en: {resultados['concava_abajo']}")
print()

# (e) GRÁFICAS
print("(e) GENERANDO GRÁFICAS...")

# Convertir a funciones numéricas
f_num = sp.lambdify(x, f_expr, 'numpy')
f1_num = sp.lambdify(x, resultados['f_prima'], 'numpy')
f2_num = sp.lambdify(x, resultados['f_segunda'], 'numpy')

# Crear dominio para graficar (evitando asíntotas)
asintotas = [float(a) for a in resultados['asint_verticales'] if a.is_real]
x_min, x_max = -5, 5
segmentos = []

# Dividir el dominio en segmentos que eviten asíntotas
puntos_division = sorted([x_min] + asintotas + [x_max])
for i in range(len(puntos_division) - 1):
    a, b = puntos_division[i], puntos_division[i + 1]
    if b - a > 0.1:  # Solo crear segmentos significativos
        # Alejarse un poco de las asíntotas
        margen = 0.05
        seg_x = np.linspace(a + margen, b - margen, 1000)
        segmentos.append(seg_x)

# Crear figura con subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Graficar f(x)
for seg_x in segmentos:
    ax1.plot(seg_x, f_num(seg_x), 'b-', linewidth=2, label='f(x)' if seg_x is segmentos[0] else "")
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(0, color='black', linewidth=0.5)
# Marcar asíntotas verticales
for asintota in asintotas:
    ax1.axvline(asintota, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.set_title('Función f(x)')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Graficar f'(x)
for seg_x in segmentos:
    ax2.plot(seg_x, f1_num(seg_x), 'r-', linewidth=2, label="f'(x)" if seg_x is segmentos[0] else "")
ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.5)
for asintota in asintotas:
    ax2.axvline(asintota, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_title("Primera derivada f'(x)")
ax2.set_xlabel('x')
ax2.set_ylabel("f'(x)")
ax2.grid(True, alpha=0.3)
ax2.legend()

# Graficar f''(x)
for seg_x in segmentos:
    ax3.plot(seg_x, f2_num(seg_x), 'g-', linewidth=2, label="f''(x)" if seg_x is segmentos[0] else "")
ax3.axhline(0, color='black', linewidth=0.5)
ax3.axvline(0, color='black', linewidth=0.5)
for asintota in asintotas:
    ax3.axvline(asintota, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax3.set_title("Segunda derivada f''(x)")
ax3.set_xlabel('x')
ax3.set_ylabel("f''(x)")
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()