import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial
sp.init_printing()
plt.style.use('seaborn-v0_8')

# Definir variable y función
x = sp.symbols('x')
f_expr = (x ** 3 - 3 * x ** 2 + 3 * x - 1) / (x ** 2 - 2 * x)
f_expr = sp.simplify(f_expr)

print("=" * 60)
print("ANÁLISIS DE LA FUNCIÓN")
print("=" * 60)
print(f"f(x) = {sp.pretty(f_expr)}")
print()


def analizar_funcion(f):
    """Analiza completamente una función racional"""

    # (a) Dominio de la función
    dominio = sp.calculus.util.continuous_domain(f, x, sp.S.Reals)

    # Factorizar numerador y denominador
    num, den = sp.fraction(sp.together(f))
    num_factorizado = sp.factor(num)
    den_factorizado = sp.factor(den)

    # (b) Intersecciones
    # Intersecciones con eje x (raíces del numerador, excluyendo puntos no definidos)
    x_intersecciones = []
    raices_num = sp.solve(sp.Eq(num, 0), x)
    for raiz in raices_num:
        if den.subs(x, raiz) != 0:  # Verificar que el punto esté en el dominio
            x_intersecciones.append(raiz)

    # Intersección con eje y
    y_interseccion = None
    if 0 in dominio:  # Verificar si x=0 está en el dominio
        y_interseccion = f.subs(x, 0)

    # (c) Asíntotas
    # Asíntotas verticales (raíces del denominador)
    asint_verticales = sp.solve(sp.Eq(den, 0), x)

    # Asíntotas horizontales/oblicuas
    grado_num = sp.degree(num, x)
    grado_den = sp.degree(den, x)

    asint_horizontal = None
    asint_oblicua = None

    if grado_num < grado_den:
        asint_horizontal = 0
    elif grado_num == grado_den:
        coef_num = sp.LC(num, x)
        coef_den = sp.LC(den, x)
        asint_horizontal = coef_num / coef_den
    else:  # grado_num > grado_den
        cociente, resto = sp.div(num, den)
        asint_oblicua = cociente

    # (d) Derivadas
    f_prima = sp.simplify(sp.diff(f, x))
    f_segunda = sp.simplify(sp.diff(f_prima, x))

    return {
        'dominio': dominio,
        'x_intersecciones': x_intersecciones,
        'y_interseccion': y_interseccion,
        'asint_verticales': asint_verticales,
        'asint_horizontal': asint_horizontal,
        'asint_oblicua': asint_oblicua,
        'f_prima': f_prima,
        'f_segunda': f_segunda,
        'puntos_indefinidos': asint_verticales
    }


def analizar_monotonia(f_prima, puntos_indefinidos):
    """Analiza intervalos de crecimiento y decrecimiento"""
    # Encontrar puntos críticos (donde f'(x) = 0 o no existe)
    puntos_criticos = set()

    # Puntos donde f'(x) = 0
    try:
        crit_numerador = sp.numer(sp.together(f_prima))
        raices_criticas = sp.solve(sp.Eq(crit_numerador, 0), x)
        puntos_criticos.update(raices_criticas)
    except:
        pass

    # Puntos donde f'(x) no existe (asintotas verticales)
    puntos_criticos.update(puntos_indefinidos)

    # Convertir a números reales y ordenar
    puntos_reales = sorted([float(p) for p in puntos_criticos if p.is_real])

    # Crear intervalos de prueba
    intervalos = []
    if puntos_reales:
        intervalos.append((-np.inf, puntos_reales[0]))
        for i in range(len(puntos_reales) - 1):
            intervalos.append((puntos_reales[i], puntos_reales[i + 1]))
        intervalos.append((puntos_reales[-1], np.inf))
    else:
        intervalos = [(-np.inf, np.inf)]

    # Evaluar signo de f' en cada intervalo
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
            valor = f_prima.subs(x, punto_prueba)
            if valor > 0:
                creciente.append((a, b))
            elif valor < 0:
                decreciente.append((a, b))
        except:
            continue

    return creciente, decreciente


def analizar_concavidad(f_segunda, puntos_indefinidos):
    """Analiza concavidad de la función"""
    # Similar a analizar_monotonia pero con f''
    return analizar_monotonia(f_segunda, puntos_indefinidos)


# Realizar análisis completo
resultados = analizar_funcion(f_expr)
creciente, decreciente = analizar_monotonia(resultados['f_prima'], resultados['puntos_indefinidos'])
concava_arriba, concava_abajo = analizar_concavidad(resultados['f_segunda'], resultados['puntos_indefinidos'])

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
print(f"   Creciente en: {creciente}")
print(f"   Decreciente en: {decreciente}")
print()

print("(g) CONCAVIDAD:")
print(f"   Cóncava hacia arriba en: {concava_arriba}")
print(f"   Cóncava hacia abajo en: {concava_abajo}")
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

print("¡Análisis completado!")