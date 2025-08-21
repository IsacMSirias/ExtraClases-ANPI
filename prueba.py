import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Definir variable y función
x = sp.symbols('x')
f = (x**3 - 3*x**2 + 3*x - 1) / (x**2 - 2*x)
f = sp.simplify(f)

# (a) Dominio
domain = sp.calculus.util.continuous_domain(f, x, sp.S.Reals)

# (b) Intersecciones
num, den = sp.fraction(sp.together(f))
num = sp.factor(sp.simplify(num))
den = sp.factor(sp.simplify(den))

# Eje x
x_intersections = sp.solve(sp.Eq(num, 0), x)
x_intersections = [xi for xi in x_intersections if den.subs(x, xi) != 0]

# Eje y
y_intersection = None
if den.subs(x, 0) != 0:
    y_intersection = f.subs(x, 0)

# (c) Asíntotas
vertical_asymptotes = sp.solve(sp.Eq(den, 0), x)

q, r = sp.div(sp.expand(num), sp.expand(den))
deg_num = sp.degree(num)
deg_den = sp.degree(den)

horizontal_asymptote = None
oblique_asymptote = None
if deg_num < deg_den:
    horizontal_asymptote = 0
elif deg_num == deg_den:
    horizontal_asymptote = sp.LC(num) / sp.LC(den)
else:
    oblique_asymptote = q

# (d) Derivadas
f1 = sp.simplify(sp.diff(f, x))
f2 = sp.simplify(sp.diff(f1, x))

# (f) Intervalos crec/decr
crit_eq_num, _ = sp.fraction(sp.together(f1))
crit_roots = sp.nroots(crit_eq_num)
crit_real = sorted([float(r) for r in crit_roots if abs(sp.im(r)) < 1e-10])

undefined_points = [float(v) for v in vertical_asymptotes]
breaks = sorted(set(crit_real + undefined_points))

interval_edges = [-sp.oo] + breaks + [sp.oo]
inc_intervals, dec_intervals = [], []

def sign_on_interval(a, b):
    if a is -sp.oo and b is sp.oo: t = 0.0
    elif a is -sp.oo: t = b - 1.0
    elif b is sp.oo: t = a + 1.0
    else: t = (a + b) / 2.0
    return np.sign(float(f1.subs(x, t)))

for i in range(len(interval_edges)-1):
    a, b = interval_edges[i], interval_edges[i+1]
    left = -1e6 if a == -sp.oo else float(a)
    right = 1e6 if b == sp.oo else float(b)
    try:
        s = sign_on_interval(left, right)
        interval = sp.Interval.open(left, right)
        if s > 0:
            inc_intervals.append(interval)
        elif s < 0:
            dec_intervals.append(interval)
    except:
        pass

# (g) Concavidad
inflec_num, _ = sp.fraction(sp.together(f2))
inflec_roots = sp.nroots(inflec_num)
inflec_real = sorted([float(r) for r in inflec_roots if abs(sp.im(r)) < 1e-10])

breaks2 = sorted(set(inflec_real + undefined_points))
interval_edges2 = [-sp.oo] + breaks2 + [sp.oo]
conc_up, conc_down = [], []

def sign2_on_interval(a, b):
    if a is -sp.oo and b is sp.oo: t = 0.0
    elif a is -sp.oo: t = b - 1.0
    elif b is sp.oo: t = a + 1.0
    else: t = (a + b) / 2.0
    return np.sign(float(f2.subs(x, t)))

for i in range(len(interval_edges2)-1):
    a, b = interval_edges2[i], interval_edges2[i+1]
    left = -1e6 if a == -sp.oo else float(a)
    right = 1e6 if b == sp.oo else float(b)
    try:
        s = sign2_on_interval(left, right)
        interval = sp.Interval.open(left, right)
        if s > 0:
            conc_up.append(interval)
        elif s < 0:
            conc_down.append(interval)
    except:
        pass

# Mostrar resultados
print("(a) Dominio:", domain)
print("(b) Intersecciones en x:", x_intersections)
print("(b) Intersección en y:", y_intersection if y_intersection else "No existe")
print("(c) Asíntotas verticales:", vertical_asymptotes)
print("(c) Asíntota horizontal:", horizontal_asymptote)
print("(c) Asíntota oblicua:", oblique_asymptote)
print("(d) f'(x):", f1)
print("(d) f''(x):", f2)
print("(f) Intervalos crecientes:", inc_intervals)
print("(f) Intervalos decrecientes:", dec_intervals)
print("(g) Cóncava hacia arriba:", conc_up)
print("(g) Cóncava hacia abajo:", conc_down)

# =====================
# (e) Graficar f, f', f''
# =====================
f_num = sp.lambdify(x, f, "numpy")
f1_num = sp.lambdify(x, f1, "numpy")
f2_num = sp.lambdify(x, f2, "numpy")

# Evitar asíntotas en graficación
def piecewise_xranges(a, b, holes, n=1000, gap=0.02):
    holes = sorted([h for h in holes if a < h < b])
    cuts = [a] + holes + [b]
    segments = []
    for i in range(len(cuts)-1):
        left, right = cuts[i], cuts[i+1]
        if right - left <= 1e-9: continue
        segments.append(np.linspace(left + gap, right - gap, n))
    return segments

holes = [0.0, 2.0]
ranges = piecewise_xranges(-6, 6, holes)

# f(x)
plt.figure()
for seg in ranges:
    plt.plot(seg, f_num(seg), 'b')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title("f(x)")
plt.grid()

# f'(x)
plt.figure()
for seg in ranges:
    plt.plot(seg, f1_num(seg), 'r')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title("f'(x)")
plt.grid()

# f''(x)
plt.figure()
for seg in ranges:
    plt.plot(seg, f2_num(seg), 'g')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title("f''(x)")
plt.grid()

plt.show()
