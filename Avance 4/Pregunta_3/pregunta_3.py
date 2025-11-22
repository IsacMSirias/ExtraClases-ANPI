import sympy as sp
import scipy.optimize as opt

def cota_simpson_puntos(f,a,b,tol):
    x,n = sp.symbols('x,n')
    h = (b-a) / n
    fs= f(x)

    fds= sp.diff(fs,x,4)
    fdn= sp.lambdify(x,fds, 'numpy')

    # Calcular el máximo
    def newfun(x):
        return -abs(fdn(x))

    xmax = opt.fminbound(newfun, a, b)
    alphamax= abs(fdn(xmax))

    num=((b-a)*(h**4))
    ecuacion=sp.Eq((num/2880)*alphamax,tol)
    soluciones=sp.solve(ecuacion,n)

    sol = [float(s) for s in soluciones if s.is_real and s > 0]
    n = sol[0]

    return n

def f(x):
    return sp.exp(x)*(26-10*x+x**2)

a=5
b=5.55
tol=1e-8

n=cota_simpson_puntos(f,a,b,tol)

print("Valor de n:")
print(n)