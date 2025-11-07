import numpy as np
import sympy as sp
from matplotlib import pyplot as plt


def f(x):
    return np.log(np.arcsin(x)) / np.log(x)

def newtonSimbolico(x,y,n):

    tabla=np.zeros([(n+1),(n+1)],dtype=float)

    for i in range(n+1):
        tabla[i,0] = y[i]

    for j in range(1,n+1):
        for i in range((n+1)-j):
            tabla[i,j]=(tabla[i+1,j-1]-tabla[i,j-1])/(x[i+j]-x[i])


    #Paso 2
    X = sp.symbols('X')
    polinomio=tabla[0,0]
    termino = 1

    for k in range(1,n+1):

        termino = termino*(X-x[k-1])
        polinomio = polinomio + tabla[0,k]*termino

    polinomio=sp.expand(polinomio)

    return polinomio

def main():
    x= np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    y= np.array([f(0.1),f(0.2),f(0.3),f(0.4),f(0.5),f(0.6),f(0.7),f(0.8)])
    n=7
    return x,y,newtonSimbolico(x,y,n)

xv,yv,p=main()
print(p)

X = sp.symbols('X')
P= sp.lambdify(X,p,'numpy')
print("Puntos del polinomio",P(xv))

print("Puntos x:", xv)
print("Puntos y:", yv)

# Gráfica de la solución exacta
x_exact = np.linspace(0.1, 0.8, 1000)  # Equivalente a a:0.0001:b
y_exact = f(x_exact)

y_polinomio = P(x_exact)

# Crear la gráfica
plt.figure(figsize=(10, 6))

# Graficar solución exacta (línea azul)
plt.plot(x_exact, y_exact, 'b', label='Función f(x)', linewidth=2)

plt.plot(x_exact, y_polinomio, 'g', label='Polinomio', linewidth=2)

# Graficar solución aproximada (stem plot rojo)
plt.scatter(xv, yv, color= 'red', label='Puntos dados')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolación por diferencias definidas')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

