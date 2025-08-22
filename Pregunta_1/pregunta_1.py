import numpy as np
from scipy.linalg import solve 

# Entradas:
# m: int, m>=5 (numero mayor o igual a 5)
# a: vec(m,1)
# b: vec(m-1,1)
# c: vec(m-1,1)
# d: vec(m-1,2)
# e: vec(m-2,1)

# Salida: una matriz pentadiagonal de dimensiones (m,m)

def pentadiagonal(m, a, b, c, d, e):
    
    if m < 5:
        print("El valor de m debe ser mayor o igual a 5")
        return None
    else:
        A = np.zeros((m, m)) # se crea una matriz nula como base
        A[np.arange(m), np.arange(m)] = a # vec  m × 1
        A[np.arange(m-1), np.arange(1, m)] = b # vec  (m − 1) × 1
        A[np.arange(1, m), np.arange(m-1)] = c # Igual que el anterir
        A[np.arange(m-2), np.arange(2, m)] = d  # vec (m − 2) × 1.
        A[np.arange(2, m), np.arange(m-2)] = e # igual que el anterior
        
        # Lo anterior es basicamente la creacion de diagonales, esta juega con las pos (x,y)
        # con el fin de poder estar arriba o debajo de la diagonal principal.
    
        return A

#por si se gusta probar... Se puede hacer le la siguuiente manera:
# Pongo numeros random la matriz

# m= 15 

# a = np.random.randint(1, 10, size=m)      
# b = np.random.randint(1, 10, size=m-1)     
# c = np.random.randint(1, 10, size=m-1)     
# d = np.random.randint(1, 10, size=m-2)     
# e = np.random.randint(1, 10, size=m-2) 
# print(pentadiagonal(m, a, b, c, d, e))


# EN EL CASO ESPECIFICO DE LA PREGUNTA:

m = 2500

# creo que esto no hace falta explicarse tanto, pero por si acaso y por temas de la claridad en la evaluacion...
# Se sabe que tenemos arrays de distintas dimensiones y con distintas formas y todas se pueden hacer facuiilmente con un fors 
a = np.array([2*(i+1) for i in range(m)], dtype=float)  # 2(i+1),para i=0,1,...,m−1
b = np.array([(i+1)/3 for i in range(m-1)], dtype=float) # i+1/3​,i=0,1,...,m−2
c = np.array([i/3 for i in range(m-1)], dtype=float) # i/3​,i=0,1,...,m−2
d = np.array([(i+2)/4 for i in range(m-2)], dtype=float) # (i+2)/4​,i=0,1,...,m−3
e = np.array([i/4 for i in range(m-2)], dtype=float) # i/4​,i=0,1,...,m−3
h = np.array([2*i for i in range(m)], dtype=float) # 2i,i=0,1,...,m−1

#  Contruccion de la matriz y su sol
A = pentadiagonal(m, a, b, c, d, e)
x = solve(A, h)
# error
error = np.linalg.norm(A @ x - h, 2)
print("Error:", error)