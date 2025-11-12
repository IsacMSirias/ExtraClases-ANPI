
import numpy as np
from skimage import io, color, img_as_float
from matplotlib import pyplot as plt

"""
Loader:

Carga una imagen desde disco y la convierte en la matriz G
    usada en el método MP.

    - Si la imagen es RGB o RGBA, la convierte a escala de grises.
    - Normaliza los valores en el rango [0, 1].
    - Devuelve una matriz NumPy 2D de tipo float6

"""

def load_image(path: str) -> np.ndarray:
    # Lee la imagen (preferiblemente jpg o png)
    
    img =  io.imread(path)
    
    # si se tiene 2 o 3 canales, convertir a escala de grises
    if img.ndim == 3:
        img = color.rgb2gray(img)
        
    # Convertir a float y normalizar en [0,1]
    G = img_as_float(img).astype(np.float64)
    
    return G


path = "C:\\Users\\isacm\\Desktop\\TEC\\IIS 2025\\ANPI\\ExtraClases\\ExtraClases-ANPI\\Proyecto ANPI\\imagenes\\rdr.jpg"

"""
c
G = load_image(path)

print("Forma:", G.shape)
print("Tipo de datos:", G.dtype)
print("Rango de valores:", float(G.min()), "->", float(G.max()))

plt.imshow(G, cmap="gray")
plt.axis("off")
plt.show()
"""

"""
Ger H Size:
    Calcula n = m+1 -1 : si force_m es True, el metodo ajusta n al multiplo 1 (segun la formula teorica)
    
"""
def get_H_size(m: int, l: int, force_m: bool = True) -> int:
    
    if not isinstance(m, int) or not isinstance(l, int):
        raise TypeError("m y l deben ser enteros")
    if m <= 0 or l <= 0:
        raise ValueError("m y l deben ser positivos")

    n = m + l - 1
    if force_m and (n % l != 0):
        n = ((n // l) + 1) * l
    return n

"""
Build H 
Construye la matriz de desenfoque H (Toeplitz no simétrica) para blur horizontal uniforme.

H es de tamaño (m x n), con n = m + l - 1 (o redondeado a múltiplo de l si force_multiple=True).
Cada fila i de H tiene un bloque contiguo de longitud l con valor 1/l, desplazado una columna a la derecha por fila.

Parámetros
----------
m : int
    Número de columnas de la imagen borrosa G (ancho).
l : int
    Longitud del desenfoque (píxeles).
force_multiple : bool
    Si True, fuerza n a ser múltiplo de l (útil para Sección 3 del paper).

Retorna
-------
H : np.ndarray, shape (m, n), dtype float64
"""

def build_H(m: int, l: int, force_m: bool = True) -> np.ndarray:
    # H de tamaño (m x n) con bloques de 1/l desplazados
    n = m + l - 1
    if force_m and (n % l != 0):
        n = ((n // l) + 1) * l
    H = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        H[i, i:i+l] = 1.0 / l
    return H



"""
Build H pseudoinversa

    Construye la pseudoinversa de MoorePenrose (H†) para la
    matriz de desenfoque lineal uniforme H
    
        Parámetros
    ----------
    m : int
        Número de columnas de la imagen borrosa G.
    l : int
        Longitud del desenfoque (píxeles).
    force_m : bool
        Si True, ajusta n al múltiplo de l (requerido por la teoría).

    Retorna
    -------
    H_dag : np.ndarray
        Matriz H† de tamaño (n × m), tipo float64.


"""

def _safe_x(x, k):
    if 1 <= k <= len(x):
        return x[k-1]
    return 0.0
def _safe_y(y, k):
    if 1 <= k <= len(y):
        return y[k-1]
    return 0.0

def build_H_pseudoinversa(m: int, l: int, force_m: bool = True) -> np.ndarray:
    """
    H† por la fórmula (12) 

    Devuelve: H† de tamaño (n x m), con n = get_H_size(m,l,force_m).
    """
    # n = m + l - 1 y, si force_m=True, se ajusta al múltiplo de l (n = l*p)
    n = get_H_size(m, l, force_m)
    if n % l != 0:
        raise ValueError(" se requiere n múltiplo de l.")
    p = n // l  # número de bloques (definición del paper)

    # Secuencias del paper (x_k, y_k) y z=1/p  [k=1..p-1] y [k=1..p]
    # (ver definición exacta en el paper)
    # x_k = -(m - l(k-1) - 1)/p, k=1..p-1
    # y_k =  (m - l(k-1))    /p, k=1..p
    x = np.array([-(m - l*(k-1) - 1)/p for k in range(1, p    )], dtype=np.float64)
    y = np.array([ (m - l*(k-1)    )/p for k in range(1, p + 1)], dtype=np.float64)
    z = 1.0 / p

    # Matriz H†
    Hdag = np.zeros((n, m), dtype=np.float64)

    # Recorremos i=1..n, j=1..m en notación 1-based (como en (12))
    for i1 in range(1, n+1):
        qi = (i1-1) // l       # qi en 0..p-1
        ri = (i1-1) %  l + 1   # ri en 1..l

        for j1 in range(1, m+1):
            qj = (j1-1) // l   # qj en 0..p-1
            rj = (j1-1) %  l + 1  # rj en 1..l

            val = 0.0

            # d en {0,1} como en el paper: d=0 si (ri - rj == 0), d=1 en otro caso
            # (aparece en los casos con signo alternado)
            d = 0 if (ri - rj) == 0 else 1

            # Casos "superiores" (i <= j) pertenecen a bloques B_k
            if i1 <= j1 and rj == 1 and ri == 1:
                # y_{qj+1}
                val = _safe_y(y, qj+1)

            elif i1 <= j1 and rj == 1 and ri == l:
                # z + x_{qj}
                val = z + _safe_x(x, qj)

            elif i1 <= j1 and rj != 1 and (qj >= qi) and ((ri - rj == 0) or (ri - rj == l - 1)):
                # (-1)^{d+1} * x_{qj}
                val = ((-1.0)**(d+1)) * _safe_x(x, qj)

            # Casos "inferiores" (i >= l y i > j) pertenecen a bloques C_k
            elif i1 >= l and i1 > j1 and rj == 1 and ri == 1:
                # z + x_{p - qj - 1}
                val = z + _safe_x(x, p - qj - 1)

            elif i1 >= l and i1 > j1 and rj == 1 and ri == l:
                # y_{p - qj}
                val = _safe_y(y, p - qj)

            elif i1 >= l and i1 > j1 and rj != 1 and (qj <= qi) and ((ri - rj == 0) or (ri - rj == l - 1)):
                # (-1)^d * x_{p - qj - 1}
                val = ((-1.0)**d) * _safe_x(x, p - qj - 1)

            # Capa intermedia vertical: rj == 1 y ri no es ni inicio ni fin
            elif rj == 1 and (ri != 1) and (ri != l):
                # z
                val = z

            # else: 0 (ya inicializado)
            Hdag[i1-1, j1-1] = val
        

    return Hdag



m, l = 8, 3
H     = build_H(m, l, force_m=True)
H_dag = build_H_pseudoinversa(m, l, force_m=True)

print("H shape:", H.shape)        # (m, n)
print("H^+ shape:", H_dag.shape)  # (n, m)

# 1) HH^+H = H
test1 = np.allclose(H @ H_dag @ H, H, atol=1e-10)
print("HH^+H = H ?", test1)

# 2) (H^+H)^T = (H^+H)
A = H_dag @ H
test2 = np.allclose(A.T, A, atol=1e-10)
print("(H^+H)^T = (H^+H) ?", test2)