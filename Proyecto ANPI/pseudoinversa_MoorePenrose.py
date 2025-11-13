
import numpy as np
from skimage import io, color
from skimage.util import img_as_float
from matplotlib import pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent      # carpeta donde está este script
IMG_DIR = BASE_DIR / "imagenes"                # subcarpeta
path = IMG_DIR / "rdr.jpg"

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
    n = get_H_size(m, l, force_m)
    H = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        H[i, i:i + l] = 1.0 / l
    return H



"""
Build H pseudoinversa


    Construye H† usando la definición de Moore–Penrose:
        H† = H^T (H H^T)^{-1}

    eps permite una pequeña regularización (Tikhonov) si hiciera falta:
        H† = H^T (H H^T + eps I)^{-1}    """

"""
def build_H_pseudoinversa(m: int, l: int, force_m: bool = True, eps: float = 0.0) -> np.ndarray:

   # 1. Construir H (m x n)
    H = build_H(m, l, force_m=force_m)
    m_rows, n_cols = H.shape  # m_rows = m, n_cols = n

    # 2. Formar A = H H^T  (m x m)
    A = H @ H.T

    # 3. Regularización opcional: A + eps I
    if eps > 0.0:
        A = A + eps * np.eye(m_rows, dtype=A.dtype)

    # 4. Calcular A^{-1} resolviendo A X = I  (evitar inv explícita)
    I = np.eye(m_rows, dtype=A.dtype)
    A_inv = np.linalg.solve(A, I)   # A_inv = A^{-1}, shape (m x m)

    # 5. H^+ = H^T A^{-1}  → (n x m)
    H_dag = H.T @ A_inv

    return H_dag


m, l = 8, 3
H     = build_H(m, l, force_m=True)
H_dag = build_H_pseudoinversa(m, l, force_m=True)

print("H shape:", H.shape)         # (8, 12)
print("H^+ shape:", H_dag.shape)   # (12, 8) ← esto es lo correcto

# 1) HH^+H = H
test1 = np.allclose(H @ H_dag @ H, H, atol=1e-10)
print("HH^+H = H ?", test1)

# 2) (H^+H)^T = (H^+H)
A = H_dag @ H
test2 = np.allclose(A.T, A, atol=1e-10)
print("(H^+H)^T = (H^+H) ?", test2)"""


# Con las correcciones del profe, se calcula la matriz pseudoinversa con Newton Schulz 

"""
    Método de Newton–Schulz para aproximar la pseudoinversa de A.

    A : matriz (m x n)
    tol : tolerancia sobre ||A Yk A - A||_F
    iterMax : máximo de iteraciones

    Retorna
    -------
    Yk : aproximación de A⁺ (n x m)
    k  : número de iteraciones realizadas
    er : error final ||A Yk A - A||_F
"""

def newton_schulz_pseudoinversa(A: np.ndarray,tol: float = 1e-10, iterMax: int = 100):

    m, n = A.shape

    # Vector inicial Y0 = A^T / ||A||_F^2
    Yk = (1.0 / np.linalg.norm(A, 'fro')**2) * A.T   # (n x m)

    Im = np.eye(m)   # identidad del tamaño de A A^+ (m x m)

    er = np.inf
    for k in range(iterMax):
        # Iteración: Y_{k+1} = Y_k (2I - A Y_k)
        Yk = Yk @ (2 * Im - A @ Yk)

        # Error de pseudoinversa: ||A Yk A - A||_F
        er = np.linalg.norm(A @ Yk @ A - A, 'fro')

        if er < tol:
            break

    return Yk, k + 1, er


"""
Construye una aproximación de H^+ mediante Newton–Schulz.

m : número de columnas de la imagen borrosa G
l : longitud del desenfoque
force_m : usa get_H_size para que n sea múltiplo de l
tol, iterMax : parámetros de Newton–Schulz

Retorna
-------
H_dag_NS : aproximación de la pseudoinversa de Moore–Penrose de H,
            de tamaño (n x m).
"""

def build_H_pseudoinversa_NS(m: int, l: int,force_m: bool = True,tol: float = 1e-10, iterMax: int = 100) -> np.ndarray:

    # Construimos H 
    H = build_H(m, l, force_m=force_m)   # H es (m x n)

    H_dag_NS, it, er = newton_schulz_pseudoinversa(H, tol=tol, iterMax=iterMax)

    #print(f"iteraciones = {it}, error = {er:.3e}")

    return H_dag_NS



"""

Test de pseudoinversa con N-S
m, l = 8, 3
H = build_H(m, l, force_m=True)
print("H shape:", H.shape)


# 2) Pseudoinversa por NEWTON–SCHULZ
H_dag_NS = build_H_pseudoinversa_NS(m, l, force_m=True,
                                    tol=1e-10, iterMax=200)
print("H^+ NS shape:", H_dag_NS.shape)

test1_NS = np.allclose(H @ H_dag_NS @ H, H, atol=1e-6)
A_NS = H_dag_NS @ H
test2_NS = np.allclose(A_NS.T, A_NS, atol=1e-6)

print("NS -> HH^+H = H ?", test1_NS)
print("NS -> (H^+H)^T = (H^+H) ?", test2_NS)

"""