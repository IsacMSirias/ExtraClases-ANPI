
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
def build_H_pseudoinversa(m: int, l: int, force_m: bool = True, eps =0.0) -> np.ndarray:
    
    

    # priemro se calcula el tamano extendidos y el num de bloques
    
    n = get_H_size(m,l, force_m)
    p = n // l
    
    # x[k] = -(m - l*(k-1) - 1)/p
    # y[k] =  (m - l*(k-1))/p
    # z = 1/p  

    x = np.array([-(m - l*(k-1) - 1)/p for k in range(1, p)], dtype=float)
    y = np.array([(m - l*(k-1))/p for k in range(1, p+1)], dtype=float)
    z = 1.0 / p
    
    #se inicia la matriz vacia de tamano nxm
    
    H_dag =np.zeros ((n,m) , dtype=np.float64)
    
    # rellenamosla matriz
       
    for i in range(n):
        for j in range(m):
            qi, ri = divmod(i, l)  # bloque y posición dentro del bloque (fila)
            qj, rj = divmod(j, l)  # bloque y posición dentro del bloque (columna)
            
            # Ahora bien,segun el caso teorico se tienen los siguientes casos:
            
            # CASO 1: i <= j, ambos están en posiciones de inicio de bloque (r=0)
            # Primer término de cada bloque en la zona superior izquierda.
            # Usa la secuencia y[qj]
            if i <= j and rj == 0 and ri == 0:
                H_dag[i, j] = y[qj] if qj < len(y) else 0           

            # CASO 2: i <= j, fila en fin de bloque y columna en inicio (rj=0, ri=l-1)
            # Transición entre bloques consecutivos (zona diagonal superior).
            # Usa z + x[qj-1]
   
            elif i <= j and rj == 0 and ri == l - 1:
                H_dag[i, j] = z + x[qj - 1] if qj - 1 < len(x) and qj - 1 >= 0 else z

            
            # CASO 3: i <= j, rj != 0, dentro de un bloque superior.
            # Ajusta la contribución interna dentro del mismo bloque.
            # Usa signo alternado (-1)^(d+1) * x[qj-1]
            
            elif i <= j and rj != 0 and (ri - j == 0 or ri - j == l - 1):
                d = 0 if (ri - j == 0) else 1
                H_dag[i, j] = ((-1)**(d+1)) * x[qj - 1] if qj - 1 < len(x) and qj - 1 >= 0 else 0

            
            # CASO 4: i >= l y i > j, ambos en posiciones de inicio de bloque (r=0)
            # Región inferior izquierda (bloques posteriores).
            # Usa z + x[p - qj - 1]
            
            elif i >= l and i > j and rj == 0 and ri == 0:
                H_dag[i, j] = z + x[p - qj - 1] if (p - qj - 1) < len(x) and (p - qj - 1) >= 0 else z

            
            # CASO 5: i >= l y i > j, fila en fin de bloque (ri=l-1) y columna en inicio (rj=0)
            # Zona diagonal inferior (transición entre bloques bajos).
            # Usa y[p - qj - 1]
            
            elif i >= l and i > j and rj == 0 and ri == l - 1:
                H_dag[i, j] = y[p - qj - 1] if (p - qj - 1) < len(y) and (p - qj - 1) >= 0 else 0

            # CASO 6: i >= l y i > j, dentro de bloque inferior (rj ≠ 0).
            # Contribución alternada dentro de bloques posteriores.
            # Usa (-1)^d * x[p - qj - 2]
            elif i >= l and i > j and rj != 0 and (ri - j == 0 or ri - j == l - 1):
                d = 0 if (ri - j == 0) else 1
                H_dag[i, j] = ((-1)**d) * x[p - qj - 2] if (p - qj - 2) < len(x) and (p - qj - 2) >= 0 else 0

            # CASO 7: rj == 0 y ri está dentro del bloque (ni al inicio ni al fin)
            # Zona intermedia vertical del bloque (contribución constante z)
            elif rj == 0 and ri != 0 and ri != l - 1:
                H_dag[i, j] = z

            # CASO 8: todos los demás (no encajan en ningún patrón anterior)
            # Valor nulo
            else:
                H_dag[i, j] = 0.0

    return H_dag




print (build_H_pseudoinversa(50,5))