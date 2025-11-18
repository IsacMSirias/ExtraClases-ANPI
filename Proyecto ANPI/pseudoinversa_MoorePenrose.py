import numpy as np
from skimage import io, color
from skimage.util import img_as_float
from matplotlib import pyplot as plt
from pathlib import Path
import time


BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "imagenes"
PATH_IMG = IMG_DIR / "rdr.jpg"   


def load_image_gray(path: str) -> np.ndarray:
    """Carga la imagen y la pasa a escala de grises [0,1]."""
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return img_as_float(img).astype(np.float64)


def load_image_color(path: str) -> np.ndarray:
    """Carga la imagen en color solo para mostrarla."""
    img = io.imread(path)
    return img


def build_H(m: int, l: int) -> np.ndarray:
    """
    Construye H (m x n) para blur horizontal uniforme.
    n = m + l - 1.
    """
    n = m + l - 1
    H = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        H[i, i:i + l] = 1.0 / l
    return H


def newton_schulz_pseudoinversa(A: np.ndarray, tol: float = 1e-10, iterMax: int = 1000):
    
    t0 = time.time()
    
    m , n  = A.shape
    Yk = (1.0 / np.linalg.norm(A, 'fro')**2) * A.T  #(n x m)
    Im = np.eye(m)
    er = np.inf
    for k in range(iterMax):
        Yk = Yk @ (2 * Im - A @ Yk)   #(n x m)
        er = np.linalg.norm(A @ Yk @ A - A, 'fro') / np.linalg.norm(A, 'fro')
        if er < tol:
            break
        
    t1 = time.time()
    t_ejecucion = t1- t0
    print(f"[NS] iteraciones={k+1}, error_rel={er:.3e}, timepo = {t_ejecucion:.6f} s")
    return Yk


# 3. Blur sintético y restauración MP
def blur_image(F_true: np.ndarray, l: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica blur sintético: G = F_true * H.
    Devuelve G, H y H^+.
    """
    r, m = F_true.shape
    H = build_H(m, l) #(m x n)
    H_dag = newton_schulz_pseudoinversa(H) #(n x m)
    G = F_true @ H #(r x n) imagen desenfocada
    return G, H, H_dag

"""
Aplica blur sintético y restauración MP canal por canal (R,G,B).

Parámetros
----------
F_color : np.ndarray
    Imagen original en color, shape (r, m, 3), valores en [0,1].
l : int
    Longitud del blur horizontal.

Retorna
-------
G_blur_color : np.ndarray
    Imagen borrosa sintética en color (r, n, 3).
F_rest_color : np.ndarray
    Imagen restaurada en color (r, m, 3).
    """
def blur_and_restore_color(F_color: np.ndarray, l: int):

    # Asegurar tipo float64 en [0,1]
    F_color = img_as_float(F_color).astype(np.float64)

    r, m, c = F_color.shape
    assert c == 3, "Se espera una imagen RGB (3 canales)."

    # Construir H y su pseudoinversa UNA sola vez
    H = build_H(m, l) #(m x n)
    H_dag = newton_schulz_pseudoinversa(H) #(n x m)

    n = H.shape[1]  # ancho de la imagen borrosa

    # Inicializar matrices para blur y restauración en color
    G_blur_color = np.zeros((r, n, 3), dtype=np.float64)
    F_rest_color = np.zeros((r, m, 3), dtype=np.float64)

    # Procesar cada canal por separado
    for ch in range(3):
        F_ch = F_color[:, :, ch] #(r x m)
        G_ch = F_ch @ H # (r x n)  blur
        F_rest_ch = G_ch @ H_dag #(r x m) restaurada

        G_blur_color[:, :, ch] = G_ch
        F_rest_color[:, :, ch] = F_rest_ch

    # Clampear a [0,1] por seguridad
    F_rest_color = np.clip(F_rest_color, 0.0, 1.0)
    G_blur_color = np.clip(G_blur_color, 0.0, 1.0)

    return G_blur_color, F_rest_color

def MP_restore_from_blur(G: np.ndarray, H_dag: np.ndarray) -> np.ndarray:
    """
    Restaura: F_rest = G * H^+.
    G: (r x n)
    H_dag: (n x m)
    F_rest: (r x m)
    """
    return G @ H_dag


def visualizar_blur_y_restaurada_color(G_blur_color: np.ndarray, F_rest_color: np.ndarray, titulo: str = "Blur vs Restaurada (color)"):
    """
    Muestra solo:
       Imagen borrosa sintética en color
       Imagen restaurada en color
    """
    plt.figure(figsize=(12, 5))
    plt.suptitle(titulo, fontsize=16)

    # 1. Blur sintético
    plt.subplot(1, 2, 1)
    plt.title("Imagen Borrosa (blur sintético)")
    plt.imshow(G_blur_color)
    plt.axis("off")

    # 2. Restaurada
    plt.subplot(1, 2, 2)
    plt.title("Imagen Restaurada (MP)")
    plt.imshow(F_rest_color)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def demo_row_restoration(F_row, l=7):
    """
    Demostración del modelo matricial:
        G = F H^T
        F_rec = G H^+
    Para una sola fila.
    """
    m = len(F_row)

    # Construcción de H y su pseudoinversa
    H = build_H(m, l)
    
    H_dag = newton_schulz_pseudoinversa(H)

    # Aplicar blur y reconstrucción
    G_row = H.T @ F_row
    F_rec = G_row @ H_dag  #(1 × m)

    # Graficar
    plt.figure(figsize=(10, 4))
    plt.plot(F_row, label="F original", color="red")
    plt.plot(G_row[:m], label="G (blur)", color="blue")
    plt.plot(F_rec, label="F reconstruida", color="green")
    plt.title("Modelo FHᵀ y Restauración F̃ = GH⁺ (por una fila)")
    plt.xlabel("Índice")
    plt.ylabel("Intensidad")
    plt.legend()
    plt.grid()
    plt.show()





if __name__ == "__main__":
    # Cargar original en color
    F_color = load_image_color(str(PATH_IMG)) #(r, m, 3)

    # Parámetro de blur
    l = 7 # puedes jugar con 3, 5, 7...

    # Blur sintético y restauración EN COLOR
    G_blur_color, F_rest_color = blur_and_restore_color(F_color, l)

    print("F_color shape:", F_color.shape)
    print("G_blur_color shape:", G_blur_color.shape)
    print("F_rest_color shape:", F_rest_color.shape)

    # Visualizar SOLO blur y restaurada (ambas en color)
    visualizar_blur_y_restaurada_color(G_blur_color, F_rest_color, titulo=f"MP + Newton–Schulz (color, l={l})")

    # Tomar una sola fila de la imagen (por ejemplo la fila central)
    fila = F_color.shape[0] // 2
    F_row = color.rgb2gray(F_color)[fila, :]

    demo_row_restoration(F_row, l)
