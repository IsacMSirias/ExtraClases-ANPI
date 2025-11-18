import numpy as np
from skimage import io, color
from skimage.util import img_as_float
from matplotlib import pyplot as plt
from pathlib import Path
import time

# ============================================
# CONFIGURACIÓN DE RUTAS
# ============================================

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "imagenes"
PATH_IMG = IMG_DIR / "rdr.jpg"   


# ===============================================================
# MÉTODO 1 — CARGA DE IMAGEN EN ESCALA DE GRISES
# ===============================================================
def load_image_gray(path: str) -> np.ndarray:
    """
    Método: load_image_gray

    Objetivo:
        Cargar una imagen desde disco y convertirla a una matriz 2D
        en escala de grises con valores normalizados en [0, 1].

    Variables:
        path : ruta del archivo de imagen.
        img  : imagen cargada en formato RGB o escala de grises.
    
    Retorna:
        Imagen gris como arreglo float64 en el rango [0,1].
    """

    img = io.imread(path)

    # Si la imagen es RGB (3 canales), convertir a gris usando luminancia.
    if img.ndim == 3:
        img = color.rgb2gray(img)

    return img_as_float(img).astype(np.float64)


# ===============================================================
# MÉTODO 2 — CARGA DE IMAGEN EN COLOR
# ===============================================================
def load_image_color(path: str) -> np.ndarray:
    """
    Método: load_image_color

    Objetivo:
        Cargar una imagen en color (sin convertir a escala de grises)
        para visualización y comparación.

    Retorna:
        Imagen RGB como matriz (r, m, 3).
    """
    img = io.imread(path)
    return img


# ===============================================================
# MÉTODO 3 — CONSTRUCCIÓN DE LA MATRIZ DE DESENFOQUE H
# ===============================================================
def build_H(m: int, l: int) -> np.ndarray:
    """
    Método: build_H

    Objetivo:
        Construir la matriz de blur horizontal uniforme H, definida en el paper.
        Esta matriz representa un *promedio móvil* de longitud l.

    Variables:
        m : ancho de la imagen (número de columnas).
        l : longitud del desenfoque.
        n : tamaño del resultado: n = m + l - 1 (con padding implícito).
        H : matriz Toeplitz no simétrica que modela el blur.

    Retorna:
        H como matriz float64 de tamaño (m x n).
    """

    n = m + l - 1
    H = np.zeros((m, n), dtype=np.float64)

    # En cada fila se coloca un bloque de l entradas con valor 1/l
    for i in range(m):
        H[i, i:i + l] = 1.0 / l

    return H


# ===============================================================
# MÉTODO 4 — PSEUDOINVERSA DE H POR NEWTON–SCHULZ
# ===============================================================
def newton_schulz_pseudoinversa(A: np.ndarray, tol: float = 13e-10, iterMax: int = 1000):
    """
    Método: newton_schulz_pseudoinversa

    Objetivo:
        Aproximar la pseudoinversa de Moore–Penrose A⁺ de una matriz A,
        utilizando el algoritmo iterativo de Newton–Schulz.

    Fundamento matemático:
        Iteración:
            Y_{k+1} = Y_k ( 2I - A Y_k )
        Criterio de parada:
            ||A Y_k A - A||_F / ||A||_F < tol
        Condición inicial (paper):
            Y_0 = A^T / ||A||_F^2

    Variables:
        A        : matriz a invertir.
        tol      : tolerancia para el error relativo.
        iterMax  : número máximo de iteraciones.
        m, n     : dimensiones de A.
        Yk       : aproximación actual a la pseudoinversa.
        Im       : matriz identidad (m x m).
        er       : error relativo de Moore–Penrose.
        t_ejec   : tiempo total de cómputo.

    Retorna:
        Yk : pseudoinversa aproximada de A (dim n x m).
    """

    t0 = time.time()                     # Iniciar cronómetro
    m , n  = A.shape

    # Condición inicial estable
    Yk = (1.0 / np.linalg.norm(A, 'fro')**2) * A.T
    Im = np.eye(m)

    er = np.inf  # Error inicial grande

    for k in range(iterMax):

        # Paso iterativo de Newton–Schulz
        Yk = Yk @ (2 * Im - A @ Yk)

        # Error relativo ||AYA - A|| / ||A||
        er = np.linalg.norm(A @ Yk @ A - A, 'fro') / np.linalg.norm(A, 'fro')

        # Criterio de parada
        if er < tol:
            break

    t1 = time.time()
    t_ejecucion = t1 - t0

    print(f"[NS] iteraciones={k+1}, error_rel={er:.3e}, timepo = {t_ejecucion:.6f} s")

    return Yk


# ===============================================================
# MÉTODO 5 — CREACIÓN DE BLUR SINTÉTICO + RESTAURACIÓN
# ===============================================================
def blur_image(F_true: np.ndarray, l: int):
    """
    Método: blur_image

    Objetivo:
        Aplicar el desenfoque sintético que describe el modelo:
            G = F_true H

    Variables:
        F_true : imagen original (matriz r x m).
        l      : longitud del blur.
        H      : matriz Toeplitz del desenfoque.
        H_dag  : pseudoinversa de H calculada con Newton–Schulz.
        G      : imagen borrosa generada.

    Retorna:
        (G, H, H_dag)
    """

    r, m = F_true.shape
    H = build_H(m, l)
    H_dag = newton_schulz_pseudoinversa(H)
    G = F_true @ H
    return G, H, H_dag


# ===============================================================
# MÉTODO 6 — BLUR Y RESTAURACIÓN EN COLOR
# ===============================================================
def blur_and_restore_color(F_color: np.ndarray, l: int):
    """
    Método: blur_and_restore_color

    Objetivo:
        Aplicar blur sintético y restauración canal por canal (R, G, B):
            G_ch = F_ch H
            F_rest_ch = G_ch H⁺

    Variables:
        F_color      : imagen original RGB.
        H, H_dag     : matriz de blur y pseudoinversa (únicas para los 3 canales).
        G_blur_color : tensor con blur sintético en cada canal.
        F_rest_color : tensor restaurado canal por canal.

    Retorna:
        (G_blur_color, F_rest_color)
    """

    F_color = img_as_float(F_color).astype(np.float64)
    r, m, c = F_color.shape

    # Verificar que sea RGB
    assert c == 3, "Se espera una imagen RGB."

    # H y H⁺ solo se calculan una vez
    H = build_H(m, l)
    H_dag = newton_schulz_pseudoinversa(H)
    n = H.shape[1]

    G_blur_color = np.zeros((r, n, 3))
    F_rest_color = np.zeros((r, m, 3))

    for ch in range(3):
        F_ch = F_color[:, :, ch]
        G_ch = F_ch @ H
        F_rest_ch = G_ch @ H_dag

        G_blur_color[:, :, ch] = G_ch
        F_rest_color[:, :, ch] = F_rest_ch

    # Mantener valores en rango válido
    G_blur_color = np.clip(G_blur_color, 0, 1)
    F_rest_color = np.clip(F_rest_color, 0, 1)

    return G_blur_color, F_rest_color


# ===============================================================
# MÉTODO 7 — RESTAURACIÓN DIRECTA F̃ = GH⁺
# ===============================================================
def MP_restore_from_blur(G: np.ndarray, H_dag: np.ndarray) -> np.ndarray:
    """
    Método: MP_restore_from_blur

    Objetivo:
        Implementa directamente:
            F̃ = G H⁺

    Variables:
        G      : imagen borrosa (r x n).
        H_dag  : pseudoinversa (n x m).

    Retorna:
        F_rest (r x m)
    """
    return G @ H_dag


# ===============================================================
# MÉTODO 8 — VISUALIZACIÓN DEL RESULTADO
# ===============================================================
def visualizar_blur_y_restaurada_color(G_blur_color, F_rest_color, titulo="Blur vs Restaurada (color)"):
    """
    Método: visualizar_blur_y_restaurada_color

    Objetivo:
        Mostrar la imagen borrosa sintética y la imagen restaurada.

    """

    plt.figure(figsize=(12, 5))
    plt.suptitle(titulo, fontsize=16)

    plt.subplot(1, 2, 1)
    plt.title("Imagen Borrosa")
    plt.imshow(G_blur_color)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Restaurada (MP)")
    plt.imshow(F_rest_color)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ===============================================================
# MÉTODO 9 — DEMOSTRACIÓN FILA POR FILA
# ===============================================================
def demo_row_restoration(F_row, l=7):
    """
    Método: demo_row_restoration

    Objetivo:
        Mostrar el comportamiento del modelo en una sola fila:
            G_row = H^T F_row
            F_rec = G_row H⁺

    Permite visualizar el efecto del blur y su reconstrucción.
    """

    m = len(F_row)

    H = build_H(m, l)
    H_dag = newton_schulz_pseudoinversa(H)

    G_row = H.T @ F_row
    F_rec = G_row @ H_dag

    plt.figure(figsize=(10, 4))
    plt.plot(F_row, label="Original", color = "red")
    plt.plot(G_row[:m], label="Blur", color = "blue")
    plt.plot(F_rec, label="Reconstruida", color = "green")
    plt.legend()
    plt.grid()
    plt.show()


# ===============================================================
# PROGRAMA PRINCIPAL
# ===============================================================
if __name__ == "__main__":

    # Cargar imagen original
    F_color = load_image_color(str(PATH_IMG))

    # Longitud del blur
    l = 7

    # Aplicar blur sintético y restauración
    G_blur_color, F_rest_color = blur_and_restore_color(F_color, l)

    visualizar_blur_y_restaurada_color(
        G_blur_color, F_rest_color,
        titulo="Restauración mediante Moore–Penrose"
    )

    # Demostración en una fila
    fila = F_color.shape[0] // 2
    F_row = color.rgb2gray(F_color)[fila, :]

    demo_row_restoration(F_row, l)
