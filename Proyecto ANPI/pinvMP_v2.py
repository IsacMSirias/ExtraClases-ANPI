import numpy as np
from skimage import io, color
from skimage.util import img_as_float
from matplotlib import pyplot as plt
from pathlib import Path


# ===============================
# 1. ---------------- LOADER ----------------
# ===============================

def load_image(path: str) -> np.ndarray:
    """Carga imagen desde disco y normaliza en [0,1]."""
    img = io.imread(path)

    if img.ndim == 3:
        img = color.rgb2gray(img)

    return img_as_float(img).astype(np.float64)


# ===============================
# 2. ------------- BLUR MODEL --------------
# ===============================

def get_H_size(m: int, l: int, force_m: bool = True) -> int:
    """Devuelve n = m + l - 1 (ajustado a múltiplo de l si force_m=True)."""
    n = m + l - 1
    if force_m and (n % l != 0):
        n = ((n // l) + 1) * l
    return n


def build_H(m: int, l: int, force_m: bool = True) -> np.ndarray:
    """Construye matriz Toeplitz de blur horizontal."""
    n = get_H_size(m, l, force_m)
    H = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        H[i, i:i + l] = 1.0 / l

    return H


def build_H_vertical(r: int, l: int, force_m: bool = True) -> np.ndarray:
    """Construye matriz Toeplitz de blur vertical."""
    n = get_H_size(r, l, force_m)
    H = np.zeros((r, n), dtype=np.float64)

    for i in range(r):
        H[i, i:i + l] = 1.0 / l

    return H


# ===============================
# 3. ------ PSEUDOINVERSE GENERATOR -------
# ===============================

def newton_schulz_pseudoinversa(A: np.ndarray, tol: float = 1e-10, iterMax: int = 100):
    """
    Método iterativo de Newton–Schulz para aproximar la pseudoinversa A⁺.
    """
    m, n = A.shape
    Yk = A.T / (np.linalg.norm(A, "fro") ** 2)
    Im = np.eye(m)

    for k in range(iterMax):
        Yk = Yk @ (2 * Im - A @ Yk)
        er = np.linalg.norm(A @ Yk @ A - A, "fro")
        if er < tol:
            break

    return Yk


def build_H_pseudoinversa_NS(m: int, l: int, force_m: bool = True) -> np.ndarray:
    """Calcula H⁺ para blur horizontal."""
    H = build_H(m, l, force_m)
    return newton_schulz_pseudoinversa(H)


def build_Hc_pseudoinverse_NS(r: int, l: int, force_m: bool = True) -> np.ndarray:
    """Calcula H_c⁺ para blur vertical."""
    Hc = build_H_vertical(r, l, force_m)
    return newton_schulz_pseudoinversa(Hc)


# ===============================
# 4. --------------- RESTORER ---------------
# ===============================

def pad_image_horizontal(G: np.ndarray, n: int) -> np.ndarray:
    """Aumenta la imagen G en columnas hasta tamaño n."""
    m_original = G.shape[1]
    padding = n - m_original

    if padding <= 0:
        return G

    pad_block = np.tile(G[:, -1:], (1, padding))
    return np.hstack([G, pad_block])


def crop_image_horizontal(F_rec: np.ndarray, m_original: int) -> np.ndarray:
    """Recorta columnas añadidas."""
    return F_rec[:, :m_original]


def restore_image_horizontal(G: np.ndarray, l: int) -> np.ndarray:
    """
    Pipeline completo 1D del paper:
      F̃ = G (H⁺)ᵀ
    """

    rows, m = G.shape

    H_dag = build_H_pseudoinversa_NS(m, l, force_m=True)
    n = H_dag.shape[0]

    G_pad = pad_image_horizontal(G, n)
    F_rec_pad = G_pad @ H_dag.T
    F_rec = crop_image_horizontal(F_rec_pad, m)

    return F_rec


def restore_image_2d(G: np.ndarray, l_vertical: int, l_horizontal: int) -> np.ndarray:
    """
    Caso general del paper:
      G = H_c F H_rᵀ
      F̃ = H_c⁺ G (H_r⁺)ᵀ
    """

    rows, cols = G.shape

    # --- Horizontal first ---
    H_r_dag = build_H_pseudoinversa_NS(cols, l_horizontal)
    n_cols = H_r_dag.shape[0]

    G_pad = pad_image_horizontal(G, n_cols)
    tmp = G_pad @ H_r_dag.T
    tmp = tmp[:, :cols]

    # --- Vertical second ---
    H_c_dag = build_Hc_pseudoinverse_NS(rows, l_vertical)
    n_rows = H_c_dag.shape[0]

    pad_bottom = n_rows - rows
    if pad_bottom > 0:
        tmp = np.vstack([tmp, np.tile(tmp[-1:], (pad_bottom, 1))])

    F_rec_pad = H_c_dag @ tmp
    F_rec = F_rec_pad[:rows, :]

    return F_rec


# ===============================
# 5. --------------- EVALUATOR ---------------
# ===============================

def psnr(F: np.ndarray, F_rec: np.ndarray):
    mse = np.mean((F - F_rec) ** 2)
    if mse == 0:
        return np.inf
    return 10 * np.log10(1.0 / mse)


def isnr(F: np.ndarray, G: np.ndarray, F_rec: np.ndarray):
    num = np.sum((F - G) ** 2)
    den = np.sum((F - F_rec) ** 2)
    return 10 * np.log10(num / den)


# ===============================
# 6. ------------ VISUALIZER -------------
# ===============================

def show_images(G, F_rec):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Imagen borrosa G")
    plt.imshow(G, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Reconstrucción F̃")
    plt.imshow(F_rec, cmap='gray')
    plt.axis('off')

    plt.show()


# ===============================
# ------------------- MAIN ---------------------
# ===============================

if __name__ == "__main__":
    # Ruta base
    BASE_DIR = Path(__file__).resolve().parent
    IMG_DIR = BASE_DIR / "imagenes"
    path = IMG_DIR / "rdr.jpg"

    # 1. Cargar imagen
    G = load_image(path)
    print("Imagen cargada:", G.shape)

    # 2. Longitud de blur (elegir según prueba)
    l = 8

    # 3. Restaurar horizontal
    F_rec = restore_image_horizontal(G, l)

    # 4. Mostrar
    show_images(G, F_rec)

    print("Restauración horizontal lista.")
