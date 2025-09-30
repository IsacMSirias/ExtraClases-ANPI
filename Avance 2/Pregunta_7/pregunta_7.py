import numpy as np


def fact_qr(A):
    """
    Factorización QR usando el método de Gram-Schmidt

    Parameters:
    A : array-like - Matriz a factorizar

    Returns:
    Q : array - Matriz ortogonal
    R : array - Matriz triangular superior
    """
    A = np.array(A, dtype=float)
    n, m = A.shape

    # Inicializar matrices Q y R
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    # Primer vector
    u1 = A[:, 0].copy()
    R[0, 0] = np.linalg.norm(u1)
    if R[0, 0] == 0:
        raise ValueError("La primera columna no puede ser cero")
    Q[:, 0] = u1 / R[0, 0]

    # Proceso de Gram-Schmidt para las demás columnas
    for k in range(1, m):
        a_k = A[:, k].copy()
        u_k = a_k.copy()

        # Restar las proyecciones sobre los vectores anteriores
        for j in range(k):
            R[j, k] = np.dot(a_k, Q[:, j])  # <a_k, e_j>
            u_k = u_k - R[j, k] * Q[:, j]  # u_k = a_k - Σ⟨a_k,e_j⟩e_j

        # Normalizar el vector residual
        R[k, k] = np.linalg.norm(u_k)
        if R[k, k] == 0:
            raise ValueError("Columnas linealmente dependientes")
        Q[:, k] = u_k / R[k, k]

    return Q, R


def main():
    A = np.array([[1, 1],
                  [1, -1]], dtype=float)

    return fact_qr(A)

Q, R = main()

print("\nMatriz Q (ortogonal):")
print(Q)
print("\nMatriz R (triangular superior):")
print(R)
print("\nVerificación A = QR:")
print(Q @ R)  # Debería ser igual a A
print("\nVerificación Q^TQ = I:")
print(Q.T @ Q)  # Debería ser la matriz identidad
