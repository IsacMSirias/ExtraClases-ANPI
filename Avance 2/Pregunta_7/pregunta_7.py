import numpy as np


def fact_qr(A):
    """
    Factorización QR usando el método de Gram-Schmidt clásico.

    Descompone una matriz A en el producto Q*R, donde:
    - Q es una matriz ortogonal (columnas ortonormales)
    - R es una matriz triangular superior

    Parámetros:
        A : array-like
            Matriz a factorizar de dimensiones n×m

    Retorna:
        tuple: (Q, R) donde:
            Q: Matriz ortogonal n×m
            R: Matriz triangular superior m×m

    Lanza:
        ValueError: Si las columnas son linealmente dependientes
    """
    A = np.array(A, dtype=float)
    n, m = A.shape

    # Inicializar matrices Q y R
    Q = np.zeros((n, m))  # Matriz ortogonal
    R = np.zeros((m, m))  # Matriz triangular superior

    # Procesar primera columna
    u1 = A[:, 0].copy()  # Primer vector de A
    R[0, 0] = np.linalg.norm(u1)  # Norma del primer vector
    if R[0, 0] == 0:
        raise ValueError("La primera columna no puede ser cero")
    Q[:, 0] = u1 / R[0, 0]  # Normalizar para obtener primer vector ortonormal

    # Proceso de Gram-Schmidt para las demás columnas
    for k in range(1, m):
        a_k = A[:, k].copy()  # k-ésima columna de A
        u_k = a_k.copy()  # Vector residual inicial

        # Restar las proyecciones sobre los vectores ortonormales anteriores
        for j in range(k):
            R[j, k] = np.dot(a_k, Q[:, j])  # Proyección: <a_k, q_j>
            u_k = u_k - R[j, k] * Q[:, j]  # Remover componente en dirección q_j

        # Normalizar el vector residual para obtener nuevo vector ortonormal
        R[k, k] = np.linalg.norm(u_k)
        if R[k, k] == 0:
            raise ValueError("Columnas linealmente dependientes")
        Q[:, k] = u_k / R[k, k]  # k-ésimo vector ortonormal

    return Q, R


def main():
    """
    Función principal para demostrar la factorización QR.

    Retorna:
        tuple: Matrices Q y R de la factorización
    """
    A = np.array([[1, 1],
                  [1, -1]], dtype=float)

    return fact_qr(A)


# Ejecutar demostración
Q, R = main()

print("\nMatriz Q (ortogonal):")
print(Q)
print("\nMatriz R (triangular superior):")
print(R)
print("\nVerificación A = QR:")
print(Q @ R)  # Debería ser igual a A
print("\nVerificación Q^TQ = I:")
print(Q.T @ Q)  # Debería ser la matriz identidad