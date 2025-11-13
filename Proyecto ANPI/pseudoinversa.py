import numpy as np


def newton_schulz_pseudoinversa(A, tol=1e-10, iterMax=100):
    """
    Método Newton-Schultz para aproximar la pseudoinversa de una matriz

    Parámetros:
    A: matriz a calcular pseudoinversa
    tol: tolerancia
    iterMax: máximo de iteraciones

    Retorna:
    Yk: aproximación de la pseudoinversa A⁺
    k: número de iteraciones realizadas
    er: error final
    """
    # Vector inicial
    Y0 = (1 / np.linalg.norm(A, 'fro') ** 2) * A.T

    Yk = Y0.copy()
    m = A.shape[0]  # Número de filas de A
    Im = np.eye(m)

    for k in range(iterMax):
        # Iteración de Newton-Schultz
        Yk = Yk @ (2 * Im - A @ Yk)

        # Error: norma de A*Yk*A - A
        er = np.linalg.norm(A @ Yk @ A - A, 'fro')

        if er < tol:
            print(f"Convergió en {k + 1} iteraciones")
            break

    return Yk, k + 1, er


# EJEMPLO DE USO - Código 1
if __name__ == "__main__":
    print("=== CÓDIGO 1: Newton-Schultz para Pseudoinversa Completa ===")

    # Matriz del ejemplo
    A = np.array([
        [2, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=float)

    print("Matriz A:")
    print(A)

    # Calcular pseudoinversa con Newton-Schultz
    Yk, iteraciones, error = newton_schulz_pseudoinversa(A)

    print(f"\nPseudoinversa aproximada Yk:")
    print(Yk)
    print(f"\nIteraciones: {iteraciones}")
    print(f"Error final: {error:.2e}")

    # Verificar propiedades de la pseudoinversa
    print("\nVerificación de propiedades:")
    print(f"Norma de A*Yk*A - A: {np.linalg.norm(A @ Yk @ A - A, 'fro'):.2e}")
    print(f"Norma de Yk*A*Yk - Yk: {np.linalg.norm(Yk @ A @ Yk - Yk, 'fro'):.2e}")

    # Comparar con pseudoinversa de numpy
    A_pinv_numpy = np.linalg.pinv(A)
    print(f"\nDiferencia con np.linalg.pinv: {np.linalg.norm(Yk - A_pinv_numpy, 'fro'):.2e}")