import numpy as np


def newton_schulz_solver(A, b, tol=1e-10, iterMax=100):
    """
    Resuelve Ax = b usando Newton-Schultz sin calcular la pseudoinversa completa

    Parámetros:
    A: matriz del sistema
    b: vector término independiente
    tol: tolerancia
    iterMax: máximo de iteraciones

    Retorna:
    xk: solución aproximada
    k: número de iteraciones realizadas
    er: error final entre iteraciones
    error_sistema: norma de A*xk - b
    """
    # Vector inicial
    Y0 = (1 / np.linalg.norm(A, 'fro') ** 2) * A.T

    Yk = Y0.copy()
    m = A.shape[0]  # Número de filas de A
    Im = np.eye(m)

    # Aproximación de la solución con valor inicial
    xk = Yk @ b

    for k in range(iterMax):
        # Iteración de Newton-Schultz
        Yk = Yk @ (2 * Im - A @ Yk)

        # Nueva aproximación de la solución
        xk_new = Yk @ b

        # Error entre iteraciones consecutivas
        er = np.linalg.norm(xk_new - xk)

        if er < tol:
            xk = xk_new
            print(f"Convergió en {k + 1} iteraciones")
            break

        xk = xk_new

    # Error del sistema
    error_sistema = np.linalg.norm(A @ xk - b)

    return xk, k + 1, er, error_sistema


# EJEMPLOS DE USO - Código 2
if __name__ == "__main__":
    print("\n=== CÓDIGO 2: Newton-Schultz para Resolver Sistema ===")

    # Caso 1: Solución única
    print("\n--- Caso 1: Solución única ---")
    A1 = np.array([[1, 2], [3, 5]], dtype=float)
    b1 = np.array([-3, -8], dtype=float)

    x1, iter1, er1, error_sist1 = newton_schulz_solver(A1, b1)
    print(f"Solución: {x1}")
    print(f"Iteraciones: {iter1}")
    print(f"Error entre iteraciones: {er1:.2e}")
    print(f"Error del sistema: {error_sist1:.2e}")

    # Comparar con solución de numpy
    x1_numpy = np.linalg.solve(A1, b1)
    print(f"Solución NumPy: {x1_numpy}")
    print(f"Diferencia: {np.linalg.norm(x1 - x1_numpy):.2e}")

    # Caso 2: Infinitas soluciones (sistema subdeterminado)
    print("\n--- Caso 2: Infinitas soluciones ---")
    A2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    b2 = np.array([6, 15], dtype=float)

    x2, iter2, er2, error_sist2 = newton_schulz_solver(A2, b2)
    print(f"Solución: {x2}")
    print(f"Iteraciones: {iter2}")
    print(f"Error entre iteraciones: {er2:.2e}")
    print(f"Error del sistema: {error_sist2:.2e}")

    # Verificar solución mínima norma
    print(f"Norma de la solución: {np.linalg.norm(x2):.6f}")

    # Caso 3: No tiene solución (sistema sobredeterminado)
    print("\n--- Caso 3: No tiene solución exacta ---")
    A3 = np.array([[1, 2], [-2, -4], [5, 3]], dtype=float)
    b3 = np.array([1, 2, 3], dtype=float)

    x3, iter3, er3, error_sist3 = newton_schulz_solver(A3, b3)
    print(f"Solución de mínimos cuadrados: {x3}")
    print(f"Iteraciones: {iter3}")
    print(f"Error entre iteraciones: {er3:.2e}")
    print(f"Error del sistema: {error_sist3:.2e}")

    # Comparar con solución de mínimos cuadrados de numpy
    x3_numpy, _, _, _ = np.linalg.lstsq(A3, b3, rcond=None)
    print(f"Solución NumPy (lstsq): {x3_numpy}")
    print(f"Diferencia: {np.linalg.norm(x3 - x3_numpy):.2e}")