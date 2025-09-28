function [L, U] = fact_LU(A)
    # fact_LU - Calculo de las matrices L U de A = L U
    [m, n] = size(A);
    if m ~= n
        error('La matriz debe ser cuadrada');
    end

    # Verificación por rangos (reemplaza subm_check y diag_check)
    if ~tiene_LU_unica(A)
        error('La matriz no tiene factorización LU única');
    endif

    U = A;
    L = eye(n);  # Identidad

    for k = 1:n-1
        if abs(U(k,k)) < 1e-12
            error('Pivote numéricamente cero en (%d,%d)', k, k);
        end

        for i = k+1:n
            factor = U(i,k) / U(k,k);
            L(i,k) = factor;  # Guardar multiplicador
            U(i,k:n) = U(i,k:n) - factor * U(k,k:n);
        end
    end
endfunction

function resultado = tiene_LU_unica(A)
    # tiene_LU_unica - Verifica si A tiene factorización LU única usando rangos
    # Teorema: A tiene LU única si todas sus submatrices principales
    # tienen rango completo

    n = size(A, 1);
    resultado = 1;
    for k = 1:n
        # Submatriz principal de tamaño k×k
        Ak = A(1:k, 1:k);

        # Verificar si el rango es igual al tamaño de la submatriz
        if rank(Ak) < k
            resultado = 0;
            return;
        endif
    endfor

endfunction
