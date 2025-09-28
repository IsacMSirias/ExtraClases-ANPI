function d = det_fact_lu(A)
    # detfactlu - Calcula el determinante usando factorización LU
    # d = det_fact_lu(A) donde d = producto de los elementos diagonales de U

    [n, n] = size(A);

    [L, U] = fact_LU(A); # Obtener L y U

    d = 1;
    # Multiplicar los elementos de la diagonal de U
    for k = 1:n
        d = d * U(k, k);
    end
end

