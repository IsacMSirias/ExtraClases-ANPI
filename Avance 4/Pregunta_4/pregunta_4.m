function pregunta_4()
  % FUNCIÓN PRINCIPAL - Comparación de métodos de cuadratura gaussiana
  clc; clear; close all

  % Definir función a integrar
  f = @(x) cos(x) * exp(x);
  a = 2;  % Límite inferior de integración
  b = 5;  % Límite superior de integración
  k = 7;  % Número de nodos para cuadratura gaussiana
  n = 20; % Número de subintervalos para método compuesto

  % Función transformada para intervalo [-1, 1]
  g = @(x) ((b-a)/2) * f(((b-a) * x + (b+a))/2);

  % Calcular integral con diferentes métodos
  I_S = cuad_gauss(g, a, b, k);      % Gaussiana simple
  I_C = cuad_gauss_comp(f, a, b, k, n); % Gaussiana compuesta
  I_T = cuad_gauss_iter(f, a, b, k);    % Gaussiana iterativa

  % Mostrar resultados
  fprintf('Gauss Simple: %g\n', I_S);
  fprintf('Gauss Compuesta: %g\n', I_C);
  fprintf('Gauss Iterativa: %g\n', I_T);

end

function I = cuad_gauss(f, a, b, k)
  % CUAD_GAUSS - Cuadratura gaussiana simple en intervalo [a,b]
  %
  % Parámetros:
  %   f : function_handle
  %       Función a integrar
  %   a, b : double
  %       Límites de integración
  %   k : int
  %       Número de nodos de cuadratura
  %
  % Retorna:
  %   I : double
  %       Aproximación de la integral

  % Obtener nodos y pesos para cuadratura gaussiana de orden k
  [x, w] = nodos_pesos(k);

  % Calcular integral como suma ponderada
  I = 0;
  for i = 1:k
    I = I + w(i) * f(x(i));
  endfor
endfunction

function I = cuad_gauss_comp(f, a, b, k, n)
  % CUAD_GAUSS_COMP - Cuadratura gaussiana compuesta
  %
  % Divide el intervalo [a,b] en n subintervalos y aplica
  % cuadratura gaussiana en cada uno
  %
  % Parámetros:
  %   f : function_handle
  %       Función a integrar
  %   a, b : double
  %       Límites de integración
  %   k : int
  %       Número de nodos por subintervalo
  %   n : int
  %       Número de subintervalos
  %
  % Retorna:
  %   I : double
  %       Aproximación de la integral

  % Crear partición del intervalo
  h = (b - a) / n;
  x = a:h:b;

  % Obtener nodos y pesos estándar en [-1,1]
  [xi, wi] = nodos_pesos(k);

  I = 0;

  % Aplicar cuadratura en cada subintervalo
  for j = 1:n
    aj = x(j);    % Límite inferior del subintervalo
    bj = x(j + 1); % Límite superior del subintervalo

    % Función transformada para subintervalo [aj, bj] -> [-1, 1]
    g = @(t) ((bj - aj) / 2) * f(((bj - aj) * t + (bj + aj)) / 2);

    % Aplicar cuadratura gaussiana en subintervalo actual
    S = 0;
    for i = 1:k
      S = S + wi(i) * g(xi(i));
    endfor

    I = I + S;  % Acumular resultado
  endfor
endfunction

function I = cuad_gauss_iter(f, a, b, k)
  % CUAD_GAUSS_ITER - Cuadratura gaussiana con refinamiento iterativo
  %
  % Aumenta el número de subintervalos hasta alcanzar tolerancia
  %
  % Parámetros:
  %   f : function_handle
  %       Función a integrar
  %   a, b : double
  %       Límites de integración
  %   k : int
  %       Número de nodos por subintervalo
  %
  % Retorna:
  %   I : double
  %       Aproximación de la integral dentro de la tolerancia

  tol = 1e-8;      % Tolerancia de convergencia
  iterMax = 1e6;   % Número máximo de iteraciones
  I = 0;

  % Primera aproximación con n = 1 subintervalo
  S_prev = cuad_gauss_comp(f, a, b, k, 1);

  % Refinamiento iterativo
  for n = 2:iterMax
    S_curr = cuad_gauss_comp(f, a, b, k, n);  % Aproximación actual
    err = abs(S_curr - S_prev);               % Error absoluto

    % Verificar convergencia
    if err < tol
      I = S_curr;
      return
    endif

    S_prev = S_curr;  % Actualizar aproximación anterior
  endfor

  % Si no converge, mostrar advertencia
  warning("No se alcanzó la tolerancia antes del máximo de iteraciones");
  I = S_prev;  % Retornar mejor aproximación obtenida

endfunction

function [x, w] = nodos_pesos(k)
    % NODOS_PESOS - Devuelve nodos y pesos para cuadratura gaussiana
    %
    % Parámetros:
    %   k - orden de la cuadratura (1-10)
    %
    % Retorna:
    %   x - vector de nodos x_i para el orden k en [-1, 1]
    %   w - vector de pesos para el orden k

    % Validar entrada
    if k < 1 || k > 10
        error('El orden k debe estar entre 1 y 10');
    end

    % Tabla de nodos y pesos para diferentes órdenes
    % Los valores son precalculados para cuadraturas gaussianas
    switch k
        case 1
            % Cuadratura de punto medio
            x = 0;
            w = 2;

        case 2
            % Gauss-Legendre 2 puntos
            x = [-0.5773502691896257; 0.5773502691896257];
            w = [1; 1];

        case 3
            % Gauss-Legendre 3 puntos
            x = [-0.7745966692414834; 0; 0.7745966692414834];
            w = [0.5555555555555556; 0.8888888888888888; 0.5555555555555556];

        % Casos 4-10: Cuadraturas de mayor orden
        % (Los coeficientes son valores estándar de Gauss-Legendre)
        case 4
            x = [-0.8611363115940526; -0.3399810435848563; 0.3399810435848563; 0.8611363115940526];
            w = [0.3478548451374538; 0.6521451548625461; 0.6521451548625461; 0.3478548451374538];

        case 5
            x = [-0.9061798459386640; -0.5384693101056831; 0; 0.5384693101056831; 0.9061798459386640];
            w = [0.2369268850561891; 0.4786286704993665; 0.5688888888888889; 0.4786286704993665; 0.2369268850561891];

        case 6
            x = [-0.9324695142031521; -0.6612093864662645; -0.2386191860831969; 0.2386191860831969; 0.6612093864662645; 0.9324695142031521];
            w = [0.1713244923791704; 0.3607615730481386; 0.4679139345726910; 0.4679139345726910; 0.3607615730481386; 0.1713244923791704];

        case 7
            x = [-0.9491079123427585; -0.7415311855993945; -0.4058451513773972; 0; 0.4058451513773972; 0.7415311855993945; 0.9491079123427585];
            w = [0.1294849661688697; 0.2797053914892766; 0.3818300505051189; 0.4179591836734694; 0.3818300505051189; 0.2797053914892766; 0.1294849661688697];

        case 8
            x = [-0.9602898564975363; -0.7966664774136267; -0.5255324099163290; -0.1834346424956498; 0.1834346424956498; 0.5255324099163290; 0.7966664774136267; 0.9602898564975363];
            w = [0.1012285362903763; 0.2223810344533745; 0.3137066458778873; 0.3626837833783620; 0.3626837833783620; 0.3137066458778873; 0.2223810344533745; 0.1012285362903763];

        case 9
            x = [-0.9681602395076261; -0.8360311073266358; -0.6133714327005904; -0.3242534234038089; 0; 0.3242534234038089; 0.6133714327005904; 0.8360311073266358; 0.9681602395076261];
            w = [0.0812743883615744; 0.1806481606948574; 0.2606106964029354; 0.3123470770400029; 0.3302393550012598; 0.3123470770400029; 0.2606106964029354; 0.1806481606948574; 0.0812743883615744];

        case 10
            x = [-0.9739065285171717; -0.8650633666889845; -0.6794095682990244; -0.4333953941292472; -0.1488743389816312; 0.1488743389816312; 0.4333953941292472; 0.6794095682990244; 0.8650633666889845; 0.9739065285171717];
            w = [0.0666713443086881; 0.1494513491505806; 0.2190863625159820; 0.2692667193099963; 0.2955242247147529; 0.2955242247147529; 0.2692667193099963; 0.2190863625159820; 0.1494513491505806; 0.0666713443086881];
    end
end
