function pregunta_2()
  % FUNCIÓN PRINCIPAL - Cálculo de cota de error de interpolación
  clc; clear; close all

  % Definir la función f(x) como función anónima
  f = @(x) log(asin(x)) / log(x);
  a = 0.1;    % Límite inferior del intervalo
  b = 0.8;    % Límite superior del intervalo
  xv = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];  % Nodos de interpolación
  x_0 = 0.55; % Punto donde se evalúa el error

  % Calcular cota de error de interpolación
  ct = cota_interpolacion(f, a, b, xv, x_0)
  fprintf('Cota de error: %g\n', ct);

end

function ct = cota_interpolacion(f, a, b, xv, x_0)
  % COTA_INTERPOLACION - Calcula cota de error para interpolación polinómica
  %
  % Parámetros:
  %   f   : function_handle
  %         Función a interpolar
  %   a, b: double
  %         Extremos del intervalo
  %   xv  : vector
  %         Nodos de interpolación
  %   x_0 : double
  %         Punto donde se evalúa el error
  %
  % Retorna:
  %   ct  : double
  %         Cota superior del error de interpolación

  % Cargar paquete de cálculo simbólico
  pkg load symbolic
  syms x

  % Determinar número de puntos (grado del polinomio + 1)
  nPlusOne = length(xv);

  % Convertir a función simbólica y calcular derivada de orden n+1
  fs = sym(f);
  fds = diff(fs, nPlusOne);  % Derivada de orden (n+1)

  % Convertir a función numérica para evaluación
  fdn = matlabFunction(fds);

  % Encontrar máximo absoluto de |f^(n+1)(x)| en [a, b]
  % Usamos minimización del negativo para encontrar máximo
  newfdn = @(x) -1 * abs(fdn(x));

  % Función wrapper para fminbnd
  fmaxbnd = @(newfdn, a, b) fminbnd(newfdn, a, b);

  % Encontrar punto donde |f^(n+1)(x)| es máximo
  xmax = fmaxbnd(newfdn, a, b);

  % Valor máximo de la derivada (n+1)-ésima
  alphamax = abs(fdn(xmax));

  % Calcular productorio |(x_0 - x_0)(x_0 - x_1)...(x_0 - x_n)|
  n = nPlusOne - 1;  % Grado del polinomio interpolante
  mult = 1;
  for j = 0:n
    mult = mult * (x_0 - xv(j + 1));  % Productorio de diferencias
  endfor

  % Fórmula de cota de error para interpolación polinómica:
  % |f(x) - P(x)| ≤ [max|f^(n+1)(ξ)| / (n+1)!] * |Π(x - x_i)|
  ct = (alphamax / factorial(nPlusOne)) * abs(mult);

endfunction

