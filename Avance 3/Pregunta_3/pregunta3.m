function pregunta3()
  % pregunta3 - Compara distintos métodos numéricos para resolver una EDO de primer orden.
  %
  % Descripción:
  %   Este script resuelve la EDO:
  %       y' = (x + y) / x,   con  y(2) = 4,  en el intervalo [2, 10].
  %   Se aplican múltiples métodos numéricos (Euler, RK, Taylor, Adams, etc.)
  %   y se comparan sus resultados con la solución analítica.
  %
  %   Finalmente, se grafica la comparación y se muestra una tabla con valores
  %   representativos de las soluciones.
  %
  % No recibe parámetros ni retorna valores; ejecuta todo el proceso al ser llamado.

  clc; clear; close all;

  % ---------------------------
  % DATOS DEL PROBLEMA
  % ---------------------------
  f = @(x, y) (x + y) / x;  % Definición de la EDO
  a = 2;                    % Límite inferior del intervalo
  b = 10;                   % Límite superior del intervalo
  y0 = 4;                   % Condición inicial
  n = 6;                    % Número de subintervalos
  h = (b - a) / n;          % Tamaño de paso
  x = a:h:b;                % Vector de nodos

  % Solución analítica exacta para comparación
  y_exacta = @(x) (log(x) - log(2) + 2) .* x;

  % ---------------------------
  % CÁLCULO CON MÉTODOS NUMÉRICOS
  % ---------------------------
  [x, y_euler]   = metodo_euler(f, a, b, y0, n);
  [x, y_pc]      = predictor_corrector(f, a, b, y0, n);
  [x, y_rk2]     = metodo_rk2(f, a, b, y0, n);
  [x, y_rk3]     = metodo_rk3(f, a, b, y0, n);
  [x, y_rk4]     = metodo_rk4(f, a, b, y0, n);

  fsym = @(x, y) (x + y) / x;
  [x, y_taylor2] = metodo_taylor(fsym, a, b, y0, n, 2);

  [x, y_adams2]  = metodo_adams2(f, a, b, y0, n);
  [x, y_adams3]  = metodo_adams3(f, a, b, y0, n);
  [x, y_adams4]  = metodo_adams4(f, a, b, y0, n);

  % ---------------------------
  % GRÁFICA DE RESULTADOS
  % ---------------------------
  figure;
  hold on; grid on;

  plot(x, y_exacta(x), 'k', 'LineWidth', 2, 'DisplayName', 'Solución exacta');
  plot(x, y_euler, 'r-o', 'DisplayName', 'Euler');
  plot(x, y_pc, 'm-s', 'DisplayName', 'Predictor-Corrector');
  plot(x, y_rk2, 'g-^', 'DisplayName', 'RK2');
  plot(x, y_rk3, 'b-d', 'DisplayName', 'RK3');
  plot(x, y_rk4, 'c-*', 'DisplayName', 'RK4');
  plot(x, y_taylor2, 'y-p', 'DisplayName', 'Taylor 2');
  plot(x, y_adams2, '--r', 'DisplayName', 'Adams-Bashforth 2');
  plot(x, y_adams3, '--g', 'DisplayName', 'Adams-Bashforth 3');
  plot(x, y_adams4, '--b', 'DisplayName', 'Adams-Bashforth 4');

  title('Comparación de métodos numéricos');
  xlabel('x'); ylabel('y(x)');
  legend('Location', 'northwest');
  hold off;

  % ---------------------------
  % TABLA DE RESULTADOS
  % ---------------------------
  fprintf('\n   x\t\tExacta\t\tEuler\t\tRK4\t\tAdams4\n');
  fprintf('-------------------------------------------------------------\n');
  for k = 1:length(x)
    fprintf('%.2f\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
      x(k), y_exacta(x(k)), y_euler(k), y_rk4(k), y_adams4(k));
  end

  fprintf('\nComparación completada. Gráfica generada correctamente.\n');
end


% =========================================================================
% MÉTODOS NUMÉRICOS
% =========================================================================

function [x, y] = metodo_euler(f, a, b, y0, n)
  % metodo_euler - Método de Euler explícito para EDOs de primer orden.
  h = (b - a) / n;
  x = a:h:b;
  y = zeros(1, n + 1);
  y(1) = y0;

  for k = 1:n
    y(k + 1) = y(k) + h * f(x(k), y(k));
  end
end


function [x, y] = predictor_corrector(f, a, b, y0, n)
  % predictor_corrector - Método de Heun (Euler mejorado).
  h = (b - a) / n;
  x = a:h:b;
  y = zeros(1, n + 1);
  y(1) = y0;

  for k = 1:n
    y_pred = y(k) + h * f(x(k), y(k)); % Predicción (Euler)
    y(k + 1) = y(k) + (h / 2) * (f(x(k), y(k)) + f(x(k + 1), y_pred)); % Corrección
  end
end


function [x, y] = metodo_rk2(f, a, b, y0, n)
  % metodo_rk2 - Método de Runge-Kutta de orden 2 (RK2).
  h = (b - a) / n;
  x = a:h:b;
  y = zeros(1, n + 1);
  y(1) = y0;

  for k = 1:n
    K1 = f(x(k), y(k));
    K2 = f(x(k) + h / 2, y(k) + (h / 2) * K1);
    y(k + 1) = y(k) + h * K2;
  end
end


function [x, y] = metodo_rk3(f, a, b, y0, n)
  % metodo_rk3 - Método de Runge-Kutta de orden 3 (RK3).
  h = (b - a) / n;
  x = a:h:b;
  y = zeros(1, n + 1);
  y(1) = y0;

  for k = 1:n
    K1 = f(x(k), y(k));
    K2 = f(x(k) + h / 2, y(k) + (h / 2) * K1);
    K3 = f(x(k) + h, y(k) - h * K1 + 2 * h * K2);
    y(k + 1) = y(k) + (h / 6) * (K1 + 4 * K2 + K3);
  end
end


function [x, y] = metodo_rk4(f, a, b, y0, n)
  % metodo_rk4 - Método de Runge-Kutta clásico de orden 4 (RK4).
  h = (b - a) / n;
  x = a:h:b;
  y = zeros(1, n + 1);
  y(1) = y0;

  for k = 1:n
    K1 = f(x(k), y(k));
    K2 = f(x(k) + h / 2, y(k) + (h / 2) * K1);
    K3 = f(x(k) + h / 2, y(k) + (h / 2) * K2);
    K4 = f(x(k) + h, y(k) + h * K3);
    y(k + 1) = y(k) + (h / 6) * (K1 + 2 * K2 + 2 * K3 + K4);
  end
end


function [x, y] = metodo_taylor(fsym, a, b, y0, n, p)
  % metodo_taylor - Método de Taylor de orden p para EDOs.
  % Utiliza derivadas simbólicas automáticas con el paquete symbolic.
  pkg load symbolic

  syms x y
  f_sym = fsym(x, y);

  % Derivadas sucesivas de f(x, y)
  derivs = cell(p, 1);
  derivs{1} = f_sym;
  for k = 2:p
    derivs{k} = diff(derivs{k - 1}, x) + diff(derivs{k - 1}, y) * f_sym;
  end

  % Conversión a funciones numéricas
  f_funcs = cell(p, 1);
  for k = 1:p
    f_funcs{k} = matlabFunction(derivs{k}, "vars", [x y]);
  end

  % Integración paso a paso
  h = (b - a) / n;
  x = a:h:b;
  y = zeros(1, n + 1);
  y(1) = y0;

  for k = 1:n
    suma = 0;
    for i = 1:p
      suma += (h^i / factorial(i)) * f_funcs{i}(x(k), y(k));
    end
    y(k + 1) = y(k) + suma;
  end
end


function [x, y] = metodo_adams2(f, a, b, y0, n)
  % metodo_adams2 - Método de Adams-Bashforth de orden 2.
  h = (b - a) / n;
  x = a:h:b;
  y = zeros(1, n + 1);
  y(1) = y0;

  % Primer paso con RK4
  for k = 1:1
    K1 = f(x(k), y(k));
    K2 = f(x(k) + h / 2, y(k) + (h / 2) * K1);
    K3 = f(x(k) + h / 2, y(k) + (h / 2) * K2);
    K4 = f(x(k) + h, y(k) + h * K3);
    y(k + 1) = y(k) + (h / 6) * (K1 + 2 * K2 + 2 * K3 + K4);
  end

  % Método Adams-Bashforth 2 pasos
  for k = 2:n
    y(k + 1) = y(k) + (h / 2) * (3 * f(x(k), y(k)) - f(x(k - 1), y(k - 1)));
  end
end


function [x, y] = metodo_adams3(f, a, b, y0, n)
  % metodo_adams3 - Método de Adams-Bashforth de orden 3.
  h = (b - a) / n;
  x = a:h:b;
  y = zeros(1, n + 1);
  y(1) = y0;

  % Primeros dos pasos con RK4
  for k = 1:2
    K1 = f(x(k), y(k));
    K2 = f(x(k) + h / 2, y(k) + (h / 2) * K1);
    K3 = f(x(k) + h / 2, y(k) + (h / 2) * K2);
    K4 = f(x(k) + h, y(k) + h * K3);
    y(k + 1) = y(k) + (h / 6) * (K1 + 2 * K2 + 2 * K3 + K4);
  end

  % Método Adams-Bashforth 3 pasos
  for k = 3:n
    y(k + 1) = y(k) + (h / 12) * (23 * f(x(k), y(k)) - 16 * f(x(k - 1), y(k - 1)) + ...
                                  5 * f(x(k - 2), y(k - 2)));
  end
end


function [x, y] = metodo_adams4(f, a, b, y0, n)
  % metodo_adams4 - Método de Adams-Bashforth de orden 4.
  h = (b - a) / n;
  x = a:h:b;
  y = zeros(1, n + 1);
  y(1) = y0;

  % Primeros tres pasos con RK4
  for k = 1:3
    K1 = f(x(k), y(k));
    K2 = f(x(k) + h / 2, y(k) + (h / 2) * K1);
    K3 = f(x(k) + h / 2, y(k) + (h / 2) * K2);
    K4 = f(x(k) + h, y(k) + h * K3);
    y(k + 1) = y(k) + (h / 6) * (K1 + 2 * K2 + 2 * K3 + K4);
  end

  % Método Adams-Bashforth 4 pasos
  for k = 4:n
    y(k + 1) = y(k) + (h / 24) * (55 * f(x(k), y(k)) - 59 * f(x(k - 1), y(k - 1)) + ...
                                  37 * f(x(k - 2), y(k - 2)) - 9 * f(x(k - 3), y(k - 3)));
  end
end

