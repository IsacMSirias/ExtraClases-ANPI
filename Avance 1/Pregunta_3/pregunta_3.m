function pregunta_3()
  clc; clear;close all

  % Definir la función
  f='exp(x)-2*x-10';

  % Parámetros
  a=-6;
  b=-4;
  tol=1e-8;
  iterMax=1000;

  % Llamar al método de bisección
  [xk,k,erk]=biseccion(f,a,b,tol,iterMax);

  % Mostrar resultados
  printf("Raíz aproximada: %.10f\n", xk);
  printf("Número de iteraciones: %d\n", k);
  printf("Error en la función: %.2e\n", erk);

end

function [xk,k,erk]=biseccion(f,a,b,tol,iterMax)
  % Método de bisección para encontrar una raíz de una función en un intervalo
  %
  % Parámetros:
  %   f : string
  %       Función en formato string para evaluar
  %   a : float
  %       Extremo izquierdo del intervalo inicial
  %   b : float
  %       Extremo derecho del intervalo inicial
  %   tol : float
  %       Tolerancia para el criterio de parada
  %   iterMax : int
  %       Número máximo de iteraciones permitidas
  %
  % Retorna:
  %   xk : float
  %       Aproximación de la raíz encontrada
  %   k : int
  %       Número de iteraciones realizadas
  %   erk : float
  %       Error absoluto de la función en la aproximación final

  fn=str2func(['@(x)' f]);  %Funcion Numerica (Usar)

  if fn(a) * fn(b) > 0
    error ('La función no cambia de signo en [a,b]');
  endif

  % Vectores para informacion de graficas
  err_vec = [];
  xk_vec = [];

   % Iteraciones del método de bisección
  for k=1:iterMax
    xk= (a + b) / 2; % Punto medio del intervalo

    % Guardar informacion para graficas
    err_vec(end + 1) = abs(fn(xk));
    xk_vec(end + 1) = xk;

    % Determinar en qué subintervalo está la raíz
    if fn(a)*fn(xk)<0
      b = xk;
    else
      a = xk;
    endif

    % Calcular error absoluto
    erk=abs(fn(xk));

    % Verificar criterio de parada
    if erk<tol
      break;
    endif
  endfor

  % Gráficas
  k;
  err_vec;
  xk_vec;

  % Generar gráfica de error vs iteraciones
  figure;
  plot(1:k, abs(err_vec), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
  title('Iteraciones vs el error en la función');
  xlabel('Iteraciones (k)');
  ylabel('Error (err_vec)');
  grid on;

  % Generar gráfica de aproximación vs iteraciones
  figure;
  plot(1:k, xk_vec, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
  title('Número de iteraciones vs la aproximación');
  xlabel('Iteraciones (k)');
  ylabel('Aproximacion (xk_vec)');
  grid on;

end
