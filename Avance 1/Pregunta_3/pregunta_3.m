function pregunta_3()
  clc; clear;close all

  % Definir la función
  f='exp(x)-2*x-10';

  % Parámetros
  a=-6;
  b=-4;
  tol=1e-8;
  iterMax=1000;


  [xk,k,erk]=biseccion(f,a,b,tol,iterMax);

  % Mostrar resultados
  printf("Raíz aproximada: %.10f\n", xk);
  printf("Número de iteraciones: %d\n", k);
  printf("Error en la función: %.2e\n", erk);

end

function [xk,k,erk]=biseccion(f,a,b,tol,iterMax)
  fn=str2func(['@(x)' f]);  %Funcion Numerica (Usar)

  if fn(a) * fn(b) > 0
    error ('La función no cambia de signo en [a,b]');
  endif

  % Vectores para informacion de graficas
  err_vec = [];
  xk_vec = [];

  for k=1:iterMax
    xk= (a + b) / 2;

    % Guardar informacion para graficas
    err_vec(end + 1) = abs(fn(xk));
    xk_vec(end + 1) = xk;


    if fn(a)*fn(xk)<0
      b = xk;
    else
      a = xk;
    endif

    erk=abs(fn(xk));

    if erk<tol
      break;
    endif
  endfor

  % Gráficas
  k;
  err_vec;
  xk_vec;

  figure;
  plot(1:k, abs(err_vec), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
  title('Iteraciones vs el error en la función');
  xlabel('Iteraciones (k)');
  ylabel('Error (err_vec)');
  grid on;

  figure;
  plot(1:k, xk_vec, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
  title('Número de iteraciones vs la aproximación');
  xlabel('Iteraciones (k)');
  ylabel('Aproximacion (xk_vec)');
  grid on;

end
