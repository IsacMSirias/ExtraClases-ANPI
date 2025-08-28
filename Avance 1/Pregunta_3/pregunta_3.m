function pregunta_3()
  clc; clear;

  % Definir la función
  f='exp(x)-2*x-10'

  % Parámetros
  tol=1e-8
  iterMax=1000
  a=-6
  b=-4

  [xk,k,erk]=biseccion(f,a,b,tol,iterMax)

  % Mostrar resultados
  printf("Raíz aproximada: %.10f\n", xk);
  printf("Número de iteraciones: %d\n", k);
  printf("Error en la función: %.2e\n", erk);

end

function [xk,k,erk]=biseccion(f,a,b,tol,iterMax)
  fn=str2func(['@(x)' f]);  %Funcion Numerica (Usar)

  if fn(a) * fn(b) > 0
    error ('La función no cambia de signo en [a,b]')
  endif

  for k=0:iterMax
    xk= (a+b) / 2

    if fn(a)*fn(xk)<0
      b=xk
    else
      a=xk
    endif

    erk=abs(fn(xk))

    if erk<tol
      k = k+1;
      break
    endif
  endfor
end
