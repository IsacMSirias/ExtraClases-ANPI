function pregunta_1()
   clc; clear;close all

   % Definir la función f(x)
   f = '((x-3)*exp(x-3)+2)/2';

   % Parámetros
   a = 0;
   b = 7/3;

   % Llamar al teorema
   c = unicidad_pf(f,a,b);

end

function c = unicidad_pf(f,a,b)
  % unicidad_pf - Verifica existencia y unicidad de punto fijo en [a,b]
  % Parámetros:
  %   f: función en formato string
  %   a, b: extremos del intervalo
  % Retorna:
  %   c: 1 si hay punto fijo único, 0 en caso contrario

  pkg load symbolic
  syms x

  % Crear funciones numéricas y simbólicas
  fn=str2func(['@(x)' f]);  %Funcion Numerica (Usar)

  fs=sym(f);
  fds=diff(fs);      % Primera derivada simbólica
  fds2=diff(fs,2);   % Segunda derivada simbólica
  fdn=matlabFunction(fds);  %Funcion Numerica de derivada (Usar)
  fdn2=matlabFunction(fds2);%Funcion Numerica de 2da derivada (Usar)

  % TEOREMA DE EXISTENCIA
  % Verificar que f(x) mapea [a,b] en [a,b]

  % Encontrar puntos críticos de la función en el intervalo [a,b]
  xsol=fzero(fdn,[a,b]);

  % Evaluar puntos críticos y extremos del intervalo en f(x)
  crit = fn(xsol);
  fa = fn(a);
  fb = fn(b);

  % Verificar que fa y fb estén en el intervalo [a,b]
  if fa >= a && fa <= b && fb >= a && fb <= b
    fprintf('f(x) tiene al menos un punto fijo en [a,b] \n');
    existencia = true;
  else
    fprintf('No se cumple el teorema de existencia \n');
    existencia = false;
  endif

  % TEOREMA DE UNICIDAD
  % Verificar que |f'(x)| < 1 para todo x en [a,b]

  if existencia
    % Encontrar máximo de |f'(x)| en el intervalo
    xsol2=fzero(fdn2,[a,b]);

    % Evaluar derivada en puntos críticos y extremos
    critfd = fdn(xsol2);
    fda = fdn(a);
    fdb = fdn(b);

    % Verificar condición de contractividad
    if fda > -1 && fda < 1 && fdb > -1 && fdb <= 1
      unicidad = 1;
      fprintf("f(x) tiene un único punto fijo en el intervalo [a,b] \n")
    else
      unicidad = 0;
    endif
  else
    fprintf("No se cumple el teorema de existencia \n")
  endif

  c = unicidad
end
