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

  pkg load symbolic
  syms x

  fn=str2func(['@(x)' f]);  %Funcion Numerica (Usar)

  fs=sym(f);
  fds=diff(fs);
  fds2=diff(fs,2);
  fdn=matlabFunction(fds);  %Funcion Numerica de derivada (Usar)
  fdn2=matlabFunction(fds2);%Funcion Numerica de 2da derivada (Usar)

  %Teorema de Existencia

  %Encontrar puntos criticos de la función, en el intervalo [a,b]
  %Cuando f'(x)=0, o la función se indefine
  xsol=fzero(fdn,[a,b]);


  %Evaluar puntos criticos y extremos del intervalo en h(x)
  crit = fn(xsol);
  fa = fn(a);
  fb = fn(b);

  %Verificar que fa y fb estén en el intervalo [a,b]

  if fa >= a && fa <= b && fb >= a && fb <= b
    fprintf('f(x) tiene al menos un punto fijo en [a,b] \n');
    existencia = true;
  else
    fprintf('No se cumple el teorema de existencia \n');
    existencia = false;
  endif

  %Teorema de Unicidad

  if existencia
    xsol2=fzero(fdn2,[a,b]);

    critfd = fdn(xsol2);
    fda = fdn(a);
    fdb = fdn(b);

    if fda >= -1 && fda <= 1 && fdb >= -1 && fdb <= 1
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
