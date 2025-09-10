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
  fdn=matlabFunction(fds);  %Funcion Numerica de derivada (Usar)

  %Teorema de Existencia

  %Encontrar puntos criticos de la función, en el intervalo [a,b]
  %Cuando f'(x)=0, o la función se indefine
  xsol=fzero(fdn,[a,b]);


  %Evaluar puntos criticos y exttremos del intervalo en h(x)
  crit = fn(xsol)
  fa = fn(a)
  fb = fn(b)

  if fa >= a && fa <= b && fb >= a && fb <= b
    fprintf('La función tiene al menos un punto fijo en [a,b] \n');
    existencia = true;
  else
    fprintf('No se cumple el teorema de existencia');
    existencia = false;
  endif

  %Teorema de Unicidad

  if existencia

  endif


  %xd = 6<8;
  %if xd == 1;
    %printf("hola");
  %else
    %printf("adios");
  %endif;


  c = 1;

end


%Ejemplo 1: 'fminsearch' = Calcular  el mínimo de una función f, dado un valor inicial.
%x0=50;
%xmin=fminsearch(f1,x0);
%ymin=f1(xmin);

%Ejemplo 2: 'fzero' = Calcular una solución de la ecuación f(x)=0, dado un valor inicial
%x0=-3;
%xsol=fzero(f1,x0);

%Ejemplo3: 'fminbnd' = Calcular el mínimo de una función f en un intervalo [a,b]
%a=-4;
%b=-2;
%xmin=fminbnd(f1,a,b);

%Ejercicio: Calcular el máximo de una funcion f en un intervalo [a,b]
%a=1;
%b=3;
%newf1=@(x) -1*f1(x); %Reflejar f1 en el eje x
%fmaxbnd=@(newf1,a,b) fminbnd(newf1,a,b);
%xmax=fmaxbnd(newf1,a,b)
