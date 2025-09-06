function pregunta_1()
   clc; clear;close all

   % Definir la función
   f = '(((x-3)*exp(x-3))+2)/2';

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

 end

