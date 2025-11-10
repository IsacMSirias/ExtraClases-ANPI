function pregunta_2()
  clc;clear;close all

  % Definir la función f(x)
  f = @(x) log(asin(x)) / log(x);
  a=0.1;
  b=0.8;
  xv= [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];
  x_0= 0.55;

  ct = cota_interpolacion(f,a,b,xv,x_0)

end

function ct = cota_interpolacion(f,a,b,xv,x_0)

  pkg load symbolic
  syms x

  nPlusOne=length(xv);
  fs=sym(f);
  fds=diff(fs,nPlusOne);

  fdn=matlabFunction(fds);   %Numerica para el Alpha MAX%


  %Maximo de la función en el intervalo
  newfdn=@(x) -1*abs(fdn(x));
  fmaxbnd=@(newfdn,a,b) fminbnd(newfdn,a,b);
  xmax=fmaxbnd(newfdn,a,b)

  alphamax=abs(fdn(xmax));


  n=nPlusOne-1;
  mult=1;
  for j=0:n
    mult=mult*(x_0-xv(j+1));
  endfor

  ct= (alphamax /(factorial(nPlusOne)))*abs(mult);

endfunction

