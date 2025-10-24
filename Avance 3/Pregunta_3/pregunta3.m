function pregunta3()
  clc; clear; close all;

  % DATOS DEL PROBLEMA
  %
  f = @(x, y) (x + y) / x;  % EDO dada
  a = 2;
  b = 10;
  y0 = 4;
  n = 6;
  h = (b - a)/n;
  x = a:h:b;

  % Solución analítica (exacta)
  y_exacta = @(x) (log(x) - log(2) + 2).*x;


  % MÉTODOS NUMÉRICOS
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


  % GRÁFICA

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

  % TABLA DE RESULTADOS

  fprintf('\n   x\t\tExacta\t\tEuler\t\tRK4\t\tAdams4\n');
  fprintf('-------------------------------------------------------------\n');
  for k = 1:length(x)
    fprintf('%.2f\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
      x(k), y_exacta(x(k)), y_euler(k), y_rk4(k), y_adams4(k));
  end

  fprintf('\nComparación completada. Gráfica generada correctamente.\n');
end



% MÉTODOS NUMÉRICOS

function [x,y]=metodo_euler(f,a,b,y0,n)
  h=(b-a)/n;
  x=a:h:b;
  y=zeros(1,n+1);
  y(1)=y0;
  for k=1:n
    y(k+1)=y(k)+h*f(x(k),y(k));
  end
end


function [x,y]=predictor_corrector(f,a,b,y0,n)
  h=(b-a)/n;
  x=a:h:b;
  y=zeros(1,n+1);
  y(1)=y0;
  for k=1:n
    y_pred=y(k)+h*f(x(k),y(k));
    y(k+1)=y(k)+(h/2)*(f(x(k),y(k))+f(x(k+1),y_pred));
  end
end


function [x,y]=metodo_rk2(f,a,b,y0,n)
  h=(b-a)/n;
  x=a:h:b;
  y=zeros(1,n+1);
  y(1)=y0;
  for k=1:n
    K1=f(x(k),y(k));
    K2=f(x(k)+h/2,y(k)+(h/2)*K1);
    y(k+1)=y(k)+h*K2;
  end
end


function [x,y]=metodo_rk3(f,a,b,y0,n)
  h=(b-a)/n;
  x=a:h:b;
  y=zeros(1,n+1);
  y(1)=y0;
  for k=1:n
    K1=f(x(k),y(k));
    K2=f(x(k)+h/2,y(k)+(h/2)*K1);
    K3=f(x(k)+h,y(k)-h*K1+2*h*K2);
    y(k+1)=y(k)+(h/6)*(K1+4*K2+K3);
  end
end


function [x,y]=metodo_rk4(f,a,b,y0,n)
  h=(b-a)/n;
  x=a:h:b;
  y=zeros(1,n+1);
  y(1)=y0;
  for k=1:n
    K1=f(x(k),y(k));
    K2=f(x(k)+h/2,y(k)+(h/2)*K1);
    K3=f(x(k)+h/2,y(k)+(h/2)*K2);
    K4=f(x(k)+h,y(k)+h*K3);
    y(k+1)=y(k)+(h/6)*(K1+2*K2+2*K3+K4);
  end
end


function [x, y] = metodo_taylor(fsym, a, b, y0, n, p)
  pkg load symbolic
  syms x y
  f_sym = fsym(x, y);
  derivs = cell(p, 1);
  derivs{1} = f_sym;
  for k = 2:p
    derivs{k} = diff(derivs{k-1}, x) + diff(derivs{k-1}, y)*f_sym;
  end
  f_funcs = cell(p, 1);
  for k = 1:p
    f_funcs{k} = matlabFunction(derivs{k}, "vars", [x y]);
  end
  h = (b - a)/n;
  x = a:h:b;
  y = zeros(1, n+1);
  y(1) = y0;
  for k = 1:n
    suma = 0;
    for i = 1:p
      suma = suma + (h^i / factorial(i)) * f_funcs{i}(x(k), y(k));
    end
    y(k+1) = y(k) + suma;
  end
end


function [x,y]=metodo_adams2(f,a,b,y0,n)
  h=(b-a)/n;
  x=a:h:b;
  y=zeros(1,n+1);
  y(1)=y0;
  for k=1:1
    K1=f(x(k),y(k));
    K2=f(x(k)+h/2,y(k)+(h/2)*K1);
    K3=f(x(k)+h/2,y(k)+(h/2)*K2);
    K4=f(x(k)+h,y(k)+h*K3);
    y(k+1)=y(k)+(h/6)*(K1+2*K2+2*K3+K4);
  end
  for k=2:n
    y(k+1)=y(k)+(h/2)*(3*f(x(k),y(k))-f(x(k-1),y(k-1)));
  end
end


function [x,y]=metodo_adams3(f,a,b,y0,n)
  h=(b-a)/n;
  x=a:h:b;
  y=zeros(1,n+1);
  y(1)=y0;
  for k=1:2
    K1=f(x(k),y(k));
    K2=f(x(k)+h/2,y(k)+(h/2)*K1);
    K3=f(x(k)+h/2,y(k)+(h/2)*K2);
    K4=f(x(k)+h,y(k)+h*K3);
    y(k+1)=y(k)+(h/6)*(K1+2*K2+2*K3+K4);
  end
  for k=3:n
    y(k+1)=y(k)+(h/12)*(23*f(x(k),y(k))-16*f(x(k-1),y(k-1))+5*f(x(k-2),y(k-2)));
  end
end


function [x,y]=metodo_adams4(f,a,b,y0,n)
  h=(b-a)/n;
  x=a:h:b;
  y=zeros(1,n+1);
  y(1)=y0;
  for k=1:3
    K1=f(x(k),y(k));
    K2=f(x(k)+h/2,y(k)+(h/2)*K1);
    K3=f(x(k)+h/2,y(k)+(h/2)*K2);
    K4=f(x(k)+h,y(k)+h*K3);
    y(k+1)=y(k)+(h/6)*(K1+2*K2+2*K3+K4);
  end
  for k=4:n
     y(k+1)=y(k)+(h/24)*(55*f(x(k),y(k))-59*f(x(k-1),y(k-1))+37*f(x(k-2),y(k-2))-9*f(x(k-3),y(k-3)));
  end
end

