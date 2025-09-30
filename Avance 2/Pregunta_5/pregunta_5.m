% Parte C
% Construir A
clc;clear;close all

A = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9;
      1,20, 1, 2, 3, 4, 5, 6, 7, 8;
      2, 1,30, 1, 2, 3, 4, 5, 6, 7;
      3, 2, 1,40, 1, 2, 3, 4, 5, 6;
      4, 3, 2, 1,50, 1, 2, 3, 4, 5;
      5, 4, 3, 2, 1,60, 1, 2, 3, 4;
      6, 5, 4, 3, 2, 1,70, 1, 2, 3;
      7, 6, 5, 4, 3, 2, 1,80, 1, 2;
      8, 7, 6, 5, 4, 3, 2, 1,90, 1;
      9, 8, 7, 6, 5, 4, 3, 2, 1,100];

A
% Obtener determinante de A
detA = det_fact_lu(A)

% Crear la matriz de manera vectorizada
n = 10;
B = zeros(n);

for i = 1:n
    for j = 1:n
        if i == j
            B(i,j) = i * 10;  % Diagonal principal
        else
            B(i,j) = abs(i - j);  % Esto genera el patrón |i-j|
        end
    end
end

B
% Obtener determinante de A
detB = det_fact_lu(B)
