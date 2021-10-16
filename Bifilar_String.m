% Bifilar Pendulum: Natural Frequencies and Mode Shapes 

% Author:  Gary S. Schajer, 2020.

% Variables:
% ----------
%  a1 = offset ratio 1
%  a2 = offset ratio 2
%  cp = center of percussion position, m
%  D  = distance between strings, m
%  f  = natural frequency, Hz
%  g  = gravitational acceleration = 9.81 m/s^2
%  i  = index for length ratio plotting
%  j  = index for eigenvectors
%  K  = stiffness matrix, N/m
%  L1 = string length 1, m
%  L2 = string length 2, m
%  LL = pendulum rod length, m
%  LR = length ratio = L2/L1
%  m  = pendulum mass, kg
%  M  = mass matrix, kg
%  np = nodal point position, m
%  R  = radius of gyration of pendulum rod, m
%  s  = pendulum centroid offset, m
%  v  = eigenvector ratio, V(2,:)/V(1,:)
%  V  = eigenvector matrix
%  vx = string displacement eigenvector ratio
%  Vx = string displacement eigenvector matrix
%  w2 = angular frequency squared, (rad/s)^2

%************************************************

% Initialize variables
clear all;
close all;
m = 1;   % any positive number will work 
g = 9.81;
LL = 1;
R = LL/sqrt(12);
L1 = 0.70;
L2 = 0.70;
D = 0.60;   % = 2.0000001*R;  for Fig. 7
D = 0.6;
s = 0;
a1 = 0.5 + s/D;
a2 = 0.5 - s/D;

% Solve for length ratios in the range 0.7 to 1.3
for i = 1:13
  L2 = L1 * (0.65 + 0.05*i);
  LR(i) = L2 / L1;
  M = (m/(D))*[[((a2^2)*(D^2)+R^2) ((a1*a2)*(D^2)-R^2)]' [((a1*a2)*(D^2)-R^2) ((a1^2)*(D^2)+R^2)]'];
  K = D*(a1+a2)*[[((a2*m*g)/L1) 0]' [0 (a1*m*g)/L2]'];
  [V,w2] = eig(K,M,'vector');
  [w2,index] = sort(w2);
  V = V(:,index);
  f(i,:) = sqrt(w2) / 2 / pi;
  
  % Find mode shape ratio and nodal point position
  v(i,:) = V(2,:) ./ V(1,:);
  Vx = V;
  Vx(:,:) = Vx(:,:) ./ norm(Vx(:,:));
  vx(i,:) = Vx(2,:) ./ Vx(1,:);
  np(i,:) = (a2*Vx(1,:)+a1*Vx(2,:))*D./(Vx(1,:)-Vx(2,:));
  if abs(np(i,1)) > 10
    np(i,1) = NaN;
  end
  cp(i,:) = -R^2 ./ np(i,:);
end

% Draw nodal position vs length ratio plot
figure(3)
hold on
plot(LR,np)
plot(1, 0, 'or')
text(LR(2),np(5,1),'mode 1')
text(LR(2),np(6,2),'mode 2')
text(LR(11),np(9,1),'mode 2')
text(LR(11),np(8,2),'mode 1')
title('Nodal Position vs. String Length Ratio [String Based Coordinates]')
xlabel('String Length Ratio,  L2/L1')
ylabel('Nodal Position,  m')

% Draw mode shape factor vs length ratio plot
figure(2)
hold on
plot(LR,vx)
plot(1,  1, 'ob')
plot(1, -1, 'or')
text(LR(2),vx(8,1),'mode 1')
text(LR(2),vx(1,2),'mode 2')
text(LR(11),vx(6,2),'mode 1')
text(LR(11),vx(13,1),'mode 2')
title('Mode Shape Factor vs. String Length Ratio [String Based Coordinates]')
xlabel('String Length Ratio,  L2/L1')
ylabel('Mode Shape Factor')

% Draw frequency vs length ratio plot
figure(1)
hold on
plot(LR,f)
plot(1, sqrt(g/L1)/2/pi, 'ob')
text(LR(2),f(8,1),'mode 1')
text(LR(2),f(1,2),'mode 2')
text(LR(11),f(6,2),'mode 1')
text(LR(11),f(13,1),'mode 2')
title('Natural Frequency vs. String Length Ratio [String Based Coordinates]')
xlabel('String Length Ratio,  L2/L1')
ylabel('Natural Frequency, Hz')

