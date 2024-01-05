function mse_hgan = mse_costate(file)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
R = readmatrix(file);
R = R';
%R = R(2:end,:); 
P_costate = zeros(2,25);
P_costate_m = zeros(2*size(R,2),25);
P1a_m = zeros(2*size(R,2),25);
mse1 = zeros(1,size(R,2));
mse2 = zeros(1,size(R,2));
for i = 1:size(R,2)
    psia = R(51:75,i);
    w1a = R(126:150,i);
    w2a = R(151:175,i);
    V = 0.05;
    p1a = R(76:100,i)';
    p2a = R(101:125,i)';
    P1a = [p1a;p2a];
    P1a_m(2*i-1:2*i,:) = P1a;
    for j = 1:25
        P = costate(psia(j),w1a(j),w2a(j),V);
        P_costate(:,j) = P;
    end
    P_costate_m(2*i-1:2*i,:) = P_costate; 
    d1 = p1a - P_costate(1,:);
    d2 = p2a - P_costate(2,:);
    mse1(i) = norm(d1) / sqrt(length(d1));
    mse2(i) = norm(d2) / sqrt(length(d2));
end
mse1 = mse1'; 
mse2 = mse2';
mse_hgan = [mse1,mse2];

end