function mse = mse_head(file)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
R = readmatrix(file);
R = R';
%R = R(2:end,:); 
psic_m = zeros(25,1);
D = zeros(25,size(R,2));
mse = zeros(size(R,2),1);
for i = 1:size(R,2)
    psia = R(51:75,i);
    p1a = R(76:100,i);
    p2a = R(101:125,i);
    for j =1:25 
        psic = h_angle(p1a(j),p2a(j));
        psic_m(j) = psic;
    end
    d1 = psia - psic_m;
    D(:,i) = d1;
    mse(i) = norm(d1) / sqrt(length(d1));
    
end  
end