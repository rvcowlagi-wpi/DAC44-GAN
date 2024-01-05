function hamil = hamplot(file)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
arr = readmatrix(file);
%arr = arr';
%arr = arr(2:end,:);
%arr = arr';
V = 0.05;
hamil = zeros(25,size(arr,1));

for j = 1:size(arr,1)
    for i = 1:25
    hamil(i,j) = 1 + arr(j,i+75)*V*cos(arr(j,i+50)) + arr(j,i+75)*arr(j,i+125) + arr(j,i+100)*V*sin(arr(j,i+50)) + arr(j,i+100)*arr(j,i+150);
  
    end
end   

end