function y = statistics(R)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

% Calculate the mean for each row
row_means = mean(R, 2);

% Calculate the variance for each row
row_variances = var(R, 0, 2); 

% Calculate the kurtosis for each row
row_kurtoses = kurtosis(R, 0, 2); 

% Calculate the skewness for each row
row_skewnesses = skewness(R, 0, 2); 
y = [row_means,row_variances,row_kurtoses ,row_skewnesses];

end