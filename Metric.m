% Define the file name
file_name = 'C:\Users\nubapat\Data\y_gG.csv';
file_name1 = 'C:\Users\nubapat\Data\data_mintime_7.txt';
R1 = readmatrix(file_name);
R2 = readmatrix(file_name1);
R1 = R1';
%R1 = R1(2:end,:);
R1 = R1;
R2 = R2';
R2 = R2(51:75,1:64);
%R_1 = R2(:,1:2000);
%R_2 = R2(:,2001:4000);
results = statistics(R1);
results1 = statistics(R2);
% Extract the columns into separate variables
mean_real = results(:, 1);
variance_real = results(:, 2);
kurtosis_real = results(:, 3);
skewness_real = results(:, 4);

mean_gen = results1(:, 1);
variance_gen = results1(:, 2);
kurtosis_gen = results1(:, 3);
skewness_gen = results1(:, 4);
mse_m = mean((mean_real - mean_gen).^2);
mse_v = mean((variance_real - variance_gen).^2);
mse_k = mean((kurtosis_real - kurtosis_gen).^2);
mse_s = mean((skewness_real - skewness_gen).^2);


