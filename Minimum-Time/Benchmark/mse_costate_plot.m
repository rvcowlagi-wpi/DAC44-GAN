% This file finds the MSE of the costate for generated trajectory
file_path = 'C:\Users\nubapat\Data\last_reconstructed_samples_4000_grid.csv';
file_path1 = 'C:\Users\nubapat\Data\last_reconstructed_samples_con_4000_grid.csv';
R_con = readmatrix(file_path1);
R = readmatrix(file_path);
%file_path2 = 'C:\Users\nubapat\Data\last_reconstructed_samples_con2.csv';
mse_hgan_hh = mse_costate(file_path);
mse_hgan_h = mse_costate(file_path1);
%mse_gan = mse_costate(file_path2);

% Calculate the total number of readings
num_readings = size(mse_hgan_hh, 1);

% Extract individual costate columns from each dataset
costate_p1_hh = mse_hgan_hh(:, 1);
costate_p2_hh = mse_hgan_hh(:, 2);
costate_p1p2_hh = costate_p1_hh + costate_p2_hh;
costate_p1_h = mse_hgan_h(:, 1);
costate_p2_h = mse_hgan_h(:, 2);
costate_p1p2_h = costate_p1_h + costate_p2_h;
% costate_gan1 = mse_gan(:, 1); % Using all columns of mse_gan
% costate_gan2 = mse_gan(:, 2);
% costate_gan = costate_gan1 + costate_gan2;
% Create a line plot
figure;

% Plot for the first column
% Create subplots with 2 rows and 1 column
semilogy(1:num_readings, costate_p1p2_hh, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'b');
hold on;
semilogy(1:num_readings, costate_p1p2_h, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'r');
%semilogy(1:num_readings, costate_gan, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'g'); % Plotting the first column of mse_gan
hold off;
%title('MSE Errors for Costate p_{1} for the GANs');
xlabel('\textbf{Generated Output} $\mathbf{z}_{i}^{g}$', 'Interpreter', 'latex', 'FontWeight', 'bold');
ylabel('\bf{\sigma}_{3i}', 'Interpreter', 'tex');
xlim([1, num_readings]); % Set x-axis limits
% Add a tick label for the last value
xticks([1, num_readings]); % Set tick positions
xticklabels({'1', num2str(num_readings)}); % Set tick labels
grid on;
legend('VAE', 'ZVAE1');
% Set the font size for the axis tick labels
set(gca, 'FontSize', 12); % Adjust the font size as needed
% Save the figure to a specific folder
save_folder = 'C:\Users\nubapat\Data\Paper1_results';  % Change this to your desired folder path
save_name = 'sigma_costate_4000grid.png';  % Change this to your desired file name
save_path = fullfile(save_folder, save_name);

% Save the figure
saveas(gcf, save_path);