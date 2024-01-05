% This file finds the MSE of the heading angle for generated trajectory
file_path = 'C:\Users\nubapat\Data\last_reconstructed_samples_500_grid.csv';
file_path1 = 'C:\Users\nubapat\Data\last_reconstructed_samples_con_500_grid.csv';
%file_path2 = 'C:\Users\nubapat\Data\last_reconstructed_samples_con2.csv';
H_angle_hh = mse_head(file_path);
H_angle_h = mse_head(file_path1);
%H_angle = mse_head(file_path2);

% Calculate the total number of readings
num_readings = size(H_angle_hh, 1);

% Create a line plot for the angle data
figure;
%subplot(2, 1, 1);
semilogy(1:num_readings, H_angle_hh, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'b');
hold on;
semilogy(1:num_readings, H_angle_h, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'r');

%semilogy(1:num_readings, H_angle, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'g');
hold off;
%title('MSE Heading Angle for the GANs');
xlabel('\textbf{Generated Output} $\mathbf{z}_{i}^{g}$', 'Interpreter', 'latex', 'FontWeight', 'bold');
ylabel('\bf{\sigma}_{2i}', 'Interpreter', 'tex');
xlim([1, num_readings]); % Set x-axis limits
% Add a tick label for the last value
xticks([1, num_readings]); % Set tick positions
xticklabels({'1', num2str(num_readings)}); % Set tick labels
grid on;
legend('SVAE', 'ZVAE1');
% Set the font size for the axis tick labels
set(gca, 'FontSize',12); % Adjust the font size as needed
% Save the figure to a specific folder
save_folder = 'C:\Users\nubapat\Data\Paper1_results';  % Change this to your desired folder path
save_name = 'sigma_head_500grid.png';  % Change this to your desired file name
save_path = fullfile(save_folder, save_name);

% Save the figure
saveas(gcf, save_path);