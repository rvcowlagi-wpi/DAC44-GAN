% This file finds the MSE of the hamiltonian for generated trajectory
file_path = 'C:\Users\nubapat\Data\last_reconstructed_samples_4000_grid.csv';
file_path1 = 'C:\Users\nubapat\Data\last_reconstructed_samples_con_4000_grid.csv';
%file_path2 = 'C:\Users\nubapat\Data\last_reconstructed_samples_con2.csv';
Y = readmatrix(file_path);
H = hamplot(file_path);
H1 = hamplot(file_path1);
%H2 = hamplot(file_path2);
mseH = zeros(1,size(H,2));
mseH1 = zeros(1,size(H1,2));
%mseH2 = zeros(1,size(H2,2));
for i = 1: size(H,2) 
    mseH(i) = norm(H(:,i))/sqrt(length(H(:,i)));
    mseH1(i) = norm(H1(:,i))/sqrt(length(H1(:,i)));
    %mseH2(i) = norm(H2(:,i))/sqrt(length(H2(:,i)));
end

% Assuming the code above is already executed and 'H' and 'H1' are defined.

% Define the column numbers for the 9 columns you want to plot
columns_to_plot = [3, 6, 9, 12, 15, 18, 21, 24];

% Create a figure with 3x3 subplots for the selected columns
figure;

% for i = 1:numel(columns_to_plot)
%     subplot(3, 3, i);
%     
%     % Plot both 'H' and 'H1' on the same subplot
%     plot(H(:, columns_to_plot(i)), 'o-','LineWidth', 1,'MarkerFaceColor', 'b'); % Plot H in blue
%     hold on;
%     plot(H1(:, columns_to_plot(i)), 'r'); % Plot H1 in red
%     hold on;
%     plot(H2(:, columns_to_plot(i)), 'g'); % Plot H1 in red
%     hold off;
%     
%     % Set axis labels and title for each subplot
%     xlabel('Points of discretization');
%     ylabel(sprintf('Hamiltonian Values ', columns_to_plot(i))); % Add column number to ylabel
%     
%     % You can customize the subplot further if needed (e.g., grid, legends, etc.)
%     grid on;
%     legend('VAE', 'ZVAE','GAN' ); % Add legend for the plotted lines
%     
% end

% Adjust the layout to prevent overlapping of subplots
sgtitle('Plots of the Hamiltonians of the generated trajectories', 'FontSize', 16);
% ... (previous code)

% Create a figure for mseH
figure;
semilogy(mseH,'o-','LineWidth', 2,'MarkerFaceColor', 'b');

 % Highlight specific points

hold on;
semilogy(mseH1,'o-','LineWidth', 2,'MarkerFaceColor', 'r');
%hold on

%semilogy(mseH2,'o-','LineWidth', 2,'MarkerFaceColor', 'g');
% Highlight points where mseH and mseH1 have values
% scatter(find(mseH ~= 0), mseH(mseH ~= 0), 'bo', 'filled');
% scatter(find(mseH1 ~= 0), mseH1(mseH1 ~= 0), 'ro', 'filled');
% scatter(find(mseH2 ~= 0), mseH2(mseH2 ~= 0), 'go', 'filled');

hold off;
xlabel('\textbf{Generated Output} $\mathbf{z}_{i}^{g}$', 'Interpreter', 'latex', 'FontWeight', 'bold');
ylabel('\bf{\sigma}_{1i}', 'Interpreter', 'tex');
%xlim([1, num_readings]); % Set x-axis limits
% Add a tick label for the last value
%xticks([1, num_readings]); % Set tick positions
%xticklabels({'1', num2str(num_readings)}); % Set tick labels
grid on;
legend('SVAE', 'ZVAE1');
%title('Mean Squared Error (MSE) comparison of Hamiltonian for GANs)');
% Set the font size for the axis tick labels
set(gca, 'FontSize', 12); % Adjust the font size as needed
% Save the figure to a specific folder
save_folder = 'C:\Users\nubapat\Data\Paper1_results';  % Change this to your desired folder path
save_name = 'sigma_ham_4000grid.png';  % Change this to your desired file name
save_path = fullfile(save_folder, save_name);

% Save the figure
saveas(gcf, save_path);
