%This file plots the numerically simulated trajectory against the generated
%trajectory
file_path = 'C:\Users\nubapat\Data\last_reconstructed_samples_con_4000_grid.csv';
R = readmatrix(file_path);
x1nd = zeros(25, 1);
x2nd = zeros(25, 1);
d = zeros(6,1);
% Create a figure with a specified size
fig = figure('Position', [100, 100, 1200, 800]);
for i = 1:6
    k = 1;
    subplot(2, 3, i); % 2 rows, 3 columns, i-th subplot
    current_R = R(i, :)';
    [q_sim_traj, X1, Y1, X] = numgen(current_R);
    arrow_scale = 1;
    x1n = q_sim_traj(:, 1);
    x2n = q_sim_traj(:, 2);
    for j = 1:4:length(x1n)
        x1nd(k) = x1n(j);
        x2nd(k) = x2n(j);
        k = k + 1;
    end
    d1 = sqrt((X1 - x1nd).^2 + (Y1 - x2nd).^2);
    d(i) = norm(d1, 2)
    plot(x1nd, x2nd, '-o', 'MarkerSize', 4);  % Customize the plot style as needed
    hold on
    plot(X1, Y1, 'r', 'LineWidth', 1);
    hold on
    quiver(X(:, 1), X(:, 2), arrow_scale * X(:, 3), arrow_scale * X(:, 4), 'k', 'AutoScale', 'off');
    
    % Set the x and y-axis limits for the quiver plot
    % xlim([-1.5, 1.5]);
    % ylim([-1.5, 1.5]);
    xlabel('$r_{1}$', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('$r_{2}$', 'Interpreter', 'latex', 'FontSize', 14);
    %title(['Subplot ', num2str(i)]);
    legend('Numerically Simulated Trajectory', 'Generated Trajectory');
    set(gca, 'FontSize', 12);
    grid on;
    
end


