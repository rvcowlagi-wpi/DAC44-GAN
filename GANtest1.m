%{
PROGRAM DESCRIPTION
-------------------
This script finds the MSE between the generated costates by the traditional GAN and the ones calculated by formula
and also plots the generated costates vs the costates calculated through formula.  
%}
file_path = 'C:\Users\nubapat\Data\y_gG.csv';
R = readmatrix(file_path);
R = R';
R = R(2:end,:); 
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
mse_gan = [mse1,mse2]; % The MSE errors for the costates
num_rows = size(P_costate_m, 1);

figure; % Create a new figure for the plots

% Assuming you have already computed the matrices P_costate_m and P1a_m

num_rows = size(P_costate_m, 1);

figure; % Create a new figure for the plots

for row_idx = 1:num_rows
    % Extract the row data for P_costate_m and P1a_m
    P_costate_row = P_costate_m(row_idx, :);
    P1a_row = P1a_m(row_idx, :);
    
    % Plot the row data
    subplot(num_rows, 1, row_idx);
    plot(P_costate_row, 'b', 'LineWidth', 2); % Plot P_costate_m in blue
    hold on;
    plot(P1a_row, 'r--', 'LineWidth', 2); % Plot P1a_m in red dashed line
    hold off;
    
    % Add title and axis labels
    title(['Row ' num2str(row_idx)]);
    xlabel('Index');
    ylabel('Value');
    
    % Add a legend to distinguish the plots
    legend('P\_costate\_m = generated', 'P1a\_m = real');
    
    % Adjust plot properties if needed
    % (e.g., xlim, ylim, grid, etc.)
end



