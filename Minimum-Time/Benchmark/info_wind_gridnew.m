function wind_grid = info_wind_gridnew(wind_params)
    % Define grid coordinates
    x_grid = linspace(-2, 2, 5);
    y_grid = linspace(-2, 2, 5);
    [X_grid, Y_grid] = meshgrid(x_grid, y_grid);

    % Initialize matrices
    [wind_x1, wind_x2, wg11, wg12, wg21, wg22] = deal(zeros(size(X_grid)));

    % Function to calculate wind
    wind_fcn = @calculate_wind02;

    % Calculate wind parameters on the grid
    for i = 1:numel(X_grid)
        [wind_x1(i), wind_x2(i), wg11(i), wg12(i), wg21(i), wg22(i)] = wind_fcn(X_grid(i), Y_grid(i), wind_params);
    end

    % Reshape matrices for concatenation
    wind_x1 = reshape(wind_x1, size(X_grid));
    wind_x2 = reshape(wind_x2, size(X_grid));
    wg11 = reshape(wg11, size(X_grid));
    wg12 = reshape(wg12, size(X_grid));
    wg21 = reshape(wg21, size(X_grid));
    wg22 = reshape(wg22, size(X_grid));

    % Create wind_grid
    wind_grid = [X_grid(:), Y_grid(:), wind_x1(:), wind_x2(:), wg11(:), wg12(:), wg21(:), wg22(:)];
end
