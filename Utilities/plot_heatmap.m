function plot_heatmap(A)
%PLOT_HEATMAP Plot a zero-centered heatmap with grid lines

    if ~ismatrix(A)
        error('Input must be a 2D matrix.');
    end

    figure;
    ax = gca;

    % symmetric color limits
    maxabs = max(abs(A(:)));
    if maxabs == 0
        maxabs = 1;
    end

    imagesc(ax, A);
    axis(ax, 'tight');
    set(ax, 'YDir', 'normal');

    caxis(ax, [-maxabs maxabs]);    % CRITICAL

    % horizontal grid lines every 10 rows
    [n, ~] = size(A);
    hold(ax, 'on');
    for y = 10.5:10:n
        yline(ax, y, 'k-', 'LineWidth', 0.5, 'Alpha', 0.6);
    end
    hold(ax, 'off');

    % diverging colormap
    cmap = bluewhitered(256);
    colormap(ax, cmap);
    colorbar(ax);

    xlabel(ax, 'Column index');
    ylabel(ax, 'Row index');
    title(ax, 'Heat map (zero-centered)');
end


function cmap = bluewhitered(m)
%BLUEWHITERED Create a blue-white-red diverging colormap

    if nargin < 1
        m = 256;
    end

    bottom = [0 0 0.8];
    middle = [1 1 1];
    top    = [0.8 0 0];

    cmap = zeros(m,3);
    t = linspace(0,1,m);

    for i = 1:3
        cmap(:,i) = interp1([0 0.5 1], [bottom(i) middle(i) top(i)], t);
    end
end