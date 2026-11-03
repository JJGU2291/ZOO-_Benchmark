function Draw_radar2(data, lim, labels, legendlabels)
    hold on;
    % 1. 定义 7 种高区分度颜色
    colors = [
        0.85, 0.33, 0.10;  % 橙
        0.00, 0.45, 0.74;  % 蓝
        0.47, 0.67, 0.19;  % 绿
        0.49, 0.18, 0.56;  % 紫
        0.93, 0.69, 0.13;  % 黄
        0.00, 1.00, 1.00;  % 青
        0.64, 0.08, 0.18   % 深红
    ];
    % 2. 定义标记和线型
    markers = {'o', 's', '^', 'd'};
    linestyles = {'-', '--'};

    num_trackers = size(data, 1);
    
    for i = 1:num_trackers
        % 通过取模循环组合样式
        c_idx = mod(i-1, size(colors, 1)) + 1;
        m_idx = mod(i-1, length(markers)) + 1;
        l_idx = mod(i-1, length(linestyles)) + 1;
        
        % 构建样式结构体
        style.color = colors(c_idx, :);
        style.marker = markers{m_idx};
        style.line = linestyles{l_idx};
        
        % 调用底层绘图函数
        H = Draw_radar(data(i,:), lim, labels, style);
        
        % 隐藏辅助线的图例，只保留主数据线
        for j = 1:length(H)-1
            set(get(get(H{j},'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        end
    end
    
    % 设置图例，置于右侧防止遮挡
    legend(legendlabels, 'Interpreter', 'none', 'Location', 'eastoutside', 'FontSize', 9);
    grid on;
end