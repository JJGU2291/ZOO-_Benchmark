function plot_draw_save_att(num_tracker, plot_style, att_metric, rank_num, ...
                            ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, ...
                            x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf)
% 为属性帧级别评估结果绘图并保存
% 输入参数：
%   att_metric: 二维数组 (跟踪器数量 × 阈值数量)，每个跟踪器在该属性下的指标
%   其他参数同原函数，适配帧级数据（无序列维度）

% 计算每个跟踪器的性能分数（AUC或特定阈值）
perf = zeros(1, num_tracker);
for i = 1:num_tracker
    tracker_values = att_metric(i, :);  % 该跟踪器在所有阈值下的指标
    switch ranking_type
        case 'AUC'
            % AUC取所有阈值的平均值
            perf(i) = mean(tracker_values);
        case 'threshold'
            % 特定阈值下的指标（如误差阈值5px或重叠率0.5）
            perf(i) = tracker_values(rank_idx);
    end
end

% 按性能分数降序排序
[~, index_sort] = sort(perf, 'descend');

% 绘图设置
font_size = 14;
font_size_legend = 14;
axex_font_size = 14;

% 创建图形
tmp_figure = figure;
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.45, 0.6]);  % 调整图形位置和大小

% 绘制曲线
tmp_axes = axes('Parent', tmp_figure, 'FontSize', axex_font_size);
hold on;
legend_names = cell(rank_num, 1);  % 存储图例名称

for i = 1:rank_num
    tracker_idx = index_sort(i);  % 排序后的跟踪器索引
    tracker_values = att_metric(tracker_idx, :);  % 该跟踪器的指标曲线
    
    % 计算当前跟踪器的分数（用于图例）
    switch ranking_type
        case 'AUC'
            score = mean(tracker_values);
        case 'threshold'
            score = tracker_values(rank_idx);
    end
    legend_names{i} = sprintf('[%0.3f] %s', score, name_tracker_all{tracker_idx});
    
    % 绘制曲线（使用指定的颜色和线型）
    plot(threshold_set, tracker_values, ...
         'color', plot_style{i}.color, ...
         'lineStyle', plot_style{i}.lineStyle, ...
         'lineWidth', 2, ...
         'Parent', tmp_axes);
end

% 设置网格和坐标轴
grid on;
set(gca, 'GridLineStyle', ':', 'GridColor', 'k', 'GridAlpha', 0.7, 'LineWidth', 1.2);
xlabel(x_label_name, 'fontsize', font_size);
ylabel(y_label_name, 'fontsize', font_size);
title(title_name, 'fontsize', font_size);

% 设置图例
if strcmp(ranking_type, 'threshold')
    legend_pos = 'Northwest';
else
    legend_pos = 'Northeast';
end
legend(legend_names, 'Interpreter', 'none', 'fontsize', font_size_legend, 'Location', legend_pos);

hold off;

% 保存图片
if ~exist(save_fig_path, 'dir')
    mkdir(save_fig_path);
end
full_path = fullfile(save_fig_path, fig_name);
if strcmpi(save_fig_suf, 'eps')
    print(tmp_figure, '-depsc', full_path);
else
    saveas(tmp_figure, full_path, save_fig_suf);
end
close(tmp_figure);  % 关闭图形释放内存

end