%% 帧级别挑战属性评估工具（按帧归类计算）
clc; clear all; close all; warning off;

addpath('./utils/');
addpath('./sequence_evaluation_config/');

% 配置路径
tmp_mat_path  = './tmp_mat/frame_att/';
path_anno     = './annos/zoo_anno/';      % 序列标注路径
path_frame_att = './annos/att/';    % 帧级别属性标注路径（每个序列对应一个帧属性文件）
rp_all        = './zoo_tracking_results/';
save_fig_path = './res_fig/frame_att/';
mkdir(tmp_mat_path);
mkdir(save_fig_path);

% 挑战属性名称（包含"Out of View"）
att_name = ["Out of View", 'Occlusion', "Fast Motion", "Similar Object", ...
            "Deformation", "Background Mixing", "Rotation", "Small Target"];
att_fig_name = {'OV','OC', 'FM', 'SO', 'DEF', 'BM', 'RO', 'ST'};
num_att = length(att_name);  % 8个属性

% 评估参数
evaluation_dataset_type = 'test_set';
norm_dst = true;
eval_type = 'OPE';
ranking_type = 'AUC';
rank_num = 50;
metric_type_set = {'error', 'overlap'};

% 加载跟踪器和序列
trackers = config_tracker();
sequences = config_sequence(evaluation_dataset_type);
plot_style = config_plot_style();
num_tracker = numel(trackers);
num_seq = numel(sequences);
% 修正后代码：
name_tracker_all = cell(num_tracker, 1);  % 先初始化单元格数组
for i = 1:num_tracker
    name_tracker_all{i} = trackers{i}.name;  % 逐个访问单元格中的结构体
end

% 初始化每个属性的帧数据存储（跟踪器×属性×帧数据）
% 存储格式：cell(tracker_idx, att_idx) = 该跟踪器在该属性下所有帧的指标
att_frame_data_overlap = cell(num_tracker, num_att);
att_frame_data_error = cell(num_tracker, num_att);
for t = 1:num_tracker
    for a = 1:num_att
        att_frame_data_overlap{t,a} = [];  % 存储重叠率
        att_frame_data_error{t,a} = [];    % 存储中心误差
    end
end

% 阈值设置
threshold_set_error = 0:50;
if norm_dst
    threshold_set_error = threshold_set_error / 100;
end
threshold_set_overlap = 0:0.05:1;

rank_num = num_tracker;

%% 核心：按帧遍历并归类到属性
for seq_idx = 1:num_seq
    s = sequences{seq_idx};
    fprintf('处理序列: %s\n', s);
    
    % 加载序列标注和跟踪结果
    anno = dlmread([path_anno s '/groundtruth.txt']);
    absent_anno = dlmread([path_anno s '/absent.txt']);
    absent_anno = 1 - absent_anno;  % 1表示存在，0表示缺失
    seq_len = size(anno, 1);
    
    % 加载帧级别属性标注（每行对应一帧，列对应属性，1表示存在该属性）
    frame_att_file = [path_frame_att s '.txt'];
    if ~exist(frame_att_file, 'file')
        warning('帧属性文件不存在: %s', frame_att_file);
        continue;
    end
    frame_att = dlmread(frame_att_file, ',');  % 尺寸: seq_len × 8（8个属性，包含Out of View）
    
    % 特殊处理：标记Out of View之后的恢复帧
    % 1表示当前帧是Out of View或之后的恢复帧，0表示正常帧
    out_of_view_recovery = zeros(seq_len, 1);
    has_out_of_view = false;  % 标记是否出现过Out of View
    
    for frame_idx = 1:seq_len
        % 检查当前帧是否为Out of View
        if frame_att(frame_idx, 1) == 1
            has_out_of_view = true;
            out_of_view_recovery(frame_idx) = 1;
        elseif has_out_of_view && absent_anno(frame_idx) == 0
            % 如果之前出现过Out of View，且当前帧是有效可见帧，则标记为恢复帧
            out_of_view_recovery(frame_idx) = 1;
        end
    end
    
    % 更新Out of View属性：包含原始Out of View帧和之后的恢复帧
    frame_att(:, 1) = out_of_view_recovery;

    % 遍历每个跟踪器
    for tracker_idx = 1:num_tracker
        t = trackers{tracker_idx};
        res_file = [rp_all t.name '_tracking_result/' s '.txt'];
        if ~exist(res_file, 'file')
            warning('跟踪结果文件不存在: %s', res_file);
            continue;
        end
        res = dlmread(res_file);

        % 计算每帧的重叠率和中心误差
        [err_coverage, err_center] = calc_seq_err_robust(res, anno, absent_anno, norm_dst);
        
        % 记录有效帧的原始索引（未被过滤的帧）
        valid_frames = find(absent_anno == 0);  % 有效帧的原始索引列表
        
        % 遍历每帧，按属性归类（基于原始帧索引）
        for frame_idx = 1:seq_len
            % 跳过缺失帧（absent_anno=1）
            if absent_anno(frame_idx) == 1
                continue;
            end
            
            % 找到当前有效帧在err_coverage/err_center中的对应位置
            [is_valid, err_idx] = ismember(frame_idx, valid_frames);
            if ~is_valid || err_idx < 1 || err_idx > length(err_coverage)
                warning('帧 %d 有效但未找到对应误差值', frame_idx);
                continue;
            end
            
            % 获取当前有效帧的误差值
            current_coverage = err_coverage(err_idx);
            current_center = err_center(err_idx);
            
            % 检查当前帧属于哪些属性（8个属性中值为1的）
            frame_att_current = frame_att(frame_idx, :);  % 1×8向量
            for att_idx = 1:num_att
                if frame_att_current(att_idx) == 1  % 该帧属于当前属性
                    % 将该帧的指标加入对应属性的存储
                    att_frame_data_overlap{tracker_idx, att_idx} = [att_frame_data_overlap{tracker_idx, att_idx}; current_coverage];
                    att_frame_data_error{tracker_idx, att_idx} = [att_frame_data_error{tracker_idx, att_idx}; current_center];
                end
            end
        end
    end
end

%% 计算每个属性的整体指标并绘图
for metric_idx = 1:numel(metric_type_set)
    metric_type = metric_type_set{metric_idx};
    if strcmp(metric_type, 'error')
        threshold_set = threshold_set_error;
        rank_idx = 21;
        x_label = 'Location error threshold';
        y_label = 'Precision';
        data_cell = att_frame_data_error;
    else
        threshold_set = threshold_set_overlap;
        rank_idx = 11;
        x_label = 'Overlap threshold';
        y_label = 'Success rate';
        data_cell = att_frame_data_overlap;
    end

    % 遍历每个属性
    for att_idx = 1:num_att
        % 检查该属性是否有足够的帧数据
        total_frames = 0;
        for t = 1:num_tracker
            total_frames = total_frames + length(data_cell{t, att_idx});
        end
        if total_frames < 10  % 至少需要10帧才计算
            fprintf('属性 %s 有效帧不足，跳过\n', att_name{att_idx});
            continue;
        end
        fprintf('计算属性 %s 指标（总帧数: %d）\n', att_name{att_idx}, total_frames/num_tracker);

        % 计算每个跟踪器在该属性下的整体指标
        att_metric = zeros(num_tracker, length(threshold_set));
        for tracker_idx = 1:num_tracker
            frame_values = data_cell{tracker_idx, att_idx};
            if isempty(frame_values)
                att_metric(tracker_idx, :) = 0;
                continue;
            end
            % 计算每个阈值下的成功率
            for t_idx = 1:length(threshold_set)
                if strcmp(metric_type, 'error')
                    % 误差指标：小于等于阈值为成功
                    att_metric(tracker_idx, t_idx) = mean(frame_values <= threshold_set(t_idx));
                else
                    % 重叠率指标：大于阈值为成功
                    att_metric(tracker_idx, t_idx) = mean(frame_values > threshold_set(t_idx));
                end
            end
        end

        % 绘图
        % 确定数据集名称
        if strcmp(evaluation_dataset_type, 'all')
            dataset_name = 'ZOO++';
        else
            dataset_name = 'ZOO++ Testing Set';
        end
        
        % 构建fig_name
        if strcmp(metric_type, 'overlap')
            metric_fig_suffix = 'Success';
        elseif strcmp(metric_type, 'error')
            if norm_dst
                metric_fig_suffix = 'NormPrecision';
            else
                metric_fig_suffix = 'Precision';
            end
        end
        fig_name = [att_fig_name{att_idx} '_' metric_fig_suffix '_' ranking_type];
        
        % 构建title_name
        switch metric_type
            case 'overlap'
                title_core = ['Success plots of ' att_name{att_idx}];
            case 'error'
                if norm_dst
                    precision_prefix = 'Normalized Precision';
                else
                    precision_prefix = 'Precision';
                end
                title_core = [precision_prefix ' plots of ' att_name{att_idx}];
        end
        title_name = [title_core ' on ' dataset_name];

        plot_draw_save_att(num_tracker, plot_style, att_metric, ...
            rank_num, ranking_type, rank_idx, ...
            name_tracker_all, threshold_set, title_name, ...
            x_label, y_label, fig_name, save_fig_path, 'png');
    end
end

fprintf('所有属性帧级别评估完成！结果保存至：%s\n', save_fig_path);
