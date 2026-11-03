%% ZOO++ 自动化属性评估与雷达图集成工具
clc; clear all; close all; warning off;
addpath('./utils/');
addpath('./sequence_evaluation_config/');

%% 1. 基础配置
tmp_mat_path   = './tmp_mat/frame_att/';
path_anno      = './annos/zoo_anno/';      
path_frame_att = './annos/att/';    
rp_all         = './zoo_tracking_results/';
save_fig_path  = './res_fig/frame_att/';
if ~exist(tmp_mat_path, 'dir'), mkdir(tmp_mat_path); end
if ~exist(save_fig_path, 'dir'), mkdir(save_fig_path); end

% 属性定义
att_name = ["Out of View", 'Occlusion', "Fast Motion", "Similar Object", ...
            "Deformation", "Background Mixing", "Rotation", "Small Target"];
att_fig_name = {'OV','OC', 'FM', 'SO', 'DEF', 'BM', 'RO', 'ST'};
num_att = length(att_name);

% 评估参数
evaluation_dataset_type = 'test_set';
norm_dst = true;
eval_type = 'OPE';
ranking_type = 'AUC'; % 雷达图通常使用成功率的AUC作为指标
threshold_set_overlap = 0:0.05:1;

% 加载跟踪器和序列
trackers = config_tracker();
sequences = config_sequence(evaluation_dataset_type);
num_tracker = numel(trackers);
num_seq = numel(sequences);

name_tracker_all = cell(num_tracker, 1);
for i = 1:num_tracker
    name_tracker_all{i} = trackers{i}.name;
end

% 初始化存储（仅针对成功率，用于雷达图绘制）
att_frame_data_overlap = cell(num_tracker, num_att);
for t = 1:num_tracker
    for a = 1:num_att
        att_frame_data_overlap{t,a} = [];
    end
end

%% 2. 核心：按帧遍历并归类属性数据
for seq_idx = 1:num_seq
    s = sequences{seq_idx};
    fprintf('正在处理序列 [%d/%d]: %s\n', seq_idx, num_seq, s);
    
    anno = dlmread([path_anno s '/groundtruth.txt']);
    absent_anno = 1 - dlmread([path_anno s '/absent.txt']); % 1表示存在
    seq_len = size(anno, 1);
    
    frame_att_file = [path_frame_att s '.txt'];
    if ~exist(frame_att_file, 'file'), continue; end
    frame_att = dlmread(frame_att_file, ','); 
    
    % 处理 Out of View 及其恢复帧逻辑
    out_of_view_recovery = zeros(seq_len, 1);
    has_out_of_view = false;
    for f = 1:seq_len
        if frame_att(f, 1) == 1
            has_out_of_view = true;
            out_of_view_recovery(f) = 1;
        elseif has_out_of_view && absent_anno(f) == 0
            out_of_view_recovery(f) = 1;
        end
    end
    frame_att(:, 1) = out_of_view_recovery;

    for tracker_idx = 1:num_tracker
        t_name = trackers{tracker_idx}.name;
        res_file = [rp_all t_name '_tracking_result/' s '.txt'];
        if ~exist(res_file, 'file'), continue; end
        res = dlmread(res_file);
        
        [err_coverage, ~] = calc_seq_err_robust(res, anno, absent_anno, norm_dst);
        valid_frames = find(absent_anno == 0);
        
        for f_idx = 1:seq_len
            if absent_anno(f_idx) == 1, continue; end
            [is_valid, err_idx] = ismember(f_idx, valid_frames);
            if ~is_valid, continue; end
            
            current_coverage = err_coverage(err_idx);
            for a_idx = 1:num_att
                if frame_att(f_idx, a_idx) == 1
                    att_frame_data_overlap{tracker_idx, a_idx} = ...
                        [att_frame_data_overlap{tracker_idx, a_idx}; current_coverage];
                end
            end
        end
    end
end

%% 3. 计算雷达图所需的 AUC 矩阵
fprintf('\n--- 正在计算属性 AUC 矩阵 ---\n');
% radar_data 维度: [跟踪器数量 x 属性数量]
radar_data = zeros(num_tracker, num_att);

for a_idx = 1:num_att
    for t_idx = 1:num_tracker
        frame_values = att_frame_data_overlap{t_idx, a_idx};
        if isempty(frame_values)
            radar_data(t_idx, a_idx) = 0;
            continue;
        end
        
        % 计算该属性下所有阈值的成功率均值
        tmp_success = zeros(length(threshold_set_overlap), 1);
        for th_idx = 1:length(threshold_set_overlap)
            tmp_success(th_idx) = mean(frame_values > threshold_set_overlap(th_idx));
        end
        
        % 计算 AUC (通过梯形积分或均值近似)
        radar_data(t_idx, a_idx) = mean(tmp_success) * 100; % 转为百分制
    end
end

%% 4. 自动绘制雷达图
%% 4. 根据总体 SR 筛选 Top-10 并绘制动态雷达图
fprintf('\n--- 正在根据总体性能筛选 Top-10 并绘制雷达图 ---\n');

% 1. 计算每个跟踪器在所有测试序列上的总体 SR (AUC)
% 注意：这里直接计算 radar_data 的行均值即可代表该跟踪器在所有挑战场景下的平均表现
% 或者如果你有全局评估得到的 overall_SR 向量，也可以替换为该向量
overall_performance = mean(radar_data, 2); 

% 2. 进行降序排列，获取前 10 名的索引
[~, sort_idx] = sort(overall_performance, 'descend');
top_k = min(10, num_tracker);
top_idx = sort_idx(1:top_k);

% 3. 提取前 10 名的属性数据和名称
top_radar_data = radar_data(top_idx, :);
top_names = name_tracker_all(top_idx);

% 4. 动态计算显示范围（针对 Top-10 的表现自动缩放刻度）
% 这样可以清晰地观察到顶尖算法在不同挑战下的优势
min_val = min(top_radar_data(:));
max_val = max(top_radar_data(:));

% 留出 3-5% 的边距，并取 5 的倍数以保持刻度整洁
axis_min = floor(max(0, min_val - 3) / 5) * 5; 
axis_max = ceil(min(100, max_val + 3) / 5) * 5;
radar_lim = repmat([axis_min, axis_max], num_att, 1);

% 5. 创建画布并调用 Draw_radar2
figure('Color', 'w', 'Name', 'Top-10 Attribute Analysis (Sorted by Overall SR)', 'Position', [100, 100, 950, 750]);
Draw_radar2(top_radar_data, radar_lim, att_fig_name, top_names);

% 6. 保存高分辨率结果
saveas(gcf, [save_fig_path 'Top10_Radar_SortedBySR.png']);
fprintf('已成功筛选总体 SR 前 %d 名并生成雷达图。\n', top_k);