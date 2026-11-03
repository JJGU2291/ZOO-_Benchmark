%% Modality gain analysis w.r.t. object size on ZOO++
% This script replaces the original SR/PR/NPR benchmark evaluation with a
% size-aware modality gain analysis.
%
% Workflow:
%   1) compute the mean target area ratio for each test sequence from GT
%   2) sort 50 test sequences by size and split them into 5 equal bins
%   3) compute sequence-level SR (success AUC) for each tracker
%   4) average SR inside each bin and draw the final line plot

clc;
clearvars;
close all;
warning off;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

cwd0 = pwd;
cd(script_dir);
cleanup_obj = onCleanup(@() cd(cwd0));

addpath(fullfile(script_dir, 'utils'));

path_anno     = fullfile(script_dir, 'annos', 'zoo_anno');
rp_all        = fullfile(script_dir, 'zoo_tracking_results');
save_fig_path = fullfile(script_dir, 'res_fig');
save_mat_path = fullfile(script_dir, 'tmp_mat');

num_bins = 5;
img_width  = 1920;
img_height = 1080;
threshold_set_overlap = 0:0.05:1;

% Use the selected trackers as modality representatives.
trackers = {
    struct('name', 'OSTrack_official_weight', 'label', 'RGB')
    struct('name', 'ViPT_RGBT',               'label', 'RGBT')
    struct('name', 'ViPT_RGBD',               'label', 'RGBD')
};

tracker_names = cell(numel(trackers), 1);
tracker_labels = cell(numel(trackers), 1);
for i = 1:numel(trackers)
    tracker_names{i} = trackers{i}.name;
    tracker_labels{i} = trackers{i}.label;
end

% Read test sequence list from the benchmark config.
sequences = config_sequence('test_set');
num_seq = numel(sequences);

if mod(num_seq, num_bins) ~= 0
    error('The number of test sequences (%d) must be divisible by %d.', num_seq, num_bins);
end

bin_size = num_seq / num_bins;

% -------------------------------------------------------------------------
% Step 1: compute mean target area ratio for each sequence
% -------------------------------------------------------------------------
seq_mean_area_ratio = zeros(num_seq, 1);
for i = 1:num_seq
    seq_name = sequences{i};
    gt_path = fullfile(path_anno, seq_name, 'groundtruth.txt');
    gt = readmatrix(gt_path);
    if size(gt, 2) > 4
        gt = gt(:, 1:4);
    end

    area_ratio = (gt(:, 3) .* gt(:, 4)) / (img_width * img_height);
    seq_mean_area_ratio(i) = mean(area_ratio);
end

[seq_mean_area_ratio_sorted, seq_sort_idx] = sort(seq_mean_area_ratio, 'ascend');
seq_sorted_names = sequences(seq_sort_idx);

bin_idx = reshape(seq_sort_idx, bin_size, num_bins)';

fprintf('\nSorted sequence mean area ratios (ascending):\n');
for i = 1:num_seq
    fprintf('%2d. %-6s  mean area ratio = %.6f (%.4f%%)\n', ...
        i, seq_sorted_names{i}, seq_mean_area_ratio_sorted(i), 100 * seq_mean_area_ratio_sorted(i));
end

% -------------------------------------------------------------------------
% Step 2: compute sequence-level SR (AUC) for each tracker
% -------------------------------------------------------------------------
seq_sr = nan(numel(trackers), num_seq);
for k = 1:numel(trackers)
    tracker_name = trackers{k}.name;
    fprintf('Computing SR for %s...\n', tracker_name);

    for i = 1:num_seq
        seq_name = sequences{i};
        seq_sr(k, i) = compute_sequence_sr( ...
            seq_name, tracker_name, path_anno, rp_all, threshold_set_overlap);
    end
end

% -------------------------------------------------------------------------
% Step 3: average SR inside each size bin
% -------------------------------------------------------------------------
bin_sr = nan(numel(trackers), num_bins);
bin_sr_std = nan(numel(trackers), num_bins);
bin_area_mean = nan(1, num_bins);

for b = 1:num_bins
    idx = bin_idx(b, :);
    bin_area_mean(b) = mean(seq_mean_area_ratio(idx));

    for k = 1:numel(trackers)
        values = seq_sr(k, idx);
        values = values(~isnan(values));
        if isempty(values)
            bin_sr(k, b) = NaN;
            bin_sr_std(k, b) = NaN;
        else
            bin_sr(k, b) = mean(values);
            if numel(values) > 1
                bin_sr_std(k, b) = std(values);
            else
                bin_sr_std(k, b) = 0;
            end
        end
    end
end

% -------------------------------------------------------------------------
% Step 4: draw and save the figure
% -------------------------------------------------------------------------
if ~exist(save_fig_path, 'dir')
    mkdir(save_fig_path);
end
if ~exist(save_mat_path, 'dir')
    mkdir(save_mat_path);
end

fig_name = 'modality_gain_with_size_SR';

figure_handle = figure;
set(figure_handle, 'Units', 'normalized', 'Position', [0.18, 0.2, 0.56, 0.52]);
axes_handle = axes('Parent', figure_handle);
hold(axes_handle, 'on');
grid(axes_handle, 'on');
set(axes_handle, 'GridLineStyle', ':', 'GridColor', [0, 0, 0], 'GridAlpha', 0.2, 'LineWidth', 1.0);

line_styles = {
    struct('color', [0.10, 0.10, 0.10], 'marker', 'o', 'lineStyle', '-')
    struct('color', [0.85, 0.33, 0.10], 'marker', 's', 'lineStyle', '-')
    struct('color', [0.00, 0.45, 0.74], 'marker', '^', 'lineStyle', '-')
};

legend_text = cell(numel(trackers), 1);
for k = 1:numel(trackers)
    style = line_styles{k};
    plot(1:num_bins, bin_sr(k, :), ...
        'Color', style.color, ...
        'LineStyle', style.lineStyle, ...
        'Marker', style.marker, ...
        'LineWidth', 2.5, ...
        'MarkerSize', 7);
    legend_text{k} = tracker_labels{k};
end

bin_labels = cell(1, num_bins);
for b = 1:num_bins
    bin_labels{b} = sprintf('Bin %d\n%.3f%%', b, 100 * bin_area_mean(b));
end

set(axes_handle, 'XTick', 1:num_bins, 'XTickLabel', bin_labels);
xlim([0.8, num_bins + 0.2]);
ylim([0, 1]);
xlabel('Target size bins (sorted by mean area ratio)');
ylabel('Sequence-level SR (AUC)');
title('Modality gain vs. target size on ZOO++ testing set');
legend(legend_text, 'Location', 'northwest', 'Interpreter', 'none');

saveas(figure_handle, fullfile(save_fig_path, [fig_name, '.png']));

analysis_result = struct();
analysis_result.trackers = tracker_names;
analysis_result.tracker_labels = tracker_labels;
analysis_result.sequences = sequences;
analysis_result.sorted_sequences = seq_sorted_names;
analysis_result.seq_mean_area_ratio = seq_mean_area_ratio;
analysis_result.seq_mean_area_ratio_sorted = seq_mean_area_ratio_sorted;
analysis_result.seq_sort_idx = seq_sort_idx;
analysis_result.seq_sorted_names = seq_sorted_names;
analysis_result.bin_idx = bin_idx;
analysis_result.seq_sr = seq_sr;
analysis_result.bin_sr = bin_sr;
analysis_result.bin_sr_std = bin_sr_std;
analysis_result.bin_area_mean = bin_area_mean;
analysis_result.img_width = img_width;
analysis_result.img_height = img_height;
analysis_result.threshold_set_overlap = threshold_set_overlap;

save(fullfile(save_mat_path, 'modality_gain_with_size_SR.mat'), 'analysis_result');

% Console summary for quick inspection.
fprintf('\nSize-bin summary (mean area ratio and SR):\n');
for b = 1:num_bins
    fprintf('Bin %d | mean area = %.4f%% | sequences = %s\n', ...
        b, 100 * bin_area_mean(b), strjoin(seq_sorted_names((b - 1) * bin_size + 1:b * bin_size), ', '));
    for k = 1:numel(trackers)
        fprintf('  %-18s SR = %.4f (std = %.4f)\n', tracker_labels{k}, bin_sr(k, b), bin_sr_std(k, b));
    end
end

% -------------------------------------------------------------------------
% Local function: compute sequence-level SR (AUC of success plot)
% -------------------------------------------------------------------------
function seq_sr = compute_sequence_sr(seq_name, tracker_name, path_anno, rp_all, threshold_set_overlap)
    gt_path = fullfile(path_anno, seq_name, 'groundtruth.txt');
    absent_path = fullfile(path_anno, seq_name, 'absent.txt');
    res_path = fullfile(rp_all, [tracker_name, '_tracking_result'], [seq_name, '.txt']);

    if ~exist(gt_path, 'file') || ~exist(res_path, 'file')
        seq_sr = NaN;
        return;
    end

    anno = readmatrix(gt_path);
    if size(anno, 2) > 4
        anno = anno(:, 1:4);
    end

    if exist(absent_path, 'file')
        absent_anno = readmatrix(absent_path);
        absent_anno = 1 - absent_anno;
    else
        absent_anno = zeros(size(anno, 1), 1);
    end

    res = readmatrix(res_path);
    if isempty(res)
        seq_sr = NaN;
        return;
    end

    [err_coverage, ~] = calc_seq_err_robust(res, anno, absent_anno, false);

    success_curve = zeros(1, numel(threshold_set_overlap));
    for t_idx = 1:numel(threshold_set_overlap)
        success_curve(t_idx) = sum(err_coverage > threshold_set_overlap(t_idx)) / (numel(err_coverage) + eps);
    end

    seq_sr = mean(success_curve);
end
