# ZOO++ Tracking Benchmark and Evaluation Toolkit

ZOO++ is a multi-modal visual tracking benchmark for animal tracking. This repository contains the benchmark annotations, tracker result folders, and evaluation scripts for Python and MATLAB.

## Dataset

ZOO++ provides synchronised visible-light (RGB), depth, and infrared (IR) video data for animal tracking. The dataset contains 278 sequences: 228 training sequences and 50 testing sequences. It includes 108,548 temporally aligned frames per modality, bounding-box annotations, and challenge annotations. The 50 test sequences have dense, frame-by-frame annotations; the training annotations are sparse.

The three modalities provide complementary visual cues:

- **RGB**: visible appearance and texture.
- **Depth**: geometric and spatial structure.
- **IR**: thermal patterns that can remain informative under low illumination.

The test set covers a range of species, target scales, and tracking challenges. The dataset paper reports image resolutions of 1920 x 1080 for RGB and depth and 640 x 512 for the native IR sensor images. It also provides affine transforms for mapping IR frames into the RGB/depth coordinate system.

### Download
##### Download Link1: [Baidu](https://pan.baidu.com/s/1mBM-dUZ_pzmZYbCA_WUSYw?pwd=mmmm)

##### Download Link2: [Google: coming soon]


**[Evaluation Toolkit for ZOO++ Benchmark](https://github.com/JJGU2291/ZOO-_Benchmark)**

Download the dataset files provided by the project and extract all parts. The complete extracted dataset is approximately **600 GB**. Ensure that all archive parts have been downloaded before extraction and that the destination drive has sufficient free space.

### Dataset layout

The paper describes each sequence as containing the three modality folders, alignment transforms, and bounding-box annotations:

```text
ZOO++/
|-- 001/
|   |-- colour/                  # RGB frames
|   |-- depth/                   # Depth frames
|   |-- infrared/                # IR frames
|   |-- M.txt                    # IR-to-RGB/depth affine transforms
|   `-- groundtruth.txt          # x, y, width, height, one row per frame
|-- 002/
|   `-- ...
|-- ...
|-- 278/
`-- att/                         # Frame-level challenge attribute labels
    |-- 001.txt
    `-- ...
```

The evaluation scripts in this repository use the benchmark's evaluation annotations in the following layout:

```text
ZOO++_eval_toolkit/
|-- annos/
|   |-- zoo_anno/
|   |   |-- 001/
|   |   |   |-- groundtruth.txt
|   |   |   `-- absent.txt
|   |   `-- ...                  # 002 through 050
|   |-- ZOO_testing_subset.txt
|   |-- att/
|   |   |-- 001.txt
|   |   `-- ...                  # Per-frame challenge attributes
|   |-- classes_train.txt
|   `-- calsses_test.txt         # Species labels; filename retained as supplied
|-- zoo_tracking_results/
|   |-- ViPT_RGBD_tracking_result/
|   |   |-- 001.txt
|   |   `-- ...                  # One result file per test sequence
|   `-- ...
|-- py-eval-toolkit/
|-- sequence_evaluation_config/
|-- utils/
|-- Evaluate_ZOO_benchmark_SP_RR_Only.m
`-- ...
```

The full image data are not required to calculate tracking metrics. Evaluation uses the ground-truth boxes, absent-frame flags, and tracker result files. The challenge and species annotations are used by the additional analysis scripts.

## Tracker result files

Place each tracker’s results under `zoo_tracking_results/`. The folder name must be the tracker name followed by `_tracking_result`, and each sequence file must be named with its three-digit sequence ID:

```text
zoo_tracking_results/
`-- <TRACKER_NAME>_tracking_result/
    |-- 001.txt
    |-- 002.txt
    `-- ...                     # 050.txt
```

Each row in a result file represents one frame and contains a comma-separated bounding box:

```text
x,y,width,height
```

For example:

```text
271.0,506.0,64.0,101.0
272.6,503.2,60.6,107.7
```

The result sequence must correspond to the same sequence ID and frame order as its ground truth. Use the same coordinate convention as the ground-truth files. The first-frame tracker box is replaced with the ground-truth initial box by the evaluation implementation, following the benchmark protocol.

## Evaluation

Both evaluation implementations read the 50-sequence test subset, annotations under `annos/zoo_anno/`, and tracker outputs under `zoo_tracking_results/`. Run commands from the repository root.

### Python

Python 3 and the following packages are required:

```bash
python -m pip install numpy matplotlib
```

Run the default tracker list:

```bash
python py-eval-toolkit/eval_tracker.py
```

To evaluate selected trackers, pass the tracker names without the `_tracking_result` suffix:

```bash
python py-eval-toolkit/eval_tracker.py --trackers ViPT_RGBD ViPT_RGBT OSTrack_official_weight
```

The default tracker list is defined by `TRACKERS` in `py-eval-toolkit/eval_tracker.py`. Add or remove names there to change the default run. Each requested tracker must have a corresponding result directory.

By default, Python writes its outputs to `py-eval-toolkit/output/eval_tracker/`. A different output directory can be set with `--output-dir`:

```bash
python py-eval-toolkit/eval_tracker.py --trackers ViPT_RGBD --output-dir py-eval-toolkit/output/my_evaluation
```

The Python evaluator produces:

- `overall_SR.png`: success plot and success rate (SR, the area under the success curve).
- `overall_PR.png`: precision plot and precision at a 20-pixel centre-error threshold.
- `overall_NPR.png`: normalised precision plot and normalised precision at a 0.20 threshold.
- `summary.csv`: tracker-level SR, PR, NPR, and number of evaluated sequences.

Missing result folders are skipped with a message. If no valid tracker result is found, the script exits with an error.

### MATLAB

Open MATLAB with the repository root as the current folder, then run:

```matlab
Evaluate_ZOO_benchmark_SP_RR_Only
```

The configured tracker names are defined in `utils/config_tracker.m`. They must match the folder names under `zoo_tracking_results/` after appending `_tracking_result`. The test sequence list is `sequence_evaluation_config/ZOO_testing_subset.txt`.

The Python evaluator looks for its optional sequence list at `annos/zoo_anno/ZOO_testing_subset.txt`. If that file is not present, it automatically uses the sequence directories found under `annos/zoo_anno/`; the supplied annotations contain the 50 test sequence directories.

The main script currently sets `ranking_type = 'threshold'` and `norm_dst = false`. With these defaults, it evaluates and saves the pixel-based precision plot (PR) under `res_fig/`; temporary metric arrays are saved under `tmp_mat/`. The overlap/SR plot is skipped by the current threshold-ranking condition in the script. To produce success (SR) plots as well, change `ranking_type` to `'AUC'` in `Evaluate_ZOO_benchmark_SP_RR_Only.m` before running. The `norm_dst` setting controls whether the error metric uses normalised centre distance.

### Metrics

- **SR (Success Rate)**: the area under the success plot, where success is measured by bounding-box IoU over thresholds from 0 to 1.
- **PR (Precision)**: the proportion of frames whose centre-location error is within 20 pixels.
- **NPR (Normalised Precision)**: the proportion of frames whose target-size-normalised centre-location error is within 0.20.

Absent frames are excluded from per-frame error measurements according to `absent.txt`. Evaluation uses the full sequence length when computing each sequence's curve, in line with the implementation used by this toolkit.

## Additional analysis scripts

The `py-eval-toolkit/` directory also contains scripts for attribute-level evaluation and modality-gain analyses by target size, animal category, species, and challenge attribute. Their outputs are written under `py-eval-toolkit/output/` by default. Inspect each script’s configuration and tracker list before running it.

The MATLAB directory also includes attribute and modality-gain analysis scripts, including `Eval_attributes_and_plot_radar_plot.m`, `Eval_modality_gain_with_size.m`, and `Eval_modality_gain_with_category.m`.

## Citation

If you use the dataset or evaluation toolkit in academic work, please cite the accompanying paper:

> Coming soon 

Please also cite the original papers for any tracking methods included in your comparisons.

## Notes

- Keep the test split separate from training and model selection to avoid test-set leakage.
- Ensure that all result files use the same frame order and bounding-box format before comparing trackers.
- The Python and MATLAB scripts are separate implementations. Use the same tracker set and test sequences when comparing their outputs.
