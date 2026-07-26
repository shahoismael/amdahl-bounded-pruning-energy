# Green Illusion: Pruning Audit

Hardware-measured energy audit testing whether pruning's theoretical filter-count reduction produces real, measured energy savings on physical CPU hardware. This repository contains the MATLAB pipeline, code, and results supporting the paper:

**"The Green Illusion: An Engineering Audit of Hardware-Measured Energy Costs in Pruned Neural Networks"**
Shaho Ismael Hassen, Ahmed Abdulfatah Abdlrazaq — Salahaddin University-Erbil
Submitted to ICIEAHS 2026 (1st International Conference on Informatics, Engineering and Applied Health Sciences)

*This repository is currently private and will be made public upon publication.*

## Overview

Pruning is widely assumed to reduce a neural network's environmental footprint, on the basis that a reduction in filter count or FLOPs implies a reduction in energy consumption. This project tests that assumption directly by:

1. Training baseline convolutional classifiers on four benchmark datasets
2. Applying a fixed, conservative Taylor-score pruning configuration (2 iterations, 4 filters/iteration)
3. Measuring real inference energy on physical hardware (CPU package power, via HWiNFO) rather than estimating it in software
4. Repeating each baseline-vs-pruned comparison 5 times per dataset (20 total runs) to capture natural hardware variability
5. Comparing the fixed theoretical filter-count reduction against the measured energy outcome

## Datasets

- **MNIST** (LeCun et al., 1998)
- **Fashion-MNIST** (Xiao et al., 2017)
- **MIT-BIH Arrhythmia Database** (Moody & Mark, 2001) — via PhysioNet
- **NIH ChestX-ray14** (Wang et al., 2017) — 2,000-image subset

Datasets are not included in this repository due to size; see each dataset's original source (linked in the paper) to download.

## Repository structure

```
code/
├── load_idx_dataset.m           # MNIST / Fashion-MNIST loader (IDX format)
├── load_mitbih_record.m         # MIT-BIH .dat/.hea loader (format 212)
├── read_atr_annotations.m       # MIT-BIH .atr annotation parser
├── train_baseline.m             # Baseline CNN training (2D image datasets)
├── train_baseline_1d.m          # Baseline CNN training (1D signal data)
├── prune_model.m                # Taylor-score channel pruning
├── quantize_model.m             # Post-training INT8 quantization
├── measure_energy.m             # Hardware-timed sustained inference measurement
├── get_next_run_number.m        # Auto-numbering helper for repeated runs
├── main_pipeline.m              # MNIST pipeline
├── main_pipeline_fashion.m      # Fashion-MNIST pipeline
├── main_pipeline_mitbih.m       # MIT-BIH pipeline
├── main_pipeline_chestxray.m    # ChestX-ray14 pipeline
├── check_pruned_params.m        # Verifies exact parameter/filter counts pre/post pruning
├── check_chestxray_balance.m    # Verifies ChestX-ray14 subset class balance
├── check_mitbih_balance.m       # Verifies MIT-BIH Normal/Abnormal class balance
├── make_figure1_pipeline.m      # Generates Figure 1 (pipeline diagram)
├── make_figure2_bars.m          # Generates Figure 2 (theoretical vs. measured bar chart)
└── make_figure3_boxplot.m       # Generates Figure 3 (variability box plot)
```

## Requirements

- MATLAB R2025b (or later)
- Deep Learning Toolbox
- Deep Learning Toolbox Model Compression Library
- [HWiNFO](https://www.hwinfo.com/) (for hardware power logging, run alongside MATLAB)

## Usage

1. Place datasets in `data/<dataset_name>/` following the structure referenced in each pipeline script.
2. Run HWiNFO in sensors-only mode with CSV logging enabled.
3. Run the desired pipeline script, e.g.:
   ```matlab
   main_pipeline
   ```
4. Cross-reference the printed start/end timestamps against the HWiNFO CSV log to extract average power draw for energy calculations.

## Key finding

Across 20 measurement runs spanning 4 datasets, a fixed 20.5% theoretical filter-count reduction (22.84% real parameter reduction) did **not** produce a reliable, predictable reduction in measured energy per inference. The mean effect across all runs was statistically indistinguishable from zero (t(19) = -0.307, p = 0.76), while run-to-run variability (SD ≈ 15.9 percentage points) far exceeded the average effect itself.

## License

MIT License — see `LICENSE` file.

## Citation

If you use this code, please cite the associated paper (details to be updated upon publication).

## Contact

Shaho Ismael Hassen — shaho.hassen@su.edu.krd
