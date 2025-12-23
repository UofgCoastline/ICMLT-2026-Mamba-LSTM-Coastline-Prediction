# ICMLT-2026-Mamba-LSTM-Coastline-Prediction
This is a file for opening the coding for ICMLT 2026
# Coastline Prediction: A Multi-Method Comparison

This repository implements and compares five methods for shoreline time series forecasting, supporting the EU Horizon TERRA project's Digital Twin for Monitoring Coastal Change.

## Case Study: St Andrews Coastline

This work places a particular emphasis on the St Andrews coastal area in Scotland, which serves as a representative and well-instrumented case study within the TERRA initiative. The region exhibits diverse shoreline dynamics influenced by tidal processes, seasonal variability, and long-term morphological changes, making it a suitable benchmark for evaluating data-driven shoreline prediction methods.

Within the StAndrews Digital Twin, both learning-based approaches and traditional transects-based prediction methods are considered. This allows a systematic comparison between classical coastal analysis techniques and modern sequence modeling approaches, including LSTM, diffusion-based baselines, and input-driven state space models. The St Andrews case study therefore provides a concrete and interpretable setting for assessing prediction accuracy, robustness, and uncertainty behavior across different modeling paradigms.


## Research Significance and Context

This study builds upon our previous work on automated coastal feature detection from satellite imagery （https://github.com/UofgCoastline/ICASSP-2026-Robust-Unet）. Those segmentation models provide accurate and spatially consistent coastline representations, which form the essential geometric and temporal inputs for downstream time-series modeling and prediction.

By leveraging these high-quality, segmentation-derived coastline trajectories, the present work shifts the focus from static boundary extraction to dynamic shoreline evolution modeling. In particular, we investigate whether modern sequence modeling approaches, including LSTM, diffusion-based generative baselines, and input-driven state space models, can more effectively capture long-term temporal dependencies and variability in coastline change.

Within a coastal digital twin context, this progression from segmentation to prediction is crucial. Reliable segmentation establishes trustworthy state observations, while robust temporal modeling enables forecasting, scenario comparison, and uncertainty-aware decision support. This study therefore serves as a natural methodological extension, connecting pixel-level coastal perception with system-level digital twin intelligence.


## Overview

We predict future coastline positions from historical satellite-derived shoreline sequences using:

| Method | Architecture | Type |
|--------|--------------|------|
| **LSTM** | 2-layer LSTM encoder-decoder | Direct regression |
| **Diff-Ridge** | Linear diffusion + Ridge regression | Closed-form diffusion |
| **Diff-Linear** | Linear diffusion + SGD | Neural diffusion |
| **Mamba+LSTM** | Mamba SSM encoder + LSTM aggregation | Direct regression |
| **Mamba-Diffusion** | Mamba encoder + Cross-attention denoiser | Conditional diffusion |

## Requirements

```bash
pip install torch numpy matplotlib scipy scikit-learn
```

- Python >= 3.8
- PyTorch >= 1.7
- NumPy, Matplotlib, SciPy, scikit-learn

## Data Format

The code expects LabelMe polygon annotations in JSON format from Sentinel-2 Imagery:

```
./labelme_images/annotations/
├── 2017_05.json
├── 2017_06.json
├── ...
└── 2025_05.json
```

Each JSON file should contain polygon vertices defining the coastline boundary:

```json
{
  "shapes": [
    {
      "label": "coastline",
      "points": [[x1, y1], [x2, y2], ...]
    }
  ],
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

## Usage

### Basic Usage

```bash
python main.py
```

### Configuration

Key parameters can be modified at the top of `main()`:

```python
history_length = 5      # Number of past timesteps (L)
num_points = 256        # Resampled points per coastline (P)
seed = 42               # Random seed for reproducibility
output_dir = "./compare_results_5_methods"
```

### Model-specific Parameters

```python
# Mamba+LSTM
mamba_blocks = 2        # Number of Mamba blocks (K)
mamba_dim = 128         # Model dimension (d)
lstm_hidden = 256       # LSTM hidden size

# Mamba-Diffusion
diffusion_steps = 100   # Diffusion timesteps (T)
denoiser_blocks = 6     # Denoiser depth
cfg_scale = 2.0         # Classifier-free guidance scale
```

## Output

Results are saved to `./compare_results_5_methods/`:

```
compare_results_5_methods/
├── lstm_model.pth                    # Trained LSTM weights
├── mamba_lstm_model.pth              # Trained Mamba+LSTM weights
├── mamba_diffusion_model.pth         # Trained Mamba-Diffusion weights
├── metrics_report_5_methods.txt      # Quantitative results
├── coastline_evolution.png           # Temporal evolution visualization
├── training_curves.png               # Loss curves
├── error_distribution.png            # MSE histograms
├── temporal_mse_curve.png            # MSE over time
├── multi_year_comparison.png         # Predictions at 3 timestamps
├── prediction_compare_5_methods.png  # Single-sample comparison
├── metrics_radar.png                 # Radar plot of metrics
├── metrics_bar_chart.png             # Bar chart comparison
├── pointwise_error_heatmap.png       # Spatial error distribution
└── summary_dashboard.png             # Comprehensive 3×3 dashboard
```
## Involved Algorithm
### LSTM Baseline
Standard encoder-decoder architecture with 2-layer LSTM for sequential modelling.

### Diff-Ridge & Diff-Linear
Linearised conditional diffusion models:
- **Diff-Ridge**: Closed-form Ridge regression solution (fast, limited expressiveness)
- **Diff-Linear**: SGD-trained linear model (more flexible)

### Mamba+LSTM (Proposed)
Hybrid architecture combining:
- **Mamba encoder**: Selective state space model for efficient long-range dependency modelling
- **LSTM aggregation**: Sequential feature refinement
- **Direct regression**: Deterministic prediction

### Mamba-Diffusion (Proposed)
Conditional diffusion with:
- **Mamba condition encoder**: SSM-based history encoding
- **Cross-attention denoiser**: Conditioning via attention mechanism
- **Classifier-Free Guidance (CFG)**: Enhanced conditioning at inference
- **DDIM sampling**: Fast deterministic sampling option

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| Hausdorff | Maximum distance between point sets |
| MaxError | Maximum point-wise Euclidean distance |

## Results

Representative results on our dataset (N=97 timestamps, May 2017 – May 2025):

| Method            | MSE    | RMSE   | MAE    | Hausdorff | MaxError |
|-------------------|--------|--------|--------|-----------|----------|
| LSTM              | 0.0016 | 0.0235 | 0.0159 | 0.0373    | 0.0894   |
| Diff-Ridge        | 0.0017 | 0.0249 | 0.0160 | 0.0502    | 0.0977   |
| Diff-Linear       | 0.0017 | 0.0260 | 0.0166 | 0.0492    | 0.1138   |
| **Mamba+LSTM**    | 0.0016 | **0.0228** | **0.0154** | **0.0357** | **0.0874** |
| Mamba-Diffusion   | 0.0023 | 0.0384 | 0.0237 | 0.0821    | 0.2062   |


## Citation

If you use this code, please cite:

```bibtex
@article{Tian2025MambaShoreline,
  title   = {Mamba–LSTM for Long-Horizon Shoreline Prediction with Uncertainty Quantification},
  author  = {Tian, Zhen and Gao, Zhiwei and Anagnostopoulos, Christos, Wang, Qiyuan and Kolomvatsos, Kostas},
  journal = {Under review},
  year    = {2025}
}

or

@misc{Tian2025MambaShoreline,
  title   = {Mamba–LSTM for Long-Horizon Shoreline Prediction with Uncertainty Quantification},
  author  = {Tian, Zhen and Gao, Zhiwei and Anagnostopoulos, Christos, Wang, Qiyuan and Kolomvatsos, Kostas},
  year   = {2025},
  note   = {Code and experimental results available at GitHub}
}
```

## Acknowledgement

This work has been supported by the European Union’s Horizon Europe research and innovation programme under the TERRA project, which focuses on federated, explainable, and privacy-preserving digital twins for coastal monitoring and environmental risk assessment.

The research contributes to the development of coastal digital twin components within the TERRA initiative, including data-driven shoreline modelling, uncertainty-aware prediction, and comparative evaluation against traditional transects-based methods.


## License

MIT License
