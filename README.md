# Depth Estimation Comparison: Monocular vs Stereo Methods

This repository implements and compares monocular and stereo depth estimation methods on the CARLA simulator dataset, providing comprehensive quantitative and qualitative analysis.

## 🎯 Project Overview

This project benchmarks different depth estimation approaches:
- **Monocular Depth Estimation**: MiDaS models (DPT_Large, DPT_Hybrid, MiDaS_Small)
- **Stereo Depth Estimation**: Semi-Global Block Matching (SGBM) variants

## 📁 Repository Structure

```
depth-estimation-comparison/
├── config.py                 # Model configurations and settings
├── inference.py              # Main inference script
├── visu.py                   # Quick visualization example
├── src/
│   ├── models/
│   │   ├── monocular/
│   │   │   └── midas.py      # MiDaS model implementations
│   │   └── stereo/
│   │       └── sgbm.py       # SGBM model implementations
│   └── utils/
│       ├── data_loader.py    # CARLA dataset loader
│       ├── evaluation.py     # Metrics calculation
│       └── visualization.py  # Visualization utilities
├── results/                  # Generated results (created during inference)
├── report.md                # Detailed analysis report
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Usage

1. **List available models:**
```bash
python inference.py --list_models
```

2. **Run inference on all models:**
```bash
python inference.py --data_path data/images --output_dir results
```

3. **Run specific model types:**
```bash
# Monocular models only
python inference.py --data_path data/images --mono --output_dir results

# Stereo models only
python inference.py --data_path data/images --stereo --output_dir results

# Specific models
python inference.py --data_path data/images --models midas_large sgbm_balanced
```

4. **Quick visualization test:**
```bash
python visu.py
```

## 📊 Available Models

### Monocular Models
- **midas_large**: MiDaS DPT Large - highest accuracy, slowest inference
- **midas_hybrid**: MiDaS DPT Hybrid - balanced accuracy and speed
- **midas_small**: MiDaS Small - fastest but less accurate

### Stereo Models
- **sgbm_fast**: SGBM Fast preset - faster but less accurate
- **sgbm_balanced**: SGBM Balanced preset - good speed/accuracy trade-off
- **sgbm_accurate**: SGBM Accurate preset - high accuracy but slower

## 📈 Evaluation Metrics

### Depth Metrics
- **abs_rel**: Absolute relative error
- **rmse**: Root mean square error
- **rmse_log**: Root mean square log error
- **a1, a2, a3**: Threshold accuracies (δ < 1.25, 1.25², 1.25³)

### Disparity Metrics
- **bad_1.0, bad_2.0, bad_3.0**: Bad pixel ratios (error > threshold)
- **mae**: Mean absolute error
- **rmse**: Root mean square error

## 🗂️ Dataset Format

The CARLA dataset should be organized as:
```
data/
├── images/
│   ├── left/          # Left stereo images
│   └── right/         # Right stereo images
├── depth/             # Ground truth depth maps
├── disparity/         # Ground truth disparity maps
└── camera_config.json # Camera parameters (optional)
```

## 📋 Command Line Arguments

- `--data_path`: Path to dataset directory (required)
- `--output_dir`: Output directory for results (default: results)
- `--num_samples`: Number of samples to evaluate (default: all)
- `--models`: Specific models to evaluate
- `--mono`: Evaluate all monocular models
- `--stereo`: Evaluate all stereo models
- `--device`: Device to use (cuda/cpu, default: auto-detect)
- `--list_models`: List available models and exit

## 📊 Output Files

After running inference, the following files are generated:

- `results/metrics.json`: Summary metrics for all models
- `results/detailed_metrics.json`: Per-sample detailed results
- `results/visualizations/`: Comparison visualizations for each sample
- `results/{model_name}/`: Individual depth maps for each model (if enabled)

## 🔧 Configuration

Edit `config.py` to:
- Add new models
- Modify evaluation settings
- Change visualization parameters
- Adjust default model selections

## 📝 Results Analysis

See `report.md` for detailed analysis including:
- Quantitative performance comparison
- Qualitative analysis with example cases
- Strengths and weaknesses of each approach
- Failure case analysis
- Recommendations for different use cases

## 🛠️ Dependencies

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- tqdm

See `requirements.txt` for complete list with versions.

## 📚 References

- **MiDaS**: [Towards Robust Monocular Depth Estimation](https://arxiv.org/abs/1907.01341)
- **SGBM**: Semi-Global Matching algorithm from OpenCV
- **CARLA**: [Open-source simulator for autonomous driving research](https://carla.org/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions or issues, please:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed description

---

**Note**: This implementation is designed for academic research and comparison purposes. For production use, consider additional optimizations and error handling.
