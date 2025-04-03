# NaCl-Na₂SO₄-H₂O System Solubility Prediction Model

This repository contains the code and data for the paper "A Data-Driven Predictive Model for Solubility: A Case Study of the NaCl-Na₂SO₄-H₂O System".

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)

## Introduction

This research proposes an innovative method combining Weighted Local Outlier Factor (WLOF) and Deep Ensemble Neural Network (DENS) for predicting solubility in the NaCl-Na₂SO₄-H₂O ternary system. The method effectively detects and eliminates local outliers while preserving the integrity of data distribution, enhancing prediction model accuracy.

Our model demonstrates excellent predictive performance over a temperature range of -20°C to 150°C, achieving a coefficient of determination (R²) of 0.989.

## Data

The `data/` directory contains solubility data used to train and validate the model:

- `data.xlsx`: Original dataset
- `data-cleaned-IQR.xlsx`: Data cleaned using IQR method
- `data-cleaned-WLOF.xlsx`: Data cleaned using WLOF method
- `data-cleaned-Zscore.xlsx`: Data cleaned using Z-score method

## Code Structure

```
code/
├── anomaly_detection/             # Anomaly detection algorithms
│   ├── iqr_detection.py           # Interquartile Range detection
│   ├── wlof_detection.py          # Weighted Local Outlier Factor detection
│   └── zscore_detection.py        # Z-score detection
├── model/                         # Deep learning models
│   ├── wlof_densbo.py             # WLOF-DENS model with Bayesian optimization
│   └── prediction.py              # Prediction using trained model
└── utils/                         # Utility functions
    └── data_processor.py          # Data preprocessing tools
```

## Installation and Environment Setup

```bash
# Clone repository
git clone https://github.com/username/NaCl-Na2SO4-H2O-Solubility-Prediction.git
cd NaCl-Na2SO4-H2O-Solubility-Prediction

# Create virtual environment
conda create -n solubility python=3.8
conda activate solubility

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Anomaly Detection

Use the following scripts for anomaly detection:

```bash
# IQR method
python code/anomaly_detection/iqr_detection.py

# Weighted Local Outlier Factor method
python code/anomaly_detection/wlof_detection.py

# Z-score method
python code/anomaly_detection/zscore_detection.py
```

### Model Training and Optimization

Train the WLOF-DENS model with Bayesian optimization:

```bash
python code/model/wlof_densbo.py
```

### Solubility Prediction

Make predictions using the trained model:

```bash
python code/model/prediction.py
```

## Reproducing Results

To reproduce the results from the paper, follow these steps:

1. Process data using scripts in the `anomaly_detection` directory
2. Run `wlof_densbo.py` for model training and optimization
3. Use `prediction.py` to generate prediction results and figures

## Contributing

We welcome contributions to this project. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{wang2025data,
  title={A Data-Driven Predictive Model for Solubility: A Case Study of the {NaCl-Na$_2$SO$_4$-H$_2$O} System},
  author={Wang, Yuan and Chen, Mengyue and Tian, Jingwei and Zhang, Weidong and Liu, Dahuan},
  journal={[Journal Name]},
  volume={[Volume]},
  number={[Issue]},
  pages={[Pages]},
  year={2025},
  publisher={[Publisher]}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions about this project, please contact:
- Dahuan Liu: liudh@mail.buct.edu.cn
- Weidong Zhang: weidzhang1208@126.com
