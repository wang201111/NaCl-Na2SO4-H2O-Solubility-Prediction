# NaCl-Na₂SO₄-H₂O System Solubility Prediction Model# NaCl-Na₂SO₄-H₂O体系溶解度预测模型NaCl-Na₂SO₄-H₂O体系溶解度预测模型# NaCl-Na₂SO₄-H₂O体系溶解度预测模型

This repository contains the code and data for the paper "A Data-Driven Predictive Model for Solubility: A Case Study of the NaCl-Na₂SO₄-H₂O System".该存储库包含论文“数据驱动的溶解度预测模型：NaCl-Na₂SO₄-H₂O系统的案例研究”的代码和数据。该存储库包含论文“数据驱动的溶解度预测模型：NaCl-Na₂SO₄-H₂O系统的案例研究”的代码和数据。

[![DOI   受伤了](https://zenodo.org/badge/DOI/10.5281/zenodo.15128258.svg)](https://doi.org/10.5281/zenodo.15128258)[[]] (https://zenodo.org/badge/DOI/10.5281/zenodo.15128258.svg DOI) ! (https://doi.org/10.5281/zenodo.15128258)

## Introduction   # #的介绍   介绍

This research proposes an innovative method combining Weighted Local Outlier Factor (WLOF) and Deep Ensemble Neural Network (DENS) for predicting solubility in the NaCl-Na₂SO₄-H₂O ternary system. The method effectively detects and eliminates local outliers while preserving the integrity of data distribution, enhancing prediction model accuracy.本研究提出了一种结合加权局部离群因子（WLOF）和深度集成神经网络（DENS）的创新方法来预测NaCl-Na₂SO₄-H₂O三元体系中的溶解度。该方法在保持数据分布完整性的同时，有效地检测和消除了局部异常点，提高了预测模型的精度。

Our model demonstrates excellent predictive performance over a temperature range of -20°C to 150°C, achieving a coefficient of determination (R²) of 0.989.我们的模型在-20°C至150°C的温度范围内具有出色的预测性能，实现了0.989的决定系数（R²）。

## Data   # #数据

The `data/` directory contains solubility data used to train and validate the model:‘ data/ ’目录包含用于训练和验证模型的溶解度数据：

- `data.xlsx`: Original dataset—‘ data.xlsx ’：原始数据集
- `data-cleaned-IQR.xlsx`: Data cleaned using IQR method- ' Data -cleaned-IQR.xlsx '：使用IQR方法清理数据
- `data-cleaned-WLOF.xlsx`: Data cleaned using WLOF method- ' Data -cleaned-WLOF.xlsx '：使用WLOF方法清理数据
- `data-cleaned-Zscore.xlsx`: Data cleaned using Z-score method- ' Data -cleaned- zscore .xlsx '：使用Z-score方法清理的数据

## Code Structure   代码结构

```
code/   代码/
├── anomaly_detection/             # Anomaly detection algorithms├──anomaly_detection/ #异常检测算法
│   ├── iqr_detection.py           # Interquartile Range detection│├──iqr_detection.py #四分位距离检测
│   ├── wlof_detection.py          # Weighted Local Outlier Factor detection│├──wlof_detection.py #加权局部离群因子检测
│   └── zscore_detection.py        # Z-score detection│├──zscore_detection.py # Z-score detection
├── model/                         # Deep learning models├──model/ #深度学习模型
│   ├── wlof_densbo.py             # WLOF-DENS model with Bayesian optimization│├──wlof_densbo.py #基于贝叶斯优化的wlofdens模型
│   └── prediction.py              # Prediction using trained model│├──predict .py #使用训练模型进行预测
└── utils/                         # Utility functions──utils/ #实用函数
    └── data_processor.py          # Data preprocessing toolsdata_processor.py #数据预处理工具
```

## Installation and Environment Setup安装和环境设置

```bash   ”“bash
# Clone repository   克隆存储库
git clone https://github.com/username/NaCl-Na2SO4-H2O-Solubility-Prediction.git
cd NaCl-Na2SO4-H2O-Solubility-Prediction

# Create virtual environment创建虚拟环境
conda create -n solubility python=3.8Conda create -n溶解度python=3.8
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
