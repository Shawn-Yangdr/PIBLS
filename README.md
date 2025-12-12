# PIBLS: Physics-Informed Broad Learning System

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.4.1](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)](https://pytorch.org/)

## 📖 Introduction

A meshless PDE solver that combines **Broad Learning Systems** with **physics-informed constraints** for solving partial differential equations. Unlike deep learning approaches like PINNs, PIBLS uses random feature generation with **closed-form least squares optimization**, achieving machine-precision accuracy in seconds.

## 🛠️ Getting Started

### Installation

```bash
git clone https://github.com/Shawn-Yangdr/PIBLS.git && cd PIBLS

conda env create -f environment.yml
conda activate PIBLS
```

### Quick Start

```bash
python main.py --problem TC1          # Run a benchmark problem
python main.py --list-problems        # List all available problems
```

## 🔬 Reproduce Paper Results

Pre-trained weights are provided in `assets/save_models/`:

```bash
# TC1
python main.py --config configs/experiments/TC-1.yaml --model_path assets/save_models/TC-1.pt

# TC11
python main.py --config configs/experiments/TC-11.yaml --model_path assets/save_models/TC-11.pt

# Fisher-KPP: Cell migration
python main.py --config configs/experiments/Fisher.yaml --model_path assets/save_models/Fisher-KPP_Problem(1000).pt
```

## ⚙️ Configuration

PIBLS uses a three-level configuration system: **CLI > YAML > defaults**.

```bash
python main.py --config configs/experiments/TC-1.yaml                # Use YAML config
python main.py --config configs/experiments/TC-1.yaml --n_colloc 500 # Override parameters
```

## 💘 Acknowledgements

- [Broad Learning System](https://ieeexplore.ieee.org/document/7987745) - Chen & Liu, 2017
- [Physics-Informed Neural Networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125) - Raissi et al., 2019

## 📝 Citation

```bibtex

```

## 📧 Contact

If you have any question about this project, please contact XX and XX.
