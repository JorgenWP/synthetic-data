# Synthetic Graph Data - Deep Graph Generation

A comprehensive framework for training generative models to synthesize transactional graph data using deep learning techniques.

## 🎯 Overview

This repository provides a complete boilerplate for developing generative AI models that can learn from and synthesize transactional graph data. It includes data preprocessing pipelines, model architectures, training loops, evaluation metrics, and visualization tools.

## 🚀 Features

- **Flexible Data Loading**: Support for various data formats (CSV, JSON, Parquet)
- **Graph Preprocessing**: Convert transactional data into graph representations
- **Model Templates**: Pre-built architectures including Graph VAE
- **Training Pipeline**: Complete training loop with checkpointing and logging
- **Evaluation Metrics**: Compare real and generated graphs
- **Visualization Tools**: Visualize graphs and training metrics
- **Configurable**: YAML-based configuration system

## 📁 Project Structure

```
synthetic-data/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── loader.py            # Data loading utilities
│   │   └── preprocessor.py      # Graph preprocessing
│   ├── models/                   # Model architectures
│   │   ├── base.py              # Base model class
│   │   └── vae.py               # Graph VAE implementation
│   ├── training/                 # Training and evaluation
│   │   ├── trainer.py           # Training loop
│   │   └── evaluator.py         # Evaluation metrics
│   └── utils/                    # Utility functions
│       ├── config.py            # Configuration management
│       └── visualization.py     # Plotting utilities
├── configs/                      # Configuration files
│   └── default_config.yaml      # Default configuration
├── scripts/                      # Executable scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── generate.py              # Generation script
├── notebooks/                    # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── graph_preprocessing.ipynb
│   └── model_training.ipynb
├── data/                         # Data directory
│   ├── raw/                     # Raw data files
│   ├── processed/               # Preprocessed data
│   └── graphs/                  # Generated graphs
├── pyproject.toml               # Project configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- (Optional) CUDA-capable GPU for training

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JorgenWP/synthetic-data.git
cd synthetic-data
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For development (includes testing and linting tools):
```bash
pip install -r requirements-dev.txt
```

4. (Optional) Install graph libraries:
```bash
# For PyTorch Geometric
pip install torch-geometric

# For DGL
pip install dgl
```

## 📊 Quick Start

### 1. Prepare Your Data

Place your transactional data in `data/raw/`. The data should contain at minimum:
- Source node identifiers
- Target node identifiers
- Optional: Edge/node attributes

Example CSV format:
```csv
source,target,amount,timestamp
user_1,user_2,100.50,2024-01-01
user_2,user_3,250.75,2024-01-02
```

### 2. Explore the Data

Use the provided Jupyter notebooks to explore your data:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### 3. Preprocess into Graphs

Convert your transactional data into graph structures:
```bash
jupyter notebook notebooks/graph_preprocessing.ipynb
```

### 4. Configure the Model

Edit `configs/default_config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Data paths and splits
- Evaluation metrics

### 5. Train the Model

Train using the command-line script:
```bash
python scripts/train.py --config configs/default_config.yaml
```

Or use the training notebook:
```bash
jupyter notebook notebooks/model_training.ipynb
```

### 6. Generate Synthetic Graphs

Generate new synthetic graphs:
```bash
python scripts/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --num_samples 100 \
    --output outputs/generated_graphs
```

### 7. Evaluate Results

Evaluate the quality of generated graphs:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/default_config.yaml
```

## 🔧 Customization Guide

### Adding a New Model

1. Create a new model file in `src/models/`
2. Inherit from `BaseGraphGenerator`
3. Implement required methods: `forward()`, `generate()`, `compute_loss()`
4. Register in `src/models/__init__.py`

Example:
```python
from src.models.base import BaseGraphGenerator

class MyCustomModel(BaseGraphGenerator):
    def __init__(self, config):
        super().__init__(config)
        # Your initialization
    
    def forward(self, x):
        # Your forward pass
        pass
    
    def generate(self, num_samples, device=None):
        # Your generation logic
        pass
    
    def compute_loss(self, predictions, targets):
        # Your loss computation
        pass
```

### Adding New Preprocessing Steps

Extend the `GraphPreprocessor` class in `src/data/preprocessor.py`:
```python
def custom_preprocessing(self, df):
    # Your custom preprocessing logic
    pass
```

### Custom Evaluation Metrics

Add new metrics to `Evaluator` in `src/training/evaluator.py`:
```python
def custom_metric(self, real_graphs, generated_graphs):
    # Your metric computation
    return metric_value
```

## 📈 Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir runs/
```

---

**Note**: This is a boilerplate template. You'll need to implement data-specific preprocessing and adapt the models to your specific use case of transactional graph synthesis.