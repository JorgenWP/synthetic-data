# Quick Start Guide

This guide will help you get started with the Synthetic Graph Data Generation framework in 5 minutes.

## Prerequisites

- Python 3.8+
- pip

## Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/JorgenWP/synthetic-data.git
cd synthetic-data

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

Run the test suite to ensure everything is working:

```bash
pip install pytest
pytest tests/ -v
```

You should see all tests passing.

## Create Your First Graph Model

### Step 1: Prepare Sample Data

Create a simple CSV file with transactional data in `data/raw/sample.csv`:

```csv
source,target,amount
user_1,user_2,100.50
user_2,user_3,250.75
user_3,user_4,175.20
user_1,user_3,320.00
user_4,user_2,90.50
```

### Step 2: Load and Explore Data

Create a Python script or use a notebook:

```python
import sys
sys.path.insert(0, '.')

from src.data import DataLoader, GraphPreprocessor

# Load data
loader = DataLoader(data_dir='data/raw')
df = loader.load_transactions('sample.csv')
print(f"Loaded {len(df)} transactions")

# Convert to graph
preprocessor = GraphPreprocessor()
G = preprocessor.build_graph_from_transactions(
    df, 
    source_col='source', 
    target_col='target',
    edge_attrs=['amount']
)

# Check graph statistics
stats = preprocessor.compute_graph_statistics(G)
print("Graph statistics:", stats)
```

### Step 3: Train a Simple Model

```python
import torch
from src.models import GraphVAE
from src.utils.config import load_config

# Load configuration
config = load_config('configs/default_config.yaml')

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphVAE(config['model']).to(device)

print(f"Model created with {model.count_parameters():,} parameters")

# Test generation
samples = model.generate(num_samples=10, device=device)
print(f"Generated samples shape: {samples.shape}")
```

### Step 4: Use the Training Script (Once You Have Real Data)

```bash
# Edit configs/default_config.yaml first to match your data

# Train the model
python scripts/train.py --config configs/default_config.yaml

# Generate synthetic graphs
python scripts/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --num_samples 100 \
    --output outputs/generated_graphs

# Evaluate
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/default_config.yaml
```

## Next Steps

1. **Explore Notebooks**: Check out the Jupyter notebooks in `notebooks/` for detailed examples
2. **Customize Models**: Add your own model architectures in `src/models/`
3. **Add Preprocessing**: Extend data preprocessing in `src/data/preprocessor.py`
4. **Tune Hyperparameters**: Edit `configs/default_config.yaml`

## Common Issues

### Import Errors
If you get import errors, ensure the package is in your Python path:
```python
import sys
sys.path.insert(0, '/path/to/synthetic-data')
```

### CUDA Out of Memory
If training fails with GPU memory errors:
1. Reduce `batch_size` in config
2. Use a smaller model (reduce `hidden_dim`, `latent_dim`)
3. Use CPU instead: set `device: "cpu"` in config

### Missing Dependencies
Some graph libraries are optional. Install as needed:
```bash
pip install torch-geometric  # For GNN models
pip install dgl              # Alternative GNN library
```

## Getting Help

- Check the [main README](README.md) for detailed documentation
- Review example notebooks in `notebooks/`

## What's Next?

Now you're ready to:
- Adapt the preprocessing to your specific transactional data format
- Implement custom model architectures
- Train on your real data
- Generate and evaluate synthetic graphs

Happy graph generation! 🎉
