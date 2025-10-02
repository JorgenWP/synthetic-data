# Data Directory

This directory contains data used for training and evaluating graph generation models.

## Directory Structure

- `raw/`: Raw transactional data files
- `processed/`: Preprocessed data ready for model training
- `graphs/`: Generated graph structures

## Data Format

Place your raw transactional data files in the `raw/` directory. The data should be in a format that can be converted to graph structures (e.g., CSV, JSON, Parquet).

### Expected Data Schema

Your transactional data should contain at minimum:
- Source node identifier
- Target node identifier
- Optional: Edge attributes (transaction amount, timestamp, etc.)
- Optional: Node attributes

Example CSV format:
```
source,target,amount,timestamp
node_a,node_b,100.50,2024-01-01
node_b,node_c,250.75,2024-01-02
...
```

## Usage

1. Place your raw data in `raw/`
2. Run preprocessing scripts to convert to graph format
3. Processed data will be saved to `processed/`
4. Generated graphs will be saved to `graphs/`
