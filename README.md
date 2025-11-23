# Synthetic Graph Data - Deep Graph Generation using BiGG

A framework for training generative models to synthesize transactional graph data using the BIg Graph Generation (BiGG) model.

## 🎯 Overview

This repository provides code for developing a generative AI model that can learn from and synthesize transactional graph data. It includes data preprocessing, model training and generation, evaluation metrics, and visualization. 

The repository integrates and extends exsisting open-source graph generation frameworks, attributed [here](#code-attribution).

## 📁 Project Structure

```
synthetic-data/
├── bigg/           # BiGG repo
│   ├── bigg/           # Source code
|   │   ├── common/
|   │   ├── data_process/
|   │   │   ├── preprocess_transactions.py
|   │   ├── experiments/
|   │   │   └── synthetic/
|   │   │       └── scripts/
|   │   │           └── run_transactions.sh
|   │   ├── extension/
|   │   ├── model/
|   │   │   └── tree_clib/
|   │   │       ├── src/
|   |   |       └── Makefile
|   │   ├── torch_ops/
|   │   └── unit_test/
│   ├── data/           # Data 
│   ├── setup.py    
│   └── README.md                                    
├── GRAN/           # GRAN repo, needs to be cloned from https://github.com/lrjconan/GRAN
└── README.md       # This file
```

## 🆕 Extensions to Existing Frameworks

For preprocessing of the transactional data the following files have been created:

- [preprocess_transactions.py](bigg/bigg/data_process/preprocess_transactions.py): Script for converting raw transaction CSV data into graph structures.
- [run_transactions_datagen.sh](bigg/bigg/data_process/run_transactions_datagen.sh): Shell script to run the preprocessing with configurable parameters.


For training and generating synthetic graphs, the following scripts have been created/modified:

- [batch_train.py](bigg/bigg/experiments/synthetic/batch_train.py): Script for training the BiGG model on transactional graph data (slightly modified).
- [run_transaction.sh](bigg/bigg/experiments/synthetic/scripts/run_transaction.sh): Shell script to run training and generation with configurable parameters.

A little modification was needed for running the original BiGG model code:

- [tree_lib.py](bigg/bigg/model/tree_clib/tree_lib.py): Modified C library and Makefile for tree operations.

For evaluation the following scripts have been created:

- [evaluate.py](bigg/bigg/experiments/synthetic/evaluate.py): Script for evaluating calculation of graph quality metrics using implementation from the GRAN repository.
- [run_evaluate.sh](bigg/bigg/experiments/synthetic/scripts/run_evaluate.sh): Shell script to run evaluation with configurable parameters.

- [calc_stats.py](bigg/bigg/data_process/calc_stats.py): Script for calculating statistical metrics on graphs.

For visualization the following script have been created:

- [visualize_graphs.py](bigg/bigg/experiments/synthetic/scripts/visualize_graphs.py): Script for visualizing graphs.

To enable visualization in graph tool, the following scripts have been created:

- [edge_list_gen.py](bigg/bigg/data_process/edge_list_gen.py): Script for generating edge lists from graphs.
- [validation_edge_list_gen.py](bigg/bigg/data_process/validation_edge_list_gen.py): Script for generating edge lists for training and validation graphs.

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

Using conda:

```bash
conda create -n synthetic-data python=3.8
conda activate synthetic-data
```

3. Navigate to the BiGG directory:
```bash
cd bigg
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Install the BiGG package:
```bash
pip install -e .
```

6. Add the GRAN repo:

Navigate to the root of the repository and clone the [GRAN repository](https://github.com/lrjconan/GRAN):

```bash
git clone https://github.com/lrjconan/GRAN.git
```

Then install the GRAN dependencies:
```bash
pip install -r GRAN/requirements.txt
```

7. Add the BiGG and GRAN projects to the `PYTHONPATH` environment variable:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/bigg
export PYTHONPATH=$PYTHONPATH:$(pwd)/GRAN
```

## 📊 Training and Generation

### 1. Prepare Your Data

Place the transaction data (`GT.gpickle`) from [here](https://github.com/akratiiet/RaboBank_Dataset) in the `bigg/data/Transactions/raw/` directory. 

### 2. Preprocess into Graphs

Convert your transactional data into the desired graph structures:

1. Navigate to the data process directory:
```bash
cd bigg/bigg/data_process
```

2. Configure the parameters in `run_transactions_datagen.sh` as needed.

3. Run the preprocessing script:
```bash
chmod +x run_transactions_datagen.sh
./run_transactions_datagen.sh
```

### 3. Train the Model

1. Navigate to the model training/generation directory:
```bash
cd ../experiments/synthetic/scripts
```

2. Edit the file `run_transaction.sh` to customize training parameters.

3. Train using the command-line script:
```bash
chmod +x run_transactions.sh
./run_transactions.sh
```

The trained model will be placed in the `results/` directory.

### 4. Generate Synthetic Graphs

Generate new synthetic graphs using the trained model:
```bash
./run_transaction.sh \
    -phase test \     # 'test' phase for generation
    -epoch_load 25 \  # Epoch number to load
    -num_test_gen 1 \ # Number of graphs to generate
    -display True \
```

### 5. Evaluate Results

Evaluate the quality of generated graphs using the command-line script:
```bash
chmod +x run_evaluate.sh
./run_evaluate.sh
```

### 6. Visualize Generated Graphs

Visualize the generated graphs using the python scripts:
> Note: This is only a basic visualization and is bad for large or complex graphs. For more detailed analysis, consider using specialized graph visualization tools like Gephi.
```bash
python visualize_graphs.py
```


---

## Code Attribution

This project builds upon existing open-source work in the graph generation community. We include and adapt components from the following:

- **BiGG** ([Google Research](https://github.com/google-research/google-research/tree/master/bigg))  
  Apache 2.0 License
  The BiGG project has been used and slightly modified for this work.

<!-- - **GraphRNN** ([Jiaxuan You et al.](https://github.com/JiaxuanYou/graph-generation))  
  MIT License  
  Evaluation utilities (e.g., MMD code) are adapted and extended. -->

- **GRAN** ([Lirui Jia et al.](https://github.com/lrjconan/GRAN))  
  MIT License  
  Additional evaluation utilities.
