# Synthetic Graph Data - Deep Graph Generation using BiGG

A framework for training generative models to synthesize transactional graph data using the BIg Graph Generation (BiGG) model.

## 🎯 Overview

This repository provides code for developing a generative AI model that can learn from and synthesize transactional graph data. It includes data preprocessing, model training and generation, evaluation metrics, and visualization tools. 

The repository integrates and extends exsisting open-source graph generation frameworks, attributed [here](#code-attribution).

## 📁 Project Structure

```
synthetic-data/
├── bigg/                   # BiGG repo
│   ├── bigg/                   # Source code
|   │   ├── common/
|   │   ├── data_process/
|   │   ├── experiments/
|   │   ├── extension/
|   │   ├── model/
|   │   ├── torch_ops/
|   │   └── unit_test/
│   ├── data/                     # Data 
│   ├── setup.py    
│   └── README.md                                    
├── GRAN/                   # GRAN repo
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

## 📊 Quick Start

### 1. Prepare Your Data

Place the transaction data (`transactions_data.csv`) from [this](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets) Kaggle page in `bigg/data/Transactions/raw/`. 

The CSV format for `transactions_data.csv`:
```csv
id,date,client_id,card_id,amount,use_chip,merchant_id,merchant_city,merchant_state,zip,mcc,errors
7475327,2010-01-01 00:01:00,1556,2972,$-77.00,Swipe Transaction,59935,Beulah,ND,58523.0,5499,
7475328,2010-01-01 00:02:00,561,4575,$14.57,Swipe Transaction,67570,Bettendorf,IA,52722.0,5311,
```

### 2. Preprocess into Graphs

Convert your transactional data into graph structures:
```bash
cd bigg/bigg/data_process
python preprocess_transactions.py -save_dir ../../data/Transactions/transactions-BFS -node_order BFS
```

Optionally you could add a `start_date` and `cutoff_date` to limit the data range:
```bash
python preprocess_transactions.py -save_dir ../../data/Transactions/transactions-BFS -node_order BFS -start_date 2018-01-01 -cutoff_date 2019-01-01
```

### 4. Configure the Model

Navigate to the model training/generation directory:
```bash
cd ../experiments/synthetic/scripts
```

Edit the file `run_transactions.sh` to customize training parameters.

### 5. Train the Model

Train using the command-line script:
```bash
chmod +x run_transactions.sh
./run_transactions.sh
```

The trained model will be placed in the `results/` directory.

### 6. Generate Synthetic Graphs

Generate new synthetic graphs using the trained model:
```bash
./run_transactions.sh \
    --phase ... \
    --example1 100 \
    --example2 256
```

### 7. Evaluate Results

Evaluate the quality of generated graphs:

...

---

## Code Attribution

This project builds upon existing open-source work in the graph generation community. We include and adapt components from the following:

- **BiGG** ([Google Research](https://github.com/google-research/google-research/tree/master/bigg))  
  Apache 2.0 License  
  Portions of this codebase (e.g., `FenwickTree`, `RecurTreeGen`, input preprocessing) are adapted from BiGG and clearly marked in-file.

<!-- - **GraphRNN** ([Jiaxuan You et al.](https://github.com/JiaxuanYou/graph-generation))  
  MIT License  
  Evaluation utilities (e.g., MMD code) are adapted and extended. -->

- **GRAN** ([Lirui Jia et al.](https://github.com/lrjconan/GRAN))  
  MIT License  
  Additional evaluation utilities are adapted.
