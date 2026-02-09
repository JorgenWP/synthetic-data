#!/bin/bash

# 1. Install PyTorch (1.13.1 + CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 2. Install DGL (0.9.1 + CUDA 11.7)
pip install dgl==0.9.1 -f https://data.dgl.ai/wheels/cu117/repo.html

# 3. Install PyG Dependencies (Torch 1.13.1 + CUDA 11.7)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# 4. Install PyTorch Geometric and urllib patch
pip install torch-geometric
pip install "urllib3<2.0"