#!/bin/bash

# 运行模拟实验的脚本

# 创建结果目录
mkdir -p results

# 运行模拟实验
echo "开始sxSNF模拟实验..."

python simulate_sxSNF.py \
    --n_samples 500 \
    --n_features1 1000 \
    --n_features2 800 \
    --n_clusters 3 \
    --k 20 \
    --t 20 \
    --hidden_dim 128 \
    --embedding_dim 64 \
    --dropout 0.5 \
    --gnn_type gcn \
    --lr 0.01 \
    --weight_decay 5e-4 \
    --epochs 100 \
    --seed 42 \
    --gpu -1

echo "实验完成! 结果保存在results目录和当前目录下" 