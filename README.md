# sxSNF

````markdown
# sxSNF-GNN: Single-cell Multi-omics Network Fusion with Graph Neural Networks

基于 **相似性网络融合 (Similarity Network Fusion, SNF)** 与 **图神经网络 (GNN)** 的单细胞多组学数据集成与聚类分析框架。  
该工具支持多组学数据的相似性图构建、SNF融合、深度图表示学习以及下游聚类和可视化分析。

---

## ✨ 特性

- 支持 **多组学数据 (multi-omics)** 输入与融合  
- 提供 **SNF** 算法实现，用于跨模态相似性网络融合  
- 内置 **多种图神经网络模型**：
  - GCN
  - GraphSAGE
  - GAT
  - VGAE
- 提供聚类 (K-means) 与指标评估 (NMI, ARI)  
- 可视化结果（t-SNE 降维、聚类散点图、嵌入特征热图）  
- 支持 `.mat` 数据文件和标签文件加载  

---

## 📦 安装与依赖

运行环境：

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- SciPy
- Matplotlib
- Seaborn

安装依赖：

```bash
pip install -r requirements.txt
````
---

## 🚀 使用方法

### 1. 数据准备

* 将输入的多组学数据存放在 `./data` 目录下，支持 `.mat` 格式。
* 数据矩阵变量名需为 `X`，或包含二维矩阵的 key。
* 可选：在 `./data/labels.txt` 提供真实标签（整数型）。

### 2. 运行示例

```bash
python main.py \
  --data_dir ./data \
  --save_dir ./results \
  --k 20 \
  --t 20 \
  --gnn_type gcn \
  --hidden_dim 128 \
  --embedding_dim 64 \
  --epochs 200 \
  --n_clusters 10 \
  --gpu 0 \
  --verbose
```

### 3. 输出结果

运行完成后结果会保存在 `./results`：

* `embeddings.npy` : GNN学习到的嵌入
* `cluster_labels.npy` : 聚类标签
* `results.mat` : MATLAB格式结果
* `metrics.txt` : 聚类评估指标（NMI, ARI）
* `clustering_visualization.png` : 聚类可视化
* `embedding_heatmap.png` : 嵌入热图

---

## 📂 目录结构

```
.
├── main.py                # 主程序
├── utils.py               # 工具函数：归一化、图构建、SNF、降维
├── graph_module.py        # 图神经网络定义与训练
├── data/                  # 输入数据目录 (.mat, labels.txt)
├── results/               # 输出结果目录
└── README.md              # 项目说明文档
```
