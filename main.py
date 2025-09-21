#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块
from utils import normalize_data, construct_similarity_graph, snf_fusion, dimension_reduction
from graph_module import SNF_GNN, train_gnn

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='单细胞网络融合与深度图学习')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    
    # SNF参数
    parser.add_argument('--k', type=int, default=20, help='KNN图中的邻居数')
    parser.add_argument('--t', type=int, default=20, help='SNF迭代次数')
    
    # 图神经网络参数
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--embedding_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'graphsage', 'gat', 'vgae'], help='GNN类型')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    
    # 聚类参数
    parser.add_argument('--n_clusters', type=int, default=0, help='聚类数量，0表示自动判断')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID, -1表示使用CPU')
    parser.add_argument('--verbose', action='store_true', help='是否显示详细输出')
    
    return parser.parse_args()

def setup_environment(args):
    """设置环境变量和随机种子"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    return device

def load_data(args):
    """
    加载单细胞数据
    
    返回:
        data_list: 多组学数据列表
        labels: 标签（如果有）
    """
    if args.verbose:
        print("加载数据...")
    
    # 示例：加载.mat文件
    # 实际应用中可以根据需要修改加载方式
    data_files = [f for f in os.listdir(args.data_dir) if f.endswith('.mat')]
    data_list = []
    
    for file in data_files:
        file_path = os.path.join(args.data_dir, file)
        try:
            # 加载MAT文件
            mat_data = loadmat(file_path)
            
            # 假设数据矩阵的变量名为'X'，每个文件可能需要根据实际情况调整
            if 'X' in mat_data:
                data_matrix = mat_data['X']
                data_list.append(data_matrix)
            else:
                # 尝试找到矩阵数据
                for key, value in mat_data.items():
                    if isinstance(value, np.ndarray) and len(value.shape) == 2 and min(value.shape) > 1:
                        if not key.startswith('__'):  # 排除mat文件的元数据
                            data_list.append(value)
                            if args.verbose:
                                print(f"从{file}加载数据矩阵 {key}: 形状{value.shape}")
                            break
        except Exception as e:
            print(f"无法加载文件 {file}: {e}")
    
    # 尝试加载标签
    label_file = os.path.join(args.data_dir, 'labels.txt')
    labels = None
    
    if os.path.exists(label_file):
        try:
            labels = np.loadtxt(label_file, dtype=int)
            if args.verbose:
                print(f"加载标签: 形状{labels.shape}")
        except:
            print(f"无法加载标签文件 {label_file}")
    
    if not data_list:
        raise ValueError("未能加载任何数据。请检查数据目录和文件格式。")
    
    return data_list, labels

def snf_process(data_list, args):
    """
    执行SNF网络融合过程
    
    参数:
        data_list: 多组学数据列表
        args: 命令行参数
        
    返回:
        fused_network: 融合后的网络
        similarity_matrices: 相似度矩阵列表
    """
    if args.verbose:
        print("执行数据预处理和网络融合...")
    
    # # 数据预处理
    # processed_data = []
    # for data in data_list:
    #     # 标准化数据
    #     norm_data = normalize_data(data, method='minmax')
    #     processed_data.append(norm_data)
    
    for data in data_list:
        # 标准化数据
        # norm_data = normalize_data(data, method='minmax')
        processed_data.append(data)
    
    # 构建相似度图
    similarity_matrices = []
    for data in processed_data:
        sim_matrix = construct_similarity_graph(data, k=args.k)
        similarity_matrices.append(sim_matrix)
    
    # 网络融合
    fused_network = snf_fusion(similarity_matrices, t=args.t, k=args.k)
    
    return fused_network, similarity_matrices

def gnn_embedding(fused_network, args, device):
    """
    使用GNN对融合网络进行嵌入
    
    参数:
        fused_network: 融合后的网络
        args: 命令行参数
        device: 计算设备
        
    返回:
        embeddings: 嵌入向量
    """
    if args.verbose:
        print(f"使用{args.gnn_type.upper()}进行图嵌入...")
    
    # 准备输入特征（这里使用单位矩阵作为节点特征）
    n = fused_network.shape[0]
    features = torch.eye(n).to(device)
    
    # 转换邻接矩阵为稀疏张量
    adj = torch.FloatTensor(fused_network).to(device)
    
    # 创建GNN模型
    model = SNF_GNN(
        input_dim=n,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        gnn_type=args.gnn_type,
        dropout=args.dropout
    )
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 训练模型
    model = train_gnn(model, features, adj, optimizer, epochs=args.epochs, verbose=args.verbose)
    
    # 获取嵌入
    model.eval()
    with torch.no_grad():
        if args.gnn_type == 'vgae':
            embeddings, _, _ = model(features, adj)
        else:
            embeddings = model(features, adj)
        embeddings = embeddings.cpu().numpy()
    
    return embeddings

def perform_clustering(embeddings, labels, args):
    """
    对嵌入向量进行聚类并评估结果
    
    参数:
        embeddings: 嵌入向量
        labels: 真实标签（如果有）
        args: 命令行参数
        
    返回:
        cluster_labels: 聚类标签
        metrics: 评估指标（如果有真实标签）
    """
    if args.verbose:
        print("执行聚类分析...")
    
    # 确定聚类数量
    n_clusters = args.n_clusters
    if n_clusters <= 0 and labels is not None:
        n_clusters = len(np.unique(labels))
    elif n_clusters <= 0:
        # 如果没有指定聚类数量且没有标签，使用默认值
        n_clusters = 10
        
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 评估结果（如果有真实标签）
    metrics = {}
    if labels is not None:
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        ari = adjusted_rand_score(labels, cluster_labels)
        
        metrics = {
            'NMI': nmi,
            'ARI': ari
        }
        
        if args.verbose:
            print(f"聚类评估 - NMI: {nmi:.4f}, ARI: {ari:.4f}")
    
    return cluster_labels, metrics

def visualize_results(embeddings, labels, cluster_labels, args):
    """
    可视化嵌入和聚类结果
    
    参数:
        embeddings: 嵌入向量
        labels: 真实标签（如果有）
        cluster_labels: 聚类标签
        args: 命令行参数
    """
    if args.verbose:
        print("生成可视化结果...")
    
    # 使用t-SNE降维到2D进行可视化
    if embeddings.shape[1] > 2:
        tsne_embed = dimension_reduction(embeddings, n_components=2, method='tsne')
    else:
        tsne_embed = embeddings
    
    # 绘制聚类结果
    plt.figure(figsize=(12, 5))
    
    # 绘制预测的聚类
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=cluster_labels, cmap='viridis', s=20)
    plt.colorbar(scatter)
    plt.title('预测聚类')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # 如果有真实标签，绘制真实标签
    if labels is not None:
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=labels, cmap='viridis', s=20)
        plt.colorbar(scatter)
        plt.title('真实标签')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'clustering_visualization.png'), dpi=300)
    
    # 绘制嵌入特征的热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(embeddings[:min(100, embeddings.shape[0])], cmap='viridis')
    plt.title('嵌入特征热图 (前100个样本)')
    plt.savefig(os.path.join(args.save_dir, 'embedding_heatmap.png'), dpi=300)
    
    plt.close('all')

def save_results(embeddings, cluster_labels, metrics, args):
    """
    保存结果
    
    参数:
        embeddings: 嵌入向量
        cluster_labels: 聚类标签
        metrics: 评估指标
        args: 命令行参数
    """
    if args.verbose:
        print("保存结果...")
    
    # 保存嵌入向量
    np.save(os.path.join(args.save_dir, 'embeddings.npy'), embeddings)
    
    # 保存聚类标签
    np.save(os.path.join(args.save_dir, 'cluster_labels.npy'), cluster_labels)
    
    # 保存MAT格式结果（用于MATLAB分析）
    savemat(os.path.join(args.save_dir, 'results.mat'), {
        'embeddings': embeddings,
        'cluster_labels': cluster_labels
    })
    
    # 保存评估指标
    if metrics:
        with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
                
    if args.verbose:
        print(f"结果已保存到 {args.save_dir}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置环境
    device = setup_environment(args)
    
    # 加载数据
    data_list, labels = load_data(args)
    
    # SNF网络融合
    fused_network, similarity_matrices = snf_process(data_list, args)
    
    # GNN嵌入
    embeddings = gnn_embedding(fused_network, args, device)
    
    # 聚类分析
    cluster_labels, metrics = perform_clustering(embeddings, labels, args)
    
    # 可视化结果
    visualize_results(embeddings, labels, cluster_labels, args)
    
    # 保存结果
    save_results(embeddings, cluster_labels, metrics, args)
    
    if args.verbose:
        print("sxSNF算法执行完毕！")

if __name__ == "__main__":
    main() 