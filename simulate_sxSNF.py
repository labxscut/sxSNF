#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import sys
import argparse

# 将sxSNF代码目录添加到路径
sys.path.append('sxSNF/code')

# 导入sxSNF模块
from utils import normalize_data, construct_similarity_graph, snf_fusion, dimension_reduction
from graph_module import SNF_GNN, train_gnn

def generate_simulated_data(n_samples=500, n_features1=1000, n_features2=800, n_clusters=3, random_seed=42):
    """
    生成模拟的单细胞多模态数据
    
    参数:
        n_samples: 样本数量（细胞数）
        n_features1: 第一个模态的特征数量
        n_features2: 第二个模态的特征数量
        n_clusters: 聚类数量
        random_seed: 随机种子
        
    返回:
        data_list: 包含两个模态数据的列表
        labels: 真实标签
    """
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 生成聚类中心
    centers1 = np.random.randn(n_clusters, n_features1) * 5
    centers2 = np.random.randn(n_clusters, n_features2) * 5
    
    # 初始化数据矩阵
    data1 = np.zeros((n_samples, n_features1))
    data2 = np.zeros((n_samples, n_features2))
    labels = np.zeros(n_samples, dtype=int)
    
    # 为每个聚类生成样本
    samples_per_cluster = n_samples // n_clusters
    for i in range(n_clusters):
        start_idx = i * samples_per_cluster
        end_idx = (i + 1) * samples_per_cluster if i < n_clusters - 1 else n_samples
        
        # 第一个模态数据: 基因表达矩阵
        data1[start_idx:end_idx] = centers1[i] + np.random.randn(end_idx - start_idx, n_features1) * 1.5
        
        # 第二个模态数据: 表观遗传学数据
        data2[start_idx:end_idx] = centers2[i] + np.random.randn(end_idx - start_idx, n_features2) * 1.5
        
        # 设置标签
        labels[start_idx:end_idx] = i
    
    # 添加噪声和稀疏性（模拟单细胞数据的特点）
    # 第一个模态: 基因表达数据通常是稀疏的
    sparsity1 = 0.7  # 70%的值为0
    mask1 = np.random.random(data1.shape) < sparsity1
    data1[mask1] = 0
    
    # 第二个模态: 表观遗传学数据
    sparsity2 = 0.5  # 50%的值为0
    mask2 = np.random.random(data2.shape) < sparsity2
    data2[mask2] = 0
    
    # 将负值设为0（模拟基因表达和甲基化数据的非负性）
    data1 = np.maximum(data1, 0)
    data2 = np.maximum(data2, 0)
    
    print(f"生成的数据形状 - 模态1: {data1.shape}, 模态2: {data2.shape}")
    print(f"标签分布: {np.bincount(labels)}")
    
    # 保存生成的模拟数据
    if not os.path.exists("simulated_data"):
        os.makedirs("simulated_data")
    
    np.save("simulated_data/modality1_data.npy", data1)
    np.save("simulated_data/modality2_data.npy", data2)
    np.save("simulated_data/labels.npy", labels)
    
    print(f"已保存模拟数据到 simulated_data 目录")
    
    return [data1, data2], labels

def visualize_data(data_list, labels, file_name="simulated_data_visualization.png"):
    """
    可视化模拟数据
    """
    plt.figure(figsize=(15, 6))
    
    # 可视化原始数据的一部分
    for i, data in enumerate(data_list):
        plt.subplot(1, 3, i+1)
        sns.heatmap(data[:50, :50], cmap="viridis")
        plt.title(f"Modality {i+1} Data Sample (First 50 rows and columns)")
    
    # 可视化标签分布
    plt.subplot(1, 3, 3)
    sns.countplot(x=labels)
    plt.title("Label Distribution")
    
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def visualize_loss_curve(loss_file="training_process/loss_history.npy"):
    """
    可视化训练过程中的损失曲线
    """
    if not os.path.exists(loss_file):
        print(f"损失历史文件 {loss_file} 不存在")
        return
    
    losses = np.load(loss_file)
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # 使用对数刻度更好地显示小的损失值
    plt.grid(True)
    plt.savefig("training_loss_curve.png", dpi=300)
    plt.close()
    print("已保存损失曲线图到 training_loss_curve.png")

def run_sxSNF(data_list, labels, args, save_results=True):
    """
    运行sxSNF方法
    
    参数:
        data_list: 多模态数据列表
        labels: 真实标签
        args: 参数设置
        save_results: 是否保存结果
    """
    print("开始执行sxSNF算法...")
    
    # 数据预处理
    processed_data = []
    for data in data_list:
        # 标准化数据
        norm_data = normalize_data(data, method='min-max')
        processed_data.append(norm_data)
    
    # 构建相似度图
    print("构建相似度图...")
    similarity_matrices = []
    for data in processed_data:
        sim_matrix = construct_similarity_graph(data, k=args.k)
        similarity_matrices.append(sim_matrix)
    
    # 网络融合
    print(f"使用SNF进行网络融合 (k={args.k}, t={args.t})...")
    fused_network = snf_fusion(similarity_matrices, k=args.k, t=args.t)
    
    # 强制使用CUDA:0设备
    device = torch.device('cuda:0')
    print(f"使用设备: {device}")
    
    # 确保CUDA可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        device = torch.device('cpu')
    
    # GNN嵌入
    n = fused_network.shape[0]
    features = torch.eye(n).to(device)
    adj = torch.FloatTensor(fused_network).to(device)
    
    # 修改graph_module.py中的GraphConvolution类
    # 手动修复设备不匹配问题
    class FixedGraphConvolution(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=True, device=device):
            super(FixedGraphConvolution, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features).to(device))
            if bias:
                self.bias = torch.nn.Parameter(torch.FloatTensor(out_features).to(device))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
            
        def reset_parameters(self):
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
                
        def forward(self, input, adj):
            # 确保输入张量在正确的设备上
            input = input.to(self.weight.device)
            adj = adj.to(self.weight.device)
            
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output
    
    # 使用自定义的GCNEncoder
    class FixedGCNEncoder(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, embedding_dim, dropout=0.5, activation=torch.nn.functional.relu, device=device):
            super(FixedGCNEncoder, self).__init__()
            self.gc1 = FixedGraphConvolution(input_dim, hidden_dim, device=device)
            self.gc2 = FixedGraphConvolution(hidden_dim, embedding_dim, device=device)
            self.dropout = dropout
            self.activation = activation
            
        def forward(self, x, adj):
            x = self.activation(self.gc1(x, adj))
            x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
            return x
    
    # 创建自定义的SNF_GNN模型
    class FixedSNF_GNN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, embedding_dim, dropout=0.5, device=device):
            super(FixedSNF_GNN, self).__init__()
            self.gnn = FixedGCNEncoder(input_dim, hidden_dim, embedding_dim, dropout, device=device)
            
        def forward(self, x, adj):
            return self.gnn(x, adj)
    
    # 创建修复后的模型
    print("创建 GCN 模型进行图嵌入...")
    model = FixedSNF_GNN(
        input_dim=n,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        device=device
    ).to(device)
    
    # 确保模型所有参数都在正确设备上
    for param in model.parameters():
        param.data = param.data.to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 自定义简单的训练函数
    def simple_train(model, features, adj, optimizer, epochs=100):
        model.train()
        
        # 创建保存训练过程的目录
        if not os.path.exists("training_process"):
            os.makedirs("training_process")
            
        # 保存初始嵌入
        with torch.no_grad():
            initial_embeddings = model(features, adj).cpu().numpy()
            np.save("training_process/embeddings_epoch_0.npy", initial_embeddings)
        
        losses = []
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            
            # 前向传播
            output = model(features, adj)
            
            # 使用简单的重构损失
            loss = torch.nn.functional.mse_loss(output @ output.t(), adj)
            losses.append(loss.item())
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 每10个epoch保存嵌入
            if epoch % 10 == 0:
                with torch.no_grad():
                    embeddings = model(features, adj).cpu().numpy()
                    np.save(f"training_process/embeddings_epoch_{epoch}.npy", embeddings)
            
            # 打印进度，使用更高精度显示损失
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.10f}")
        
        # 保存损失历史
        np.save("training_process/loss_history.npy", np.array(losses))
        
        return model
    
    # 训练模型
    model = simple_train(model, features, adj, optimizer, epochs=args.epochs)
    
    # 获取嵌入
    model.eval()
    with torch.no_grad():
        embeddings = model(features, adj)
        embeddings = embeddings.cpu().numpy()
    
    # 使用t-SNE降维进行可视化
    from sklearn.manifold import TSNE
    if embeddings.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=args.seed)
        tsne_embed = tsne.fit_transform(embeddings)
    else:
        tsne_embed = embeddings
    
    # 聚类分析
    from sklearn.cluster import KMeans
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 评估聚类结果
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)
    print(f"聚类评估 - NMI: {nmi:.4f}, ARI: {ari:.4f}")
    
    # 可视化嵌入和聚类结果
    plt.figure(figsize=(12, 5))
    
    # 预测聚类
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=cluster_labels, cmap='viridis', s=20)
    plt.colorbar(scatter)
    plt.title('Predicted Clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # 真实标签
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=labels, cmap='viridis', s=20)
    plt.colorbar(scatter)
    plt.title('True Labels')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig("sxSNF_clustering_results.png", dpi=300)
    
    # 保存结果
    if save_results:
        if not os.path.exists("results"):
            os.makedirs("results")
        np.save("results/embeddings.npy", embeddings)
        np.save("results/cluster_labels.npy", cluster_labels)
        
        # 保存评估指标
        with open("results/metrics.txt", "w") as f:
            f.write(f"NMI: {nmi:.4f}\n")
            f.write(f"ARI: {ari:.4f}\n")
    
    print("sxSNF算法执行完毕！")
    
    # 可视化损失曲线
    visualize_loss_curve()
    
    return embeddings, cluster_labels, {"NMI": nmi, "ARI": ari}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='sxSNF模拟数据实验')
    
    # 数据生成参数
    parser.add_argument('--n_samples', type=int, default=500, help='样本数量')
    parser.add_argument('--n_features1', type=int, default=100, help='第一个模态的特征数量')
    parser.add_argument('--n_features2', type=int, default=80, help='第二个模态的特征数量')
    parser.add_argument('--n_clusters', type=int, default=3, help='聚类数量')
    
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
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID, -1表示使用CPU')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # # 生成模拟数据
    # print("生成模拟的单细胞多模态数据...")
    # data_list, labels = generate_simulated_data(
    #     n_samples=args.n_samples,
    #     n_features1=args.n_features1,
    #     n_features2=args.n_features2,
    #     n_clusters=args.n_clusters,
    #     random_seed=args.seed
    # )
    
    # 检查是否使用模拟数据或加载真实数据
    use_simulated_data = False  # 设置为False表示使用真实数据
    
    if not use_simulated_data:
        print("加载真实的单细胞多模态数据...")
        try:
            # 从指定目录加载数据
            rna_data = np.load('./datasets/rna_pca.npy')
            atac_data = np.load('./datasets/atac_lsi.npy')
            labels = np.load('./datasets/numeric_labels.npy')
            
            print(f"RNA数据形状: {rna_data.shape}")
            print(f"ATAC数据形状: {atac_data.shape}")
            print(f"标签形状: {labels.shape}")
            
            # 将数据放入列表中，与模拟数据格式保持一致
            data_list = [rna_data, atac_data]
            
            # 检查数据一致性
            if rna_data.shape[0] != atac_data.shape[0] or rna_data.shape[0] != labels.shape[0]:
                raise ValueError("数据样本数量不一致，请检查数据！")
                
            print(f"成功加载数据: {len(data_list)}个模态，{rna_data.shape[0]}个样本")
            
        except FileNotFoundError as e:
            print(f"错误: 无法找到数据文件 - {e}")
            print("将使用模拟数据代替...")
            use_simulated_data = True
        except Exception as e:
            print(f"加载数据时出错: {e}")
            print("将使用模拟数据代替...")
            use_simulated_data = True
    
    # 如果加载真实数据失败或选择使用模拟数据，则生成模拟数据
    if use_simulated_data:
        print("生成模拟的单细胞多模态数据...")
        data_list, labels = generate_simulated_data(
            n_samples=args.n_samples,
            n_features1=args.n_features1,
            n_features2=args.n_features2,
            n_clusters=args.n_clusters,
            random_seed=args.seed
        )
    
    # 可视化模拟数据
    visualize_data(data_list, labels)
    
    # 运行sxSNF算法
    embeddings, cluster_labels, metrics = run_sxSNF(data_list, labels, args)
    
    print("实验完成！")
    print(f"聚类性能 - NMI: {metrics['NMI']:.4f}, ARI: {metrics['ARI']:.4f}")

if __name__ == "__main__":
    main() 