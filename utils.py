#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def normalize_data(data, method='z-score'):
    """
    对数据进行归一化处理
    
    参数:
        data: 输入数据，形状为 (n_samples, n_features)
        method: 归一化方法，可选 'z-score' 或 'min-max'
        
    返回:
        归一化后的数据
    """
    if method == 'z-score':
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
    elif method == 'min-max':
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    return normalized_data

def euclidean_distance(data):
    """
    计算欧氏距离矩阵
    
    参数:
        data: 输入数据，形状为 (n_samples, n_features)
        
    返回:
        距离矩阵
    """
    distances = squareform(pdist(data, 'euclidean'))
    return distances

def construct_similarity_graph(data, k=20, metric='euclidean', sigma=1.0):
    """
    构建相似度图 (KNN图)

    参数:
        data: 输入数据, shape为(n_samples, n_features)
        k: KNN中的k值
        metric: 距离度量方式，'euclidean'或'cosine'
        sigma: 高斯核宽度参数，用于将距离转换为相似度

    返回:
        相似度矩阵, shape为(n_samples, n_samples)
    """
    n_samples = data.shape[0]
    
    # 计算距离矩阵
    if metric == 'euclidean':
        dist_matrix = euclidean_distances(data)
    elif metric == 'cosine':
        # 余弦相似度转换为距离: 1 - cosine_similarity
        dist_matrix = 1 - cosine_similarity(data)
    else:
        raise ValueError(f"不支持的距离度量方式: {metric}, 请使用'euclidean'或'cosine'")
    
    # 初始化相似度矩阵
    sim_matrix = np.zeros((n_samples, n_samples))
    
    # 对每个样本，找到其k近邻
    for i in range(n_samples):
        # 获取距离并排序
        dist_i = dist_matrix[i]
        # 找出k个最近的样本的索引（不包括自己）
        indices = np.argsort(dist_i)[1:k+1]
        
        # 计算相似度值并填充相似度矩阵
        for j in indices:
            # 高斯核计算相似度: exp(-dist^2/(2*sigma^2))
            sim_value = np.exp(-dist_matrix[i, j]**2 / (2 * sigma**2))
            sim_matrix[i, j] = sim_value
            # 保持对称性
            sim_matrix[j, i] = sim_value
    
    # 归一化相似度矩阵（行和归一化）
    row_sums = sim_matrix.sum(axis=1)
    sim_matrix = sim_matrix / row_sums[:, np.newaxis]
    
    return sim_matrix

def snf_fusion(similarity_matrices, k=20, t=20):
    """
    相似度网络融合 (Similarity Network Fusion)

    参数:
        similarity_matrices: 相似度矩阵列表，每个矩阵对应一个视图
        k: 局部邻域大小
        t: 迭代次数

    返回:
        融合后的相似度矩阵
    """
    if not similarity_matrices:
        raise ValueError("相似度矩阵列表不能为空")
    
    n_views = len(similarity_matrices)
    n_samples = similarity_matrices[0].shape[0]
    
    # 检查所有矩阵尺寸是否一致
    for i in range(n_views):
        if similarity_matrices[i].shape != (n_samples, n_samples):
            raise ValueError(f"所有相似度矩阵必须具有相同的尺寸 ({n_samples}, {n_samples})")
    
    # 创建局部相似度矩阵
    P = []
    for i in range(n_views):
        P_i = np.zeros((n_samples, n_samples))
        sim_i = similarity_matrices[i].copy()
        
        # 对每个样本，保留其k近邻的相似度
        for j in range(n_samples):
            # 获取非零相似度的索引
            idx = sim_i[j, :].nonzero()[0]
            # 如果非零项少于k个，则使用所有非零项
            if len(idx) < k:
                P_i[j, idx] = sim_i[j, idx]
            else:
                # 否则选取相似度最高的k个
                idx_k = np.argsort(-sim_i[j, idx])[:k]
                P_i[j, idx[idx_k]] = sim_i[j, idx[idx_k]]
        
        # 归一化
        P_i = P_i / P_i.sum(axis=1, keepdims=True)
        P.append(P_i)
    
    # 初始化融合矩阵
    W = similarity_matrices.copy()
    
    # 迭代融合
    for _ in range(t):
        W_next = []
        
        for i in range(n_views):
            # 计算其他视图的平均
            S = np.zeros((n_samples, n_samples))
            for j in range(n_views):
                if j != i:
                    S += W[j]
            
            S = S / (n_views - 1)
            
            # 更新当前视图
            W_i_next = P[i] @ S @ P[i].T
            W_next.append(W_i_next)
        
        W = W_next
    
    # 计算最终的融合矩阵
    W_fused = np.zeros((n_samples, n_samples))
    for i in range(n_views):
        W_fused += W[i]
    
    W_fused = W_fused / n_views
    
    return W_fused

def to_sparse_tensor(dense_matrix):
    """
    将稠密矩阵转换为PyTorch稀疏张量

    参数:
        dense_matrix: 稠密矩阵

    返回:
        PyTorch稀疏张量
    """
    # 转换为COO格式的稀疏矩阵
    sparse_matrix = sp.coo_matrix(dense_matrix)
    
    # 提取坐标和值
    indices = torch.LongTensor([sparse_matrix.row, sparse_matrix.col])
    values = torch.FloatTensor(sparse_matrix.data)
    shape = torch.Size(sparse_matrix.shape)
    
    # 创建稀疏张量
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    
    return sparse_tensor

def dimension_reduction(data, n_components=2, method='pca'):
    """
    降维
    
    参数:
        data: 输入数据矩阵，形状为 (n_samples, n_features)
        n_components: 目标维度
        method: 降维方法，目前支持 'pca'
        
    返回:
        reduced_data: 降维后的数据
    """
    if method == 'pca':
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    return reduced_data

def evaluate_clustering(embeddings, true_labels, pred_labels=None):
    """
    评估聚类性能

    参数:
        embeddings: 特征嵌入
        true_labels: 真实标签
        pred_labels: 预测标签，如果为None则使用KMeans重新聚类

    返回:
        NMI和ARI评估指标
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    
    if pred_labels is None:
        # 获取聚类数量
        n_clusters = len(np.unique(true_labels))
        
        # 使用KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        pred_labels = kmeans.fit_predict(embeddings)
    
    # 计算NMI (Normalized Mutual Information)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    # 计算ARI (Adjusted Rand Index)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    return nmi, ari

def visualize_embeddings(embeddings, labels, title='嵌入可视化', method='tsne'):
    """
    可视化嵌入

    参数:
        embeddings: 特征嵌入, shape为(n_samples, n_features)
        labels: 样本标签
        title: 图表标题
        method: 降维方法，'tsne'或'pca'
    """
    import matplotlib.pyplot as plt
    
    # 如果嵌入维度大于2，则进行降维
    if embeddings.shape[1] > 2:
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=0)
        else:  # default to PCA
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    
    # 为每个类别绘制散点图
    for i, label in enumerate(unique_labels):
        # 获取该类别的样本
        idx = labels == label
        # 绘制散点图
        plt.scatter(
            embeddings_2d[idx, 0], embeddings_2d[idx, 1],
            label=f'Class {label}',
            alpha=0.7,
            edgecolors='w'
        )
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf() 