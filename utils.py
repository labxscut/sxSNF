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
    """Normalize features with the selected method."""
    if method == 'z-score':
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
    elif method == 'min-max':
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized_data

def euclidean_distance(data):
    """Compute pairwise Euclidean distances."""
    distances = squareform(pdist(data, 'euclidean'))
    return distances

def construct_similarity_graph(data, k=20, metric='euclidean', sigma=1.0):
    """Build a KNN-based similarity graph."""
    n_samples = data.shape[0]
    
    # 
    if metric == 'euclidean':
        dist_matrix = euclidean_distances(data)
    elif metric == 'cosine':
        # : 1 - cosine_similarity
        dist_matrix = 1 - cosine_similarity(data)
    else:
        raise ValueError(
            f"Unsupported metric: {metric}. Use 'euclidean' or 'cosine'."
        )
    
    # 
    sim_matrix = np.zeros((n_samples, n_samples))
    
    # ，k
    for i in range(n_samples):
        # 
        dist_i = dist_matrix[i]
        # k（）
        indices = np.argsort(dist_i)[1:k+1]
        
        # 
        for j in indices:
            # : exp(-dist^2/(2*sigma^2))
            sim_value = np.exp(-dist_matrix[i, j]**2 / (2 * sigma**2))
            sim_matrix[i, j] = sim_value
            # 
            sim_matrix[j, i] = sim_value
    
    # （）
    row_sums = sim_matrix.sum(axis=1)
    sim_matrix = sim_matrix / row_sums[:, np.newaxis]
    
    return sim_matrix

def snf_fusion(similarity_matrices, k=20, t=20):
    """Fuse similarity networks with SNF iterations."""
    if not similarity_matrices:
        raise ValueError("similarity_matrices must not be empty.")
    
    n_views = len(similarity_matrices)
    n_samples = similarity_matrices[0].shape[0]
    
    # 
    for i in range(n_views):
        if similarity_matrices[i].shape != (n_samples, n_samples):
            raise ValueError(
                f"All matrices must have shape ({n_samples}, {n_samples})."
            )
    
    # 
    P = []
    for i in range(n_views):
        P_i = np.zeros((n_samples, n_samples))
        sim_i = similarity_matrices[i].copy()
        
        # ，k
        for j in range(n_samples):
            # 
            idx = sim_i[j, :].nonzero()[0]
            # k，
            if len(idx) < k:
                P_i[j, idx] = sim_i[j, idx]
            else:
                # k
                idx_k = np.argsort(-sim_i[j, idx])[:k]
                P_i[j, idx[idx_k]] = sim_i[j, idx[idx_k]]
        
        # 
        P_i = P_i / P_i.sum(axis=1, keepdims=True)
        P.append(P_i)
    
    # 
    W = similarity_matrices.copy()
    
    # 
    for _ in range(t):
        W_next = []
        
        for i in range(n_views):
            # 
            S = np.zeros((n_samples, n_samples))
            for j in range(n_views):
                if j != i:
                    S += W[j]
            
            S = S / (n_views - 1)
            
            # 
            W_i_next = P[i] @ S @ P[i].T
            W_next.append(W_i_next)
        
        W = W_next
    
    # 
    W_fused = np.zeros((n_samples, n_samples))
    for i in range(n_views):
        W_fused += W[i]
    
    W_fused = W_fused / n_views
    
    return W_fused

def to_sparse_tensor(dense_matrix):
    """Convert a dense matrix to a PyTorch sparse tensor."""
    # COO
    sparse_matrix = sp.coo_matrix(dense_matrix)
    
    # 
    indices = torch.LongTensor([sparse_matrix.row, sparse_matrix.col])
    values = torch.FloatTensor(sparse_matrix.data)
    shape = torch.Size(sparse_matrix.shape)
    
    # 
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    
    return sparse_tensor

def dimension_reduction(data, n_components=2, method='pca'):
    """Reduce matrix dimensionality."""
    if method == 'pca':
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
    else:
        raise ValueError(f"Unsupported reduction method: {method}")
    
    return reduced_data

def evaluate_clustering(embeddings, true_labels, pred_labels=None):
    """Compute clustering quality metrics."""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    
    if pred_labels is None:
        # 
        n_clusters = len(np.unique(true_labels))
        
        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        pred_labels = kmeans.fit_predict(embeddings)
    
    # NMI (Normalized Mutual Information)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    # ARI (Adjusted Rand Index)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    return nmi, ari

def visualize_embeddings(embeddings, labels, title='Embedding Visualization', method='tsne'):
    """Plot 2D embeddings colored by label."""
    import matplotlib.pyplot as plt
    
    # 2，
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
    
    # 
    plt.figure(figsize=(10, 8))
    
    # 
    unique_labels = np.unique(labels)
    
    # 
    for i, label in enumerate(unique_labels):
        # 
        idx = labels == label
        # 
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
