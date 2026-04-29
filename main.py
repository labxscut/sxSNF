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

# 
from utils import normalize_data, construct_similarity_graph, snf_fusion, dimension_reduction
from graph_module import SNF_GNN, train_gnn

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='')
    
    # 
    parser.add_argument('--data_dir', type=str, default='./data', help='Input data directory')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory for outputs')
    
    # SNF
    parser.add_argument('--k', type=int, default=20, help='KNN')
    parser.add_argument('--t', type=int, default=20, help='SNF')
    
    # 
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'graphsage', 'gat', 'vgae'], help='GNN')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    
    # 
    parser.add_argument('--n_clusters', type=int, default=0, help='，0')
    
    # 
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID, -1CPU')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()

def setup_environment(args):
    """Set random seeds and runtime device settings."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    
    # 
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    return device

def load_data(args):
    """Load input data and labels from disk."""
    if args.verbose:
        print("...")
    
    # ：.mat
    # 
    data_files = [f for f in os.listdir(args.data_dir) if f.endswith('.mat')]
    data_list = []
    
    for file in data_files:
        file_path = os.path.join(args.data_dir, file)
        try:
            # MAT
            mat_data = loadmat(file_path)
            
            # 'X'，
            if 'X' in mat_data:
                data_matrix = mat_data['X']
                data_list.append(data_matrix)
            else:
                # 
                for key, value in mat_data.items():
                    if isinstance(value, np.ndarray) and len(value.shape) == 2 and min(value.shape) > 1:
                        if not key.startswith('__'):  # mat
                            data_list.append(value)
                            if args.verbose:
                                print(f"{file} {key}: {value.shape}")
                            break
        except Exception as e:
            print(f" {file}: {e}")
    
    # 
    label_file = os.path.join(args.data_dir, 'labels.txt')
    labels = None
    
    if os.path.exists(label_file):
        try:
            labels = np.loadtxt(label_file, dtype=int)
            if args.verbose:
                print(f": {labels.shape}")
        except:
            print(f" {label_file}")
    
    if not data_list:
        raise ValueError("。。")
    
    return data_list, labels

def snf_process(data_list, args):
    """Construct per-view similarity graphs and run SNF fusion."""
    if args.verbose:
        print("...")
    
    # # 
    # processed_data = []
    # for data in data_list:
    #     # 
    #     norm_data = normalize_data(data, method='minmax')
    #     processed_data.append(norm_data)
    
    for data in data_list:
        # 
        # norm_data = normalize_data(data, method='minmax')
        processed_data.append(data)
    
    # 
    similarity_matrices = []
    for data in processed_data:
        sim_matrix = construct_similarity_graph(data, k=args.k)
        similarity_matrices.append(sim_matrix)
    
    # 
    fused_network = snf_fusion(similarity_matrices, t=args.t, k=args.k)
    
    return fused_network, similarity_matrices

def gnn_embedding(fused_network, args, device):
    """Train a graph model and return node embeddings."""
    if args.verbose:
        print(f"{args.gnn_type.upper()}...")
    
    # （）
    n = fused_network.shape[0]
    features = torch.eye(n).to(device)
    
    # 
    adj = torch.FloatTensor(fused_network).to(device)
    
    # GNN
    model = SNF_GNN(
        input_dim=n,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        gnn_type=args.gnn_type,
        dropout=args.dropout
    )
    
    # 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 
    model = train_gnn(model, features, adj, optimizer, epochs=args.epochs, verbose=args.verbose)
    
    # 
    model.eval()
    with torch.no_grad():
        if args.gnn_type == 'vgae':
            embeddings, _, _ = model(features, adj)
        else:
            embeddings = model(features, adj)
        embeddings = embeddings.cpu().numpy()
    
    return embeddings

def perform_clustering(embeddings, labels, args):
    """Cluster embeddings and compute evaluation metrics."""
    if args.verbose:
        print("...")
    
    # 
    n_clusters = args.n_clusters
    if n_clusters <= 0 and labels is not None:
        n_clusters = len(np.unique(labels))
    elif n_clusters <= 0:
        # ，
        n_clusters = 10
        
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # （）
    metrics = {}
    if labels is not None:
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        ari = adjusted_rand_score(labels, cluster_labels)
        
        metrics = {
            'NMI': nmi,
            'ARI': ari
        }
        
        if args.verbose:
            print(f" - NMI: {nmi:.4f}, ARI: {ari:.4f}")
    
    return cluster_labels, metrics

def visualize_results(embeddings, labels, cluster_labels, args):
    """Create visualization figures for embeddings and clustering."""
    if args.verbose:
        print("...")
    
    # t-SNE2D
    if embeddings.shape[1] > 2:
        tsne_embed = dimension_reduction(embeddings, n_components=2, method='tsne')
    else:
        tsne_embed = embeddings
    
    # 
    plt.figure(figsize=(12, 5))
    
    # 
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=cluster_labels, cmap='viridis', s=20)
    plt.colorbar(scatter)
    plt.title('')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # ，
    if labels is not None:
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=labels, cmap='viridis', s=20)
        plt.colorbar(scatter)
        plt.title('')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
    
    # 
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'clustering_visualization.png'), dpi=300)
    
    # 
    plt.figure(figsize=(10, 8))
    sns.heatmap(embeddings[:min(100, embeddings.shape[0])], cmap='viridis')
    plt.title(' (100)')
    plt.savefig(os.path.join(args.save_dir, 'embedding_heatmap.png'), dpi=300)
    
    plt.close('all')

def save_results(embeddings, cluster_labels, metrics, args):
    """Save embeddings, labels, and metrics to output files."""
    if args.verbose:
        print("...")
    
    # 
    np.save(os.path.join(args.save_dir, 'embeddings.npy'), embeddings)
    
    # 
    np.save(os.path.join(args.save_dir, 'cluster_labels.npy'), cluster_labels)
    
    # MAT（MATLAB）
    savemat(os.path.join(args.save_dir, 'results.mat'), {
        'embeddings': embeddings,
        'cluster_labels': cluster_labels
    })
    
    # 
    if metrics:
        with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
                
    if args.verbose:
        print(f" {args.save_dir}")

def main():
    """Run the end-to-end pipeline."""
    # 
    args = parse_args()
    
    # 
    device = setup_environment(args)
    
    # 
    data_list, labels = load_data(args)
    
    # SNF
    fused_network, similarity_matrices = snf_process(data_list, args)
    
    # GNN
    embeddings = gnn_embedding(fused_network, args, device)
    
    # 
    cluster_labels, metrics = perform_clustering(embeddings, labels, args)
    
    # 
    visualize_results(embeddings, labels, cluster_labels, args)
    
    # 
    save_results(embeddings, cluster_labels, metrics, args)
    
    if args.verbose:
        print("sxSNF！")

if __name__ == "__main__":
    main() 
