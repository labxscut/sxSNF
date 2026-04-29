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

# sxSNF
sys.path.append('sxSNF/code')

# sxSNF
from utils import normalize_data, construct_similarity_graph, snf_fusion, dimension_reduction
from graph_module import SNF_GNN, train_gnn

def generate_simulated_data(n_samples=500, n_features1=1000, n_features2=800, n_clusters=3, random_seed=42):
    """Generate synthetic multi-omics style data."""
    # 
    np.random.seed(random_seed)
    
    # 
    centers1 = np.random.randn(n_clusters, n_features1) * 5
    centers2 = np.random.randn(n_clusters, n_features2) * 5
    
    # 
    data1 = np.zeros((n_samples, n_features1))
    data2 = np.zeros((n_samples, n_features2))
    labels = np.zeros(n_samples, dtype=int)
    
    # 
    samples_per_cluster = n_samples // n_clusters
    for i in range(n_clusters):
        start_idx = i * samples_per_cluster
        end_idx = (i + 1) * samples_per_cluster if i < n_clusters - 1 else n_samples
        
        # : 
        data1[start_idx:end_idx] = centers1[i] + np.random.randn(end_idx - start_idx, n_features1) * 1.5
        
        # : 
        data2[start_idx:end_idx] = centers2[i] + np.random.randn(end_idx - start_idx, n_features2) * 1.5
        
        # 
        labels[start_idx:end_idx] = i
    
    # （）
    # : 
    sparsity1 = 0.7  # 70%0
    mask1 = np.random.random(data1.shape) < sparsity1
    data1[mask1] = 0
    
    # : 
    sparsity2 = 0.5  # 50%0
    mask2 = np.random.random(data2.shape) < sparsity2
    data2[mask2] = 0
    
    # 0（）
    data1 = np.maximum(data1, 0)
    data2 = np.maximum(data2, 0)
    
    print(f" - 1: {data1.shape}, 2: {data2.shape}")
    print(f": {np.bincount(labels)}")
    
    # 
    if not os.path.exists("simulated_data"):
        os.makedirs("simulated_data")
    
    np.save("simulated_data/modality1_data.npy", data1)
    np.save("simulated_data/modality2_data.npy", data2)
    np.save("simulated_data/labels.npy", labels)
    
    print(f" simulated_data ")
    
    return [data1, data2], labels

def visualize_data(data_list, labels, file_name="simulated_data_visualization.png"):
    """Visualize sample matrices and label distribution."""
    plt.figure(figsize=(15, 6))
    
    # 
    for i, data in enumerate(data_list):
        plt.subplot(1, 3, i+1)
        sns.heatmap(data[:50, :50], cmap="viridis")
        plt.title(f"Modality {i+1} Data Sample (First 50 rows and columns)")
    
    # 
    plt.subplot(1, 3, 3)
    sns.countplot(x=labels)
    plt.title("Label Distribution")
    
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def visualize_loss_curve(loss_file="training_process/loss_history.npy"):
    """Plot and save the training loss curve."""
    if not os.path.exists(loss_file):
        print(f" {loss_file} ")
        return
    
    losses = np.load(loss_file)
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # 
    plt.grid(True)
    plt.savefig("training_loss_curve.png", dpi=300)
    plt.close()
    print(" training_loss_curve.png")

def run_sxSNF(data_list, labels, args, save_results=True):
    """Execute the sxSNF workflow on input data."""
    print("sxSNF...")
    
    # 
    processed_data = []
    for data in data_list:
        # 
        norm_data = normalize_data(data, method='min-max')
        processed_data.append(norm_data)
    
    # 
    print("...")
    similarity_matrices = []
    for data in processed_data:
        sim_matrix = construct_similarity_graph(data, k=args.k)
        similarity_matrices.append(sim_matrix)
    
    # 
    print(f"SNF (k={args.k}, t={args.t})...")
    fused_network = snf_fusion(similarity_matrices, k=args.k, t=args.t)
    
    # CUDA:0
    device = torch.device('cuda:0')
    print(f": {device}")
    
    # CUDA
    if not torch.cuda.is_available():
        print(": CUDA，CPU")
        device = torch.device('cpu')
    
    # GNN
    n = fused_network.shape[0]
    features = torch.eye(n).to(device)
    adj = torch.FloatTensor(fused_network).to(device)
    
    # graph_module.pyGraphConvolution
    # 
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
            # 
            input = input.to(self.weight.device)
            adj = adj.to(self.weight.device)
            
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output
    
    # GCNEncoder
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
    
    # SNF_GNN
    class FixedSNF_GNN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, embedding_dim, dropout=0.5, device=device):
            super(FixedSNF_GNN, self).__init__()
            self.gnn = FixedGCNEncoder(input_dim, hidden_dim, embedding_dim, dropout, device=device)
            
        def forward(self, x, adj):
            return self.gnn(x, adj)
    
    # 
    print(" GCN ...")
    model = FixedSNF_GNN(
        input_dim=n,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        device=device
    ).to(device)
    
    # 
    for param in model.parameters():
        param.data = param.data.to(device)
    
    # 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 
    def simple_train(model, features, adj, optimizer, epochs=100):
        model.train()
        
        # 
        if not os.path.exists("training_process"):
            os.makedirs("training_process")
            
        # 
        with torch.no_grad():
            initial_embeddings = model(features, adj).cpu().numpy()
            np.save("training_process/embeddings_epoch_0.npy", initial_embeddings)
        
        losses = []
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            
            # 
            output = model(features, adj)
            
            # 
            loss = torch.nn.functional.mse_loss(output @ output.t(), adj)
            losses.append(loss.item())
            
            # 
            loss.backward()
            optimizer.step()
            
            # 10epoch
            if epoch % 10 == 0:
                with torch.no_grad():
                    embeddings = model(features, adj).cpu().numpy()
                    np.save(f"training_process/embeddings_epoch_{epoch}.npy", embeddings)
            
            # ，
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.10f}")
        
        # 
        np.save("training_process/loss_history.npy", np.array(losses))
        
        return model
    
    # 
    model = simple_train(model, features, adj, optimizer, epochs=args.epochs)
    
    # 
    model.eval()
    with torch.no_grad():
        embeddings = model(features, adj)
        embeddings = embeddings.cpu().numpy()
    
    # t-SNE
    from sklearn.manifold import TSNE
    if embeddings.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=args.seed)
        tsne_embed = tsne.fit_transform(embeddings)
    else:
        tsne_embed = embeddings
    
    # 
    from sklearn.cluster import KMeans
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)
    print(f" - NMI: {nmi:.4f}, ARI: {ari:.4f}")
    
    # 
    plt.figure(figsize=(12, 5))
    
    # 
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=cluster_labels, cmap='viridis', s=20)
    plt.colorbar(scatter)
    plt.title('Predicted Clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # 
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], c=labels, cmap='viridis', s=20)
    plt.colorbar(scatter)
    plt.title('True Labels')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig("sxSNF_clustering_results.png", dpi=300)
    
    # 
    if save_results:
        if not os.path.exists("results"):
            os.makedirs("results")
        np.save("results/embeddings.npy", embeddings)
        np.save("results/cluster_labels.npy", cluster_labels)
        
        # 
        with open("results/metrics.txt", "w") as f:
            f.write(f"NMI: {nmi:.4f}\n")
            f.write(f"ARI: {ari:.4f}\n")
    
    print("sxSNF！")
    
    # 
    visualize_loss_curve()
    
    return embeddings, cluster_labels, {"NMI": nmi, "ARI": ari}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='sxSNF')
    
    # 
    parser.add_argument('--n_samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--n_features1', type=int, default=100, help='Feature count for modality 1')
    parser.add_argument('--n_features2', type=int, default=80, help='Feature count for modality 2')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters')
    
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID, -1CPU')
    
    return parser.parse_args()

def main():
    """Run the end-to-end pipeline."""
    # 
    args = parse_args()
    
    # 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # # 
    # print("...")
    # data_list, labels = generate_simulated_data(
    #     n_samples=args.n_samples,
    #     n_features1=args.n_features1,
    #     n_features2=args.n_features2,
    #     n_clusters=args.n_clusters,
    #     random_seed=args.seed
    # )
    
    # 
    use_simulated_data = False  # False
    
    if not use_simulated_data:
        print("...")
        try:
            # 
            rna_data = np.load('./datasets/rna_pca.npy')
            atac_data = np.load('./datasets/atac_lsi.npy')
            labels = np.load('./datasets/numeric_labels.npy')
            
            print(f"RNA: {rna_data.shape}")
            print(f"ATAC: {atac_data.shape}")
            print(f": {labels.shape}")
            
            # ，
            data_list = [rna_data, atac_data]
            
            # 
            if rna_data.shape[0] != atac_data.shape[0] or rna_data.shape[0] != labels.shape[0]:
                raise ValueError("，！")
                
            print(f": {len(data_list)}，{rna_data.shape[0]}")
            
        except FileNotFoundError as e:
            print(f":  - {e}")
            print("...")
            use_simulated_data = True
        except Exception as e:
            print(f": {e}")
            print("...")
            use_simulated_data = True
    
    # ，
    if use_simulated_data:
        print("...")
        data_list, labels = generate_simulated_data(
            n_samples=args.n_samples,
            n_features1=args.n_features1,
            n_features2=args.n_features2,
            n_clusters=args.n_clusters,
            random_seed=args.seed
        )
    
    # 
    visualize_data(data_list, labels)
    
    # sxSNF
    embeddings, cluster_labels, metrics = run_sxSNF(data_list, labels, args)
    
    print("！")
    print(f" - NMI: {metrics['NMI']:.4f}, ARI: {metrics['ARI']:.4f}")

if __name__ == "__main__":
    main() 
