#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from utils import normalize_data, construct_similarity_graph, snf_fusion
from models import SNFEmbedding, MultiViewSNFEmbedding, SNFLoss, MultiViewSNFLoss


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='SNF')
    
    # 
    parser.add_argument('--data_dir', type=str, default='./data', help='Input data directory')
    parser.add_argument('--dataset', type=str, default='toy', help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation split ratio')
    
    # 
    parser.add_argument('--model', type=str, default='single', choices=['single', 'multi'], help='：')
    parser.add_argument('--hidden_dims', type=str, default='128,64', help='，')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'elu', 'leaky_relu'], help='Activation function')
    parser.add_argument('--fusion_type', type=str, default='concat', choices=['concat', 'attention', 'mean'], help='Multi-view fusion type')
    
    # 
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--temperature', type=float, default=0.5, help='Contrastive loss temperature')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='Regularization loss weight')
    parser.add_argument('--lambda_consistency', type=float, default=0.5, help='Consistency loss weight')
    
    # 
    parser.add_argument('--k', type=int, default=20, help='KNNK')
    parser.add_argument('--sigma', type=float, default=0.5, help='sigma')
    parser.add_argument('--t', type=int, default=20, help='SNF')
    
    # 
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID，-1CPU')
    parser.add_argument('--log_interval', type=int, default=10, help='(N)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint output directory')
    
    return parser.parse_args()


def set_seed(seed):
    """Set seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    """Load input data and labels from disk."""
    # ：
    # ，
    if args.dataset == 'toy':
        # 
        n_samples = 500
        n_features = [20, 30, 25]  # 
        
        if args.model == 'single':
            # 
            X = np.random.randn(n_samples, n_features[0])
            y = np.random.randint(0, 5, size=n_samples)  # 5
            
            data_dict = {
                'X': X,
                'y': y
            }
        else:
            # 
            X1 = np.random.randn(n_samples, n_features[0])
            X2 = np.random.randn(n_samples, n_features[1])
            X3 = np.random.randn(n_samples, n_features[2])
            y = np.random.randint(0, 5, size=n_samples)  # 5
            
            data_dict = {
                'X1': X1,
                'X2': X2,
                'X3': X3,
                'y': y
            }
    else:
        # 
        # : data_dir/dataset/view1.csv, data_dir/dataset/view2.csv, ...
        data_path = os.path.join(args.data_dir, args.dataset)
        
        if args.model == 'single':
            # 
            X = np.loadtxt(os.path.join(data_path, 'view1.csv'), delimiter=',')
            y = np.loadtxt(os.path.join(data_path, 'labels.csv'), delimiter=',')
            
            data_dict = {
                'X': X,
                'y': y
            }
        else:
            # 
            # 
            view_files = [f for f in os.listdir(data_path) if f.startswith('view') and f.endswith('.csv')]
            
            data_dict = {}
            for i, file in enumerate(sorted(view_files)):
                data_dict[f'X{i+1}'] = np.loadtxt(os.path.join(data_path, file), delimiter=',')
            
            # 
            data_dict['y'] = np.loadtxt(os.path.join(data_path, 'labels.csv'), delimiter=',')
    
    return data_dict


def preprocess_data(data_dict, args):
    """Preprocess data."""
    processed_data = {}
    
    # 
    if args.model == 'single':
        processed_data['X'] = normalize_data(data_dict['X'], method='z-score')
    else:
        for key in data_dict:
            if key.startswith('X'):
                processed_data[key] = normalize_data(data_dict[key], method='z-score')
    
    # 
    processed_data['y'] = data_dict['y']
    
    # 
    if args.model == 'single':
        sim_graph = construct_similarity_graph(
            processed_data['X'], 
            k=args.k, 
            metric='euclidean', 
            sigma=args.sigma
        )
        processed_data['sim_graph'] = sim_graph
    else:
        # 
        sim_graphs = []
        for key in processed_data:
            if key.startswith('X'):
                sim_graph = construct_similarity_graph(
                    processed_data[key], 
                    k=args.k, 
                    metric='euclidean', 
                    sigma=args.sigma
                )
                sim_graphs.append(sim_graph)
        
        processed_data['sim_graphs'] = sim_graphs
        
        # 
        fused_graph = snf_fusion(sim_graphs, k=args.k, t=args.t)
        processed_data['fused_graph'] = fused_graph
    
    return processed_data


def split_data(processed_data, args):
    """Split data."""
    # 
    y = processed_data['y']
    
    # 
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), 
        test_size=args.test_size, 
        stratify=y, 
        random_state=args.seed
    )
    
    train_idx, val_idx = train_test_split(
        train_idx, 
        test_size=args.val_size/(1-args.test_size), 
        stratify=y[train_idx], 
        random_state=args.seed
    )
    
    # 
    if args.model == 'single':
        # 
        X = processed_data['X']
        sim_graph = processed_data['sim_graph']
        
        # PyTorch
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        sim_graph_tensor = torch.FloatTensor(sim_graph)
        
        # 
        train_dataset = TensorDataset(
            X_tensor[train_idx], 
            sim_graph_tensor[train_idx][:, train_idx], 
            y_tensor[train_idx]
        )
        val_dataset = TensorDataset(
            X_tensor[val_idx], 
            sim_graph_tensor[val_idx][:, val_idx], 
            y_tensor[val_idx]
        )
        test_dataset = TensorDataset(
            X_tensor[test_idx], 
            sim_graph_tensor[test_idx][:, test_idx], 
            y_tensor[test_idx]
        )
    else:
        # 
        view_data = []
        for key in processed_data:
            if key.startswith('X'):
                view_data.append(processed_data[key])
        
        sim_graphs = processed_data['sim_graphs']
        fused_graph = processed_data['fused_graph']
        
        # PyTorch
        view_tensors = [torch.FloatTensor(data) for data in view_data]
        y_tensor = torch.LongTensor(y)
        sim_graph_tensors = [torch.FloatTensor(graph) for graph in sim_graphs]
        fused_graph_tensor = torch.FloatTensor(fused_graph)
        
        # 
        train_views = [tensor[train_idx] for tensor in view_tensors]
        train_sim_graphs = [graph[train_idx][:, train_idx] for graph in sim_graph_tensors]
        train_fused_graph = fused_graph_tensor[train_idx][:, train_idx]
        train_y = y_tensor[train_idx]
        
        # 
        val_views = [tensor[val_idx] for tensor in view_tensors]
        val_sim_graphs = [graph[val_idx][:, val_idx] for graph in sim_graph_tensors]
        val_fused_graph = fused_graph_tensor[val_idx][:, val_idx]
        val_y = y_tensor[val_idx]
        
        # 
        test_views = [tensor[test_idx] for tensor in view_tensors]
        test_sim_graphs = [graph[test_idx][:, test_idx] for graph in sim_graph_tensors]
        test_fused_graph = fused_graph_tensor[test_idx][:, test_idx]
        test_y = y_tensor[test_idx]
        
        # 
        class MultiViewDataset(torch.utils.data.Dataset):
            def __init__(self, views, sim_graphs, fused_graph, labels):
                self.views = views
                self.sim_graphs = sim_graphs
                self.fused_graph = fused_graph
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                view_data = [view[idx] for view in self.views]
                return view_data, self.sim_graphs, self.fused_graph, self.labels[idx]
        
        train_dataset = MultiViewDataset(train_views, train_sim_graphs, train_fused_graph, train_y)
        val_dataset = MultiViewDataset(val_views, val_sim_graphs, val_fused_graph, val_y)
        test_dataset = MultiViewDataset(test_views, test_sim_graphs, test_fused_graph, test_y)
    
    # 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return data_loaders


def train_single_view(model, optimizer, criterion, data_loader, device):
    """Train single view."""
    model.train()
    total_loss = 0
    
    for batch_idx, (X, sim_graph, y) in enumerate(data_loader):
        X = X.to(device)
        sim_graph = sim_graph.to(device)
        
        # 
        embeddings = model(X)
        
        # 
        loss = criterion(embeddings, sim_graph)
        
        # 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def train_multi_view(model, optimizer, criterion, data_loader, device):
    """Train multi view."""
    model.train()
    total_loss = 0
    
    for batch_idx, (views, sim_graphs, fused_graph, y) in enumerate(data_loader):
        # 
        views = [view.to(device) for view in views]
        sim_graphs = [graph.to(device) for graph in sim_graphs]
        fused_graph = fused_graph.to(device)
        
        # 
        view_embeddings = model.get_view_embeddings(views)
        fused_embedding = model(views)
        
        # 
        loss = criterion(view_embeddings, fused_embedding, sim_graphs, fused_graph)
        
        # 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate(model, criterion, data_loader, device, is_multi_view=False):
    """Validate."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        if not is_multi_view:
            # 
            for batch_idx, (X, sim_graph, y) in enumerate(data_loader):
                X = X.to(device)
                sim_graph = sim_graph.to(device)
                
                # 
                embeddings = model(X)
                
                # 
                loss = criterion(embeddings, sim_graph)
                
                total_loss += loss.item()
        else:
            # 
            for batch_idx, (views, sim_graphs, fused_graph, y) in enumerate(data_loader):
                # 
                views = [view.to(device) for view in views]
                sim_graphs = [graph.to(device) for graph in sim_graphs]
                fused_graph = fused_graph.to(device)
                
                # 
                view_embeddings = model.get_view_embeddings(views)
                fused_embedding = model(views)
                
                # 
                loss = criterion(view_embeddings, fused_embedding, sim_graphs, fused_graph)
                
                total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def test(model, data_loader, device, is_multi_view=False, n_clusters=5):
    """Test."""
    from sklearn.cluster import KMeans
    
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        if not is_multi_view:
            # 
            for batch_idx, (X, sim_graph, y) in enumerate(data_loader):
                X = X.to(device)
                
                # 
                embeddings = model(X)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(y.numpy())
        else:
            # 
            for batch_idx, (views, sim_graphs, fused_graph, y) in enumerate(data_loader):
                # 
                views = [view.to(device) for view in views]
                
                # 
                fused_embedding = model(views)
                
                all_embeddings.append(fused_embedding.cpu().numpy())
                all_labels.append(y.numpy())
    
    # 
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(embeddings)
    
    # 
    nmi = normalized_mutual_info_score(labels, pred_labels)
    ari = adjusted_rand_score(labels, pred_labels)
    
    results = {
        'nmi': nmi,
        'ari': ari
    }
    
    return results


def main():
    """Run the end-to-end pipeline."""
    # 
    args = parse_args()
    
    # 
    set_seed(args.seed)
    
    # 
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f': {device}')
    
    # 
    print('...')
    data_dict = load_data(args)
    
    # 
    print('...')
    processed_data = preprocess_data(data_dict, args)
    
    # 
    print('...')
    data_loaders = split_data(processed_data, args)
    
    # 
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # 
    print('...')
    if args.model == 'single':
        # 
        input_dim = processed_data['X'].shape[1]
        model = SNFEmbedding(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            activation=args.activation
        )
        # 
        criterion = SNFLoss(
            temperature=args.temperature,
            lambda_reg=args.lambda_reg
        )
    else:
        # 
        input_dims = []
        for key in processed_data:
            if key.startswith('X'):
                input_dims.append(processed_data[key].shape[1])
        
        model = MultiViewSNFEmbedding(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            activation=args.activation,
            fusion_type=args.fusion_type
        )
        # 
        criterion = MultiViewSNFLoss(
            temperature=args.temperature,
            lambda_reg=args.lambda_reg,
            lambda_consistency=args.lambda_consistency
        )
    
    # 
    model = model.to(device)
    criterion = criterion.to(device)
    
    # 
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 
    print('...')
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # 
        if args.model == 'single':
            train_loss = train_single_view(model, optimizer, criterion, data_loaders['train'], device)
            val_loss = validate(model, criterion, data_loaders['val'], device, is_multi_view=False)
        else:
            train_loss = train_multi_view(model, optimizer, criterion, data_loaders['train'], device)
            val_loss = validate(model, criterion, data_loaders['val'], device, is_multi_view=True)
        
        # epoch
        epoch_time = time.time() - start_time
        
        # 
        if epoch % args.log_interval == 0:
            print(f'Epoch {epoch}/{args.epochs} | : {train_loss:.4f} | : {val_loss:.4f} | : {epoch_time:.2f}s')
        
        # 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 
        if patience_counter >= args.patience:
            print(f': {args.patience} epoch')
            break
    
    # 
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    
    # 
    print('...')
    # 
    n_clusters = len(np.unique(processed_data['y']))
    
    test_results = test(
        model, 
        data_loaders['test'], 
        device, 
        is_multi_view=(args.model == 'multi'),
        n_clusters=n_clusters
    )
    
    print(f': NMI: {test_results["nmi"]:.4f} | ARI: {test_results["ari"]:.4f}')
    
    # 
    results_file = os.path.join(args.save_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f'Model: {args.model}\n')
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'Embedding dim: {args.embedding_dim}\n')
        f.write(f'NMI: {test_results["nmi"]:.4f}\n')
        f.write(f'ARI: {test_results["ari"]:.4f}\n')
    
    print(f' {results_file}')


if __name__ == '__main__':
    main() 
