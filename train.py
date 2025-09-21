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
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='SNF嵌入模型训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--dataset', type=str, default='toy', help='数据集名称')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='single', choices=['single', 'multi'], help='模型类型：单视图或多视图')
    parser.add_argument('--hidden_dims', type=str, default='128,64', help='隐藏层维度，用逗号分隔')
    parser.add_argument('--embedding_dim', type=int, default=32, help='嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'elu', 'leaky_relu'], help='激活函数')
    parser.add_argument('--fusion_type', type=str, default='concat', choices=['concat', 'attention', 'mean'], help='融合类型')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--temperature', type=float, default=0.5, help='损失函数温度参数')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='正则化系数')
    parser.add_argument('--lambda_consistency', type=float, default=0.5, help='一致性损失权重')
    
    # 图构建参数
    parser.add_argument('--k', type=int, default=20, help='KNN中的K值')
    parser.add_argument('--sigma', type=float, default=0.5, help='相似度计算中的sigma参数')
    parser.add_argument('--t', type=int, default=20, help='SNF迭代次数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID，-1表示使用CPU')
    parser.add_argument('--log_interval', type=int, default=10, help='日志输出间隔(每N个批次)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    
    return parser.parse_args()


def set_seed(seed):
    """
    设置随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    """
    加载数据
    
    参数:
        args: 命令行参数
        
    返回:
        data_dict: 包含数据和标签的字典
    """
    # 示例：加载多视图数据
    # 在实际应用中，需要根据具体数据格式进行修改
    if args.dataset == 'toy':
        # 创建玩具数据集
        n_samples = 500
        n_features = [20, 30, 25]  # 三个视图的特征维度
        
        if args.model == 'single':
            # 单视图数据
            X = np.random.randn(n_samples, n_features[0])
            y = np.random.randint(0, 5, size=n_samples)  # 5个类别
            
            data_dict = {
                'X': X,
                'y': y
            }
        else:
            # 多视图数据
            X1 = np.random.randn(n_samples, n_features[0])
            X2 = np.random.randn(n_samples, n_features[1])
            X3 = np.random.randn(n_samples, n_features[2])
            y = np.random.randint(0, 5, size=n_samples)  # 5个类别
            
            data_dict = {
                'X1': X1,
                'X2': X2,
                'X3': X3,
                'y': y
            }
    else:
        # 加载真实数据集
        # 示例: data_dir/dataset/view1.csv, data_dir/dataset/view2.csv, ...
        data_path = os.path.join(args.data_dir, args.dataset)
        
        if args.model == 'single':
            # 单视图数据
            X = np.loadtxt(os.path.join(data_path, 'view1.csv'), delimiter=',')
            y = np.loadtxt(os.path.join(data_path, 'labels.csv'), delimiter=',')
            
            data_dict = {
                'X': X,
                'y': y
            }
        else:
            # 多视图数据
            # 查找所有视图文件
            view_files = [f for f in os.listdir(data_path) if f.startswith('view') and f.endswith('.csv')]
            
            data_dict = {}
            for i, file in enumerate(sorted(view_files)):
                data_dict[f'X{i+1}'] = np.loadtxt(os.path.join(data_path, file), delimiter=',')
            
            # 加载标签
            data_dict['y'] = np.loadtxt(os.path.join(data_path, 'labels.csv'), delimiter=',')
    
    return data_dict


def preprocess_data(data_dict, args):
    """
    数据预处理
    
    参数:
        data_dict: 包含数据和标签的字典
        args: 命令行参数
        
    返回:
        processed_data: 预处理后的数据
    """
    processed_data = {}
    
    # 数据标准化
    if args.model == 'single':
        processed_data['X'] = normalize_data(data_dict['X'], method='z-score')
    else:
        for key in data_dict:
            if key.startswith('X'):
                processed_data[key] = normalize_data(data_dict[key], method='z-score')
    
    # 保留原始标签
    processed_data['y'] = data_dict['y']
    
    # 构建相似度图
    if args.model == 'single':
        sim_graph = construct_similarity_graph(
            processed_data['X'], 
            k=args.k, 
            metric='euclidean', 
            sigma=args.sigma
        )
        processed_data['sim_graph'] = sim_graph
    else:
        # 构建每个视图的相似度图
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
        
        # 融合相似度图
        fused_graph = snf_fusion(sim_graphs, k=args.k, t=args.t)
        processed_data['fused_graph'] = fused_graph
    
    return processed_data


def split_data(processed_data, args):
    """
    数据集划分
    
    参数:
        processed_data: 预处理后的数据
        args: 命令行参数
        
    返回:
        data_loaders: 包含训练、验证和测试数据加载器的字典
    """
    # 提取标签
    y = processed_data['y']
    
    # 划分数据集
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
    
    # 准备数据加载器
    if args.model == 'single':
        # 单视图数据
        X = processed_data['X']
        sim_graph = processed_data['sim_graph']
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        sim_graph_tensor = torch.FloatTensor(sim_graph)
        
        # 创建数据集和数据加载器
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
        # 多视图数据
        view_data = []
        for key in processed_data:
            if key.startswith('X'):
                view_data.append(processed_data[key])
        
        sim_graphs = processed_data['sim_graphs']
        fused_graph = processed_data['fused_graph']
        
        # 转换为PyTorch张量
        view_tensors = [torch.FloatTensor(data) for data in view_data]
        y_tensor = torch.LongTensor(y)
        sim_graph_tensors = [torch.FloatTensor(graph) for graph in sim_graphs]
        fused_graph_tensor = torch.FloatTensor(fused_graph)
        
        # 创建训练集
        train_views = [tensor[train_idx] for tensor in view_tensors]
        train_sim_graphs = [graph[train_idx][:, train_idx] for graph in sim_graph_tensors]
        train_fused_graph = fused_graph_tensor[train_idx][:, train_idx]
        train_y = y_tensor[train_idx]
        
        # 创建验证集
        val_views = [tensor[val_idx] for tensor in view_tensors]
        val_sim_graphs = [graph[val_idx][:, val_idx] for graph in sim_graph_tensors]
        val_fused_graph = fused_graph_tensor[val_idx][:, val_idx]
        val_y = y_tensor[val_idx]
        
        # 创建测试集
        test_views = [tensor[test_idx] for tensor in view_tensors]
        test_sim_graphs = [graph[test_idx][:, test_idx] for graph in sim_graph_tensors]
        test_fused_graph = fused_graph_tensor[test_idx][:, test_idx]
        test_y = y_tensor[test_idx]
        
        # 创建自定义数据集
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
    
    # 创建数据加载器
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
    """
    单视图模型训练
    
    参数:
        model: 模型
        optimizer: 优化器
        criterion: 损失函数
        data_loader: 数据加载器
        device: 设备(CPU/GPU)
        
    返回:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (X, sim_graph, y) in enumerate(data_loader):
        X = X.to(device)
        sim_graph = sim_graph.to(device)
        
        # 前向传播
        embeddings = model(X)
        
        # 计算损失
        loss = criterion(embeddings, sim_graph)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def train_multi_view(model, optimizer, criterion, data_loader, device):
    """
    多视图模型训练
    
    参数:
        model: 模型
        optimizer: 优化器
        criterion: 损失函数
        data_loader: 数据加载器
        device: 设备(CPU/GPU)
        
    返回:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (views, sim_graphs, fused_graph, y) in enumerate(data_loader):
        # 将数据移动到设备
        views = [view.to(device) for view in views]
        sim_graphs = [graph.to(device) for graph in sim_graphs]
        fused_graph = fused_graph.to(device)
        
        # 前向传播
        view_embeddings = model.get_view_embeddings(views)
        fused_embedding = model(views)
        
        # 计算损失
        loss = criterion(view_embeddings, fused_embedding, sim_graphs, fused_graph)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate(model, criterion, data_loader, device, is_multi_view=False):
    """
    模型验证
    
    参数:
        model: 模型
        criterion: 损失函数
        data_loader: 数据加载器
        device: 设备(CPU/GPU)
        is_multi_view: 是否为多视图模型
        
    返回:
        avg_loss: 平均损失
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        if not is_multi_view:
            # 单视图验证
            for batch_idx, (X, sim_graph, y) in enumerate(data_loader):
                X = X.to(device)
                sim_graph = sim_graph.to(device)
                
                # 前向传播
                embeddings = model(X)
                
                # 计算损失
                loss = criterion(embeddings, sim_graph)
                
                total_loss += loss.item()
        else:
            # 多视图验证
            for batch_idx, (views, sim_graphs, fused_graph, y) in enumerate(data_loader):
                # 将数据移动到设备
                views = [view.to(device) for view in views]
                sim_graphs = [graph.to(device) for graph in sim_graphs]
                fused_graph = fused_graph.to(device)
                
                # 前向传播
                view_embeddings = model.get_view_embeddings(views)
                fused_embedding = model(views)
                
                # 计算损失
                loss = criterion(view_embeddings, fused_embedding, sim_graphs, fused_graph)
                
                total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def test(model, data_loader, device, is_multi_view=False, n_clusters=5):
    """
    模型测试
    
    参数:
        model: 模型
        data_loader: 数据加载器
        device: 设备(CPU/GPU)
        is_multi_view: 是否为多视图模型
        n_clusters: 聚类数量
        
    返回:
        results: 评估结果字典
    """
    from sklearn.cluster import KMeans
    
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        if not is_multi_view:
            # 单视图测试
            for batch_idx, (X, sim_graph, y) in enumerate(data_loader):
                X = X.to(device)
                
                # 获取嵌入
                embeddings = model(X)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(y.numpy())
        else:
            # 多视图测试
            for batch_idx, (views, sim_graphs, fused_graph, y) in enumerate(data_loader):
                # 将数据移动到设备
                views = [view.to(device) for view in views]
                
                # 获取融合嵌入
                fused_embedding = model(views)
                
                all_embeddings.append(fused_embedding.cpu().numpy())
                all_labels.append(y.numpy())
    
    # 合并所有批次的结果
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(embeddings)
    
    # 计算评估指标
    nmi = normalized_mutual_info_score(labels, pred_labels)
    ari = adjusted_rand_score(labels, pred_labels)
    
    results = {
        'nmi': nmi,
        'ari': ari
    }
    
    return results


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f'使用设备: {device}')
    
    # 加载数据
    print('加载数据...')
    data_dict = load_data(args)
    
    # 数据预处理
    print('数据预处理...')
    processed_data = preprocess_data(data_dict, args)
    
    # 划分数据集
    print('划分数据集...')
    data_loaders = split_data(processed_data, args)
    
    # 创建模型目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 解析隐藏层维度
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # 创建模型
    print('创建模型...')
    if args.model == 'single':
        # 单视图模型
        input_dim = processed_data['X'].shape[1]
        model = SNFEmbedding(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            activation=args.activation
        )
        # 损失函数
        criterion = SNFLoss(
            temperature=args.temperature,
            lambda_reg=args.lambda_reg
        )
    else:
        # 多视图模型
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
        # 损失函数
        criterion = MultiViewSNFLoss(
            temperature=args.temperature,
            lambda_reg=args.lambda_reg,
            lambda_consistency=args.lambda_consistency
        )
    
    # 将模型移动到设备
    model = model.to(device)
    criterion = criterion.to(device)
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 训练模型
    print('开始训练...')
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # 训练
        if args.model == 'single':
            train_loss = train_single_view(model, optimizer, criterion, data_loaders['train'], device)
            val_loss = validate(model, criterion, data_loaders['val'], device, is_multi_view=False)
        else:
            train_loss = train_multi_view(model, optimizer, criterion, data_loaders['train'], device)
            val_loss = validate(model, criterion, data_loaders['val'], device, is_multi_view=True)
        
        # 计算epoch花费时间
        epoch_time = time.time() - start_time
        
        # 输出训练信息
        if epoch % args.log_interval == 0:
            print(f'Epoch {epoch}/{args.epochs} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | 用时: {epoch_time:.2f}s')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= args.patience:
            print(f'早停: {args.patience} 个epoch内验证损失未改善')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    
    # 测试模型
    print('测试模型...')
    # 获取类别数量
    n_clusters = len(np.unique(processed_data['y']))
    
    test_results = test(
        model, 
        data_loaders['test'], 
        device, 
        is_multi_view=(args.model == 'multi'),
        n_clusters=n_clusters
    )
    
    print(f'测试结果: NMI: {test_results["nmi"]:.4f} | ARI: {test_results["ari"]:.4f}')
    
    # 保存结果
    results_file = os.path.join(args.save_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f'Model: {args.model}\n')
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'Embedding dim: {args.embedding_dim}\n')
        f.write(f'NMI: {test_results["nmi"]:.4f}\n')
        f.write(f'ARI: {test_results["ari"]:.4f}\n')
    
    print(f'结果已保存至 {results_file}')


if __name__ == '__main__':
    main() 