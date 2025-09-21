#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    """
    图卷积层
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 参数初始化
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        # 图卷积操作: AXW
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNEncoder(nn.Module):
    """
    基于GCN的编码器
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, dropout=0.5, activation=F.relu):
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, embedding_dim)
        self.dropout = dropout
        self.activation = activation

    def forward(self, x, adj):
        x = self.activation(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class VGAE(nn.Module):
    """
    变分图自编码器
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, dropout=0.5):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc_mu = GraphConvolution(hidden_dim, embedding_dim)
        self.gc_logvar = GraphConvolution(hidden_dim, embedding_dim)
        self.dropout = dropout
        
    def encode(self, x, adj):
        hidden = F.relu(self.gc1(x, adj))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        mu = self.gc_mu(hidden, adj)
        logvar = self.gc_logvar(hidden, adj)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class GraphSageLayer(nn.Module):
    """
    GraphSAGE层实现
    """
    def __init__(self, input_dim, output_dim, aggregator_type='mean'):
        super(GraphSageLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator_type = aggregator_type
        
        # 定义线性变换
        self.weight = nn.Linear(input_dim * 2, output_dim, bias=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.weight)
        
    def forward(self, features, adj):
        # 邻居特征聚合
        if self.aggregator_type == 'mean':
            # 均值聚合器
            neighbor_feature = torch.spmm(adj, features)
        elif self.aggregator_type == 'max':
            # 最大值聚合器 (简化版本)
            neighbor_feature = torch.spmm(adj, features)
        else:
            raise ValueError(f"不支持的聚合器类型: {self.aggregator_type}")
        
        # 拼接自身特征和邻居特征
        combined = torch.cat([features, neighbor_feature], dim=1)
        
        # 通过线性层和激活函数
        combined = self.weight(combined)
        combined = F.normalize(combined, p=2, dim=1)
        
        return combined

class GraphSage(nn.Module):
    """
    GraphSAGE模型
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, aggregator_type='mean'):
        super(GraphSage, self).__init__()
        self.layer1 = GraphSageLayer(input_dim, hidden_dim, aggregator_type)
        self.layer2 = GraphSageLayer(hidden_dim, output_dim, aggregator_type)
        self.dropout = dropout
        
    def forward(self, x, adj):
        h = self.layer1(x, adj)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.layer2(h, adj)
        return h

class GAT_Layer(nn.Module):
    """
    图注意力网络层
    """
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GAT_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        N = Wh.size()[0]
        
        # 计算注意力系数
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # 屏蔽没有连接的边
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # 归一化注意力系数
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    """
    多头图注意力网络
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, alpha=0.2, n_heads=8):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # 第一层使用多头注意力
        self.attentions = nn.ModuleList([
            GAT_Layer(input_dim, hidden_dim, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(n_heads)
        ])
        
        # 输出层
        self.out_att = GAT_Layer(hidden_dim * n_heads, output_dim, dropout=dropout, alpha=alpha, concat=False)
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x

class SNF_GNN(nn.Module):
    """
    结合SNF和图神经网络的模型
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, gnn_type='gcn', dropout=0.5):
        super(SNF_GNN, self).__init__()
        self.gnn_type = gnn_type
        
        # 选择GNN类型
        if gnn_type == 'gcn':
            self.gnn = GCNEncoder(input_dim, hidden_dim, embedding_dim, dropout)
        elif gnn_type == 'graphsage':
            self.gnn = GraphSage(input_dim, hidden_dim, embedding_dim, dropout)
        elif gnn_type == 'gat':
            self.gnn = GAT(input_dim, hidden_dim, embedding_dim, dropout)
        elif gnn_type == 'vgae':
            self.gnn = VGAE(input_dim, hidden_dim, embedding_dim, dropout)
        else:
            raise ValueError(f"不支持的GNN类型: {gnn_type}")
        
    def forward(self, x, adj):
        if self.gnn_type == 'vgae':
            z, mu, logvar = self.gnn(x, adj)
            return z, mu, logvar
        else:
            return self.gnn(x, adj)

def train_gnn(model, x, adj, optimizer, epochs=200, verbose=True):
    """
    训练GNN模型的通用函数
    
    参数:
        model: GNN模型
        x: 输入特征
        adj: 邻接矩阵
        optimizer: 优化器
        epochs: 训练轮数
        verbose: 是否打印进度
        
    返回:
        训练好的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = x.to(device)
    adj = adj.to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        if isinstance(model, VGAE):
            z, mu, logvar = model(x, adj)
            # 重构邻接矩阵
            adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
            # 变分自编码器损失
            loss = loss_function_vgae(adj_pred, adj, mu, logvar)
        else:
            # 其他GNN模型使用自定义损失
            z = model(x, adj)
            # 这里可以根据任务定义不同的损失函数
            # 例如，对于节点分类可以使用交叉熵损失
            # 对于链接预测可以使用BCE损失
            # 这里使用一个简单的重构损失作为示例
            adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
            loss = F.binary_cross_entropy(adj_pred.view(-1), adj.view(-1))
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

def loss_function_vgae(adj_pred, adj_true, mu, logvar):
    """
    VGAE的损失函数：重构损失 + KL散度
    """
    # 重构损失
    pos_weight = torch.tensor([(adj_true.shape[0] * adj_true.shape[0] - adj_true.sum()) / adj_true.sum()])
    norm = adj_true.shape[0] * adj_true.shape[0] / float((adj_true.shape[0] * adj_true.shape[0] - adj_true.sum()) * 2)
    
    reconstruction_loss = norm * F.binary_cross_entropy_with_logits(
        adj_pred.view(-1), 
        adj_true.view(-1), 
        pos_weight=pos_weight
    )
    
    # KL散度
    kl_divergence = -0.5 / adj_true.size(0) * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    return reconstruction_loss + kl_divergence 