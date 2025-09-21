#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    图卷积层实现
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
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SNFEmbedding(nn.Module):
    """
    SNF嵌入模型
    
    使用融合后的相似度网络学习低维嵌入表示
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, dropout=0.5):
        super(SNFEmbedding, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, embedding_dim)
        self.dropout = dropout
        self.decoder = nn.Linear(embedding_dim, input_dim)
        
    def encode(self, x, adj):
        """编码过程：输入数据通过GCN层得到嵌入"""
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
    def decode(self, z):
        """解码过程：从嵌入重建原始特征"""
        return self.decoder(z)
    
    def forward(self, x, adj):
        """前向传播"""
        # 编码
        z = self.encode(x, adj)
        
        # 解码
        x_reconstructed = self.decode(z)
        
        return z, x_reconstructed


class SNFCluster(nn.Module):
    """
    SNF聚类模型
    
    在SNF嵌入的基础上添加聚类层
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, n_clusters, dropout=0.5):
        super(SNFCluster, self).__init__()
        # 嵌入组件
        self.embedding = SNFEmbedding(input_dim, hidden_dim, embedding_dim, dropout)
        
        # 聚类层
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        
    def encode(self, x, adj):
        """获取嵌入"""
        return self.embedding.encode(x, adj)
    
    def forward(self, x, adj):
        """前向传播"""
        # 获取嵌入
        z = self.encode(x, adj)
        
        # 计算嵌入到各聚类中心的软分配
        q = self._soft_assign(z)
        
        # 辅助目标：重建
        x_reconstructed = self.embedding.decode(z)
        
        return z, q, x_reconstructed
    
    def _soft_assign(self, z):
        """计算软分配（聚类概率）"""
        # 计算样本到聚类中心的距离
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2))
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        return q


class MultiviewSNF(nn.Module):
    """
    多视图SNF模型
    
    处理多个数据视图并进行融合
    """
    def __init__(self, input_dims, hidden_dim, embedding_dim, n_clusters=None, dropout=0.5):
        super(MultiviewSNF, self).__init__()
        self.n_views = len(input_dims)
        
        # 为每个视图创建编码器
        self.view_encoders = nn.ModuleList([
            SNFEmbedding(input_dims[i], hidden_dim, embedding_dim, dropout)
            for i in range(self.n_views)
        ])
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * self.n_views, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 如果指定了聚类数，则添加聚类层
        if n_clusters is not None:
            self.cluster_layer = Parameter(torch.Tensor(n_clusters, embedding_dim))
            torch.nn.init.xavier_normal_(self.cluster_layer.data)
        else:
            self.register_parameter('cluster_layer', None)
    
    def encode_views(self, x_list, adj_list):
        """编码每个视图"""
        z_list = []
        for i in range(self.n_views):
            z = self.view_encoders[i].encode(x_list[i], adj_list[i])
            z_list.append(z)
        return z_list
    
    def fuse_embeddings(self, z_list):
        """融合多视图嵌入"""
        # 拼接多视图嵌入
        z_concat = torch.cat(z_list, dim=1)
        
        # 通过融合层
        z_fused = self.fusion_layer(z_concat)
        
        return z_fused
    
    def forward(self, x_list, adj_list):
        """前向传播"""
        # 编码每个视图
        z_list = self.encode_views(x_list, adj_list)
        
        # 融合嵌入
        z_fused = self.fuse_embeddings(z_list)
        
        # 如果有聚类层，计算聚类分配
        if self.cluster_layer is not None:
            # 计算嵌入到各聚类中心的软分配
            q = 1.0 / (1.0 + torch.sum(
                torch.pow(z_fused.unsqueeze(1) - self.cluster_layer, 2), 2))
            q = q.pow((1 + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            
            return z_list, z_fused, q
        
        return z_list, z_fused, None


def loss_function(x, x_recon, z, q=None, p=None, adj=None, pos_weight=None):
    """
    损失函数
    
    包括重建损失和可选的聚类损失
    
    参数:
        x: 原始特征
        x_recon: 重建特征
        z: 嵌入向量
        q: 聚类软分配
        p: 目标分布
        adj: 邻接矩阵
        pos_weight: 正样本权重
    """
    # 重建损失
    recon_loss = F.mse_loss(x_recon, x)
    
    # 如果有聚类分配和目标分布，添加KL散度作为聚类损失
    if q is not None and p is not None:
        kl_loss = torch.sum(p * torch.log(p / q))
        
        # 总损失 = 重建损失 + 聚类损失
        return recon_loss + kl_loss
    
    return recon_loss


def target_distribution(q):
    """
    计算目标分布以进行聚类
    
    参数:
        q: 初始软分配
        
    返回:
        p: 目标分布
    """
    weight = q ** 2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()
    return p 