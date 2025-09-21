#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import numpy as np
import scipy.sparse as sp
from torch.nn.parameter import Parameter

class SNFEmbedding(nn.Module):
    """
    单视图SNF嵌入模型
    """
    def __init__(self, input_dim, hidden_dims, embedding_dim, dropout=0.5, activation='relu'):
        """
        初始化单视图SNF嵌入模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，例如 [128, 64]
            embedding_dim: 嵌入空间维度
            dropout: Dropout比率
            activation: 激活函数，可选 'relu', 'elu', 'leaky_relu'
        """
        super(SNFEmbedding, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # 定义激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建编码器
        dims = [input_dim] + hidden_dims + [embedding_dim]
        encoder_layers = []
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 不在最后一层添加激活函数和dropout
                encoder_layers.append(self.activation)
                encoder_layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        重置模型参数
        """
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据，形状为 (batch_size, input_dim)
            
        返回:
            embeddings: 嵌入向量，形状为 (batch_size, embedding_dim)
        """
        embeddings = self.encoder(x)
        return embeddings
    
    def get_embeddings(self, x):
        """
        获取嵌入向量
        
        参数:
            x: 输入数据，形状为 (batch_size, input_dim)
            
        返回:
            embeddings: 嵌入向量，形状为 (batch_size, embedding_dim)
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            embeddings = self.forward(x)
        return embeddings

class MultiViewSNFEmbedding(nn.Module):
    """
    多视图SNF嵌入模型
    """
    def __init__(self, input_dims, hidden_dims, embedding_dim, dropout=0.5, activation='relu', fusion_type='concat'):
        """
        初始化多视图SNF嵌入模型
        
        参数:
            input_dims: 各视图输入特征维度列表
            hidden_dims: 隐藏层维度列表，例如 [128, 64]
            embedding_dim: 最终嵌入空间维度
            dropout: Dropout比率
            activation: 激活函数
            fusion_type: 融合类型，可选 'concat', 'attention', 'mean'
        """
        super(MultiViewSNFEmbedding, self).__init__()
        
        self.num_views = len(input_dims)
        self.fusion_type = fusion_type
        self.embedding_dim = embedding_dim
        
        # 为每个视图创建独立的编码器
        self.view_encoders = nn.ModuleList()
        for i in range(self.num_views):
            self.view_encoders.append(
                SNFEmbedding(
                    input_dim=input_dims[i],
                    hidden_dims=hidden_dims,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    activation=activation
                )
            )
        
        # 如果使用注意力融合，定义注意力层
        if fusion_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # 如果使用拼接融合，定义融合层
        if fusion_type == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(embedding_dim * self.num_views, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
    
    def forward(self, inputs):
        """
        前向传播
        
        参数:
            inputs: 各视图输入数据列表，每个元素形状为 (batch_size, input_dim_i)
            
        返回:
            fused_embedding: 融合后的嵌入向量，形状为 (batch_size, embedding_dim)
        """
        # 确保输入数量与视图数量相匹配
        assert len(inputs) == self.num_views, f"输入数量 {len(inputs)} 与视图数量 {self.num_views} 不匹配"
        
        # 获取每个视图的嵌入
        view_embeddings = []
        for i in range(self.num_views):
            view_embeddings.append(self.view_encoders[i](inputs[i]))
        
        # 根据融合类型进行嵌入融合
        if self.fusion_type == 'concat':
            # 拼接融合
            concat_embeddings = torch.cat(view_embeddings, dim=1)
            fused_embedding = self.fusion_layer(concat_embeddings)
        
        elif self.fusion_type == 'attention':
            # 注意力融合
            attentions = []
            for emb in view_embeddings:
                att = self.attention(emb)
                attentions.append(att)
            
            # 计算注意力权重
            attention_weights = F.softmax(torch.cat(attentions, dim=1), dim=1)
            
            # 加权融合
            fused_embedding = torch.zeros_like(view_embeddings[0])
            for i, emb in enumerate(view_embeddings):
                fused_embedding += emb * attention_weights[:, i].unsqueeze(1)
        
        elif self.fusion_type == 'mean':
            # 平均融合
            fused_embedding = torch.stack(view_embeddings).mean(dim=0)
        
        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")
        
        return fused_embedding
    
    def get_view_embeddings(self, inputs):
        """
        获取各视图的嵌入向量
        
        参数:
            inputs: 各视图输入数据列表
            
        返回:
            view_embeddings: 各视图嵌入向量列表
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            view_embeddings = []
            for i in range(self.num_views):
                view_embeddings.append(self.view_encoders[i](inputs[i]))
            
            return view_embeddings
    
    def get_fused_embedding(self, inputs):
        """
        获取融合嵌入向量
        
        参数:
            inputs: 各视图输入数据列表
            
        返回:
            fused_embedding: 融合后的嵌入向量
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            fused_embedding = self.forward(inputs)
            
            return fused_embedding

class SNFLoss(nn.Module):
    """
    SNF损失函数
    """
    def __init__(self, temperature=0.5, lambda_reg=0.1):
        """
        初始化SNF损失函数
        
        参数:
            temperature: 温度参数，控制相似度分布的平滑程度
            lambda_reg: 正则化系数
        """
        super(SNFLoss, self).__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg
    
    def forward(self, embeddings, adj_matrix):
        """
        计算SNF损失
        
        参数:
            embeddings: 嵌入向量，形状为 (batch_size, embedding_dim)
            adj_matrix: 邻接矩阵，形状为 (batch_size, batch_size)
            
        返回:
            loss: 损失值
        """
        batch_size = embeddings.shape[0]
        
        # 计算嵌入向量之间的相似度
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # 应用softmax获取概率分布
        exp_sim = torch.exp(sim_matrix)
        
        # 忽略自循环
        mask = torch.eye(batch_size, device=embeddings.device)
        exp_sim = exp_sim * (1 - mask)
        
        # 计算归一化常数
        exp_sum = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # 计算概率分布
        p_matrix = exp_sim / (exp_sum + 1e-10)
        
        # 计算交叉熵损失
        loss = -torch.sum(adj_matrix * torch.log(p_matrix + 1e-10)) / batch_size
        
        # 添加正则化项
        reg = torch.norm(embeddings, p=2) * self.lambda_reg
        
        return loss + reg

class MultiViewSNFLoss(nn.Module):
    """
    多视图SNF损失函数
    """
    def __init__(self, temperature=0.5, lambda_reg=0.1, lambda_consistency=0.5):
        """
        初始化多视图SNF损失函数
        
        参数:
            temperature: 温度参数
            lambda_reg: 正则化系数
            lambda_consistency: 一致性损失权重
        """
        super(MultiViewSNFLoss, self).__init__()
        self.snf_loss = SNFLoss(temperature, lambda_reg)
        self.lambda_consistency = lambda_consistency
    
    def forward(self, view_embeddings, fused_embedding, adj_matrices, fused_adj_matrix):
        """
        计算多视图SNF损失
        
        参数:
            view_embeddings: 各视图嵌入向量列表
            fused_embedding: 融合后的嵌入向量
            adj_matrices: 各视图邻接矩阵列表
            fused_adj_matrix: 融合后的邻接矩阵
            
        返回:
            loss: 总损失值
        """
        # 计算融合嵌入的SNF损失
        fused_loss = self.snf_loss(fused_embedding, fused_adj_matrix)
        
        # 计算各视图嵌入的SNF损失
        view_losses = 0
        for i, (emb, adj) in enumerate(zip(view_embeddings, adj_matrices)):
            view_losses += self.snf_loss(emb, adj)
        
        view_losses /= len(view_embeddings)
        
        # 计算一致性损失
        consistency_loss = 0
        for emb in view_embeddings:
            consistency_loss += F.mse_loss(emb, fused_embedding)
        
        consistency_loss /= len(view_embeddings)
        
        # 总损失
        total_loss = fused_loss + view_losses + self.lambda_consistency * consistency_loss
        
        return total_loss

class AutoEncoder(nn.Module):
    """
    用于无监督特征学习的自编码器
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.1):
        """
        初始化自编码器
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在空间维度
            dropout: Dropout比率
        """
        super(AutoEncoder, self).__init__()
        
        # 构建编码器
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 构建解码器
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征
            
        返回:
            x_hat: 重构特征
            z: 潜在表示
        """
        # 编码
        z = self.encoder(x)
        # 解码
        x_hat = self.decoder(z)
        
        return x_hat, z
    
    def encode(self, x):
        """
        编码函数
        
        参数:
            x: 输入特征
            
        返回:
            z: 潜在表示
        """
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
        return z

class ContrastiveLoss(nn.Module):
    """
    对比损失函数
    """
    def __init__(self, temperature=0.5):
        """
        初始化对比损失
        
        参数:
            temperature: 温度参数，控制相似度分布的平滑程度
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features):
        """
        计算对比损失
        
        参数:
            features: 特征向量 [batch_size, feature_dim]
            
        返回:
            loss: 对比损失值
        """
        batch_size = features.shape[0]
        
        # 计算余弦相似度矩阵
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 对角线上是正样本对（自己和自己）
        mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        
        # 对每个样本，计算与其他所有样本的对比损失
        # 正样本对是自己与自己，负样本对是自己与其他所有样本
        positives = similarity_matrix[mask].view(batch_size, 1)
        negatives = similarity_matrix[~mask].view(batch_size, -1)
        
        # 计算softmax损失
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss 

class SNFEncoder(nn.Module):
    """
    SNF编码器：接受相似度矩阵作为输入，输出嵌入
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(SNFEncoder, self).__init__()
        
        # 构建多层网络
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:  # 不在最后一层添加激活函数和dropout
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        参数:
            x: 输入特征矩阵 (batch_size, input_dim)
        
        返回:
            嵌入向量 (batch_size, output_dim)
        """
        return self.encoder(x)

class MultiViewSNFModel(nn.Module):
    """
    多视图SNF模型: 处理多个视图的数据，并融合它们
    """
    def __init__(self, view_dims, hidden_dims, output_dim, fusion='concat', dropout=0.1):
        """
        参数:
            view_dims: 每个视图的输入维度的列表
            hidden_dims: 每个编码器的隐藏层维度的列表
            output_dim: 输出嵌入的维度
            fusion: 融合方法 ('concat', 'sum', 'mean')
            dropout: dropout概率
        """
        super(MultiViewSNFModel, self).__init__()
        
        self.fusion = fusion
        self.n_views = len(view_dims)
        
        # 为每个视图创建编码器
        self.encoders = nn.ModuleList()
        for view_dim in view_dims:
            self.encoders.append(SNFEncoder(view_dim, hidden_dims, output_dim, dropout))
        
        # 如果使用拼接融合，需要额外的融合层
        if fusion == 'concat':
            self.fusion_layer = nn.Linear(output_dim * self.n_views, output_dim)
    
    def forward(self, views):
        """
        参数:
            views: 视图列表，每个视图是形状为(batch_size, view_dim)的张量
        
        返回:
            融合后的嵌入 (batch_size, output_dim)
        """
        # 每个视图输入到对应的编码器
        view_embeddings = [encoder(view) for encoder, view in zip(self.encoders, views)]
        
        # 根据选择的方法融合视图
        if self.fusion == 'concat':
            # 拼接所有视图的嵌入
            combined = torch.cat(view_embeddings, dim=1)
            # 通过融合层降维
            return self.fusion_layer(combined)
        
        elif self.fusion == 'sum':
            # 所有视图的嵌入求和
            return torch.sum(torch.stack(view_embeddings), dim=0)
        
        elif self.fusion == 'mean':
            # 所有视图的嵌入求平均
            return torch.mean(torch.stack(view_embeddings), dim=0)
        
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion}")

class SingleViewSNFModel(nn.Module):
    """
    单视图SNF模型: 处理单一视图的数据
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        """
        参数:
            input_dim: 输入特征的维度
            hidden_dims: 隐藏层维度的列表
            output_dim: 输出嵌入的维度
            dropout: dropout概率
        """
        super(SingleViewSNFModel, self).__init__()
        
        self.encoder = SNFEncoder(input_dim, hidden_dims, output_dim, dropout)
    
    def forward(self, x):
        """
        参数:
            x: 输入特征 (batch_size, input_dim)
        
        返回:
            嵌入向量 (batch_size, output_dim)
        """
        return self.encoder(x)

class ReconstructionLoss(nn.Module):
    """
    重构损失函数: 用于自编码器的训练
    """
    def __init__(self, loss_type='mse'):
        """
        参数:
            loss_type: 损失类型 ('mse', 'mae')
        """
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, reconstructed, original):
        """
        参数:
            reconstructed: 重构的数据
            original: 原始数据
        
        返回:
            损失值
        """
        if self.loss_type == 'mse':
            return F.mse_loss(reconstructed, original)
        elif self.loss_type == 'mae':
            return F.l1_loss(reconstructed, original)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

class SNFAutoEncoder(nn.Module):
    """
    SNF自编码器: 编码和解码SNF相似度矩阵
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.1):
        """
        参数:
            input_dim: 输入特征的维度
            hidden_dims: 编码器和解码器的隐藏层维度列表
            latent_dim: 潜在空间的维度
            dropout: dropout概率
        """
        super(SNFAutoEncoder, self).__init__()
        
        # 构建编码器
        encoder_layers = []
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        
        for i in range(len(encoder_dims)-1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            if i < len(encoder_dims)-2:  # 不在最后一层添加激活函数和dropout
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 构建解码器（反向的维度）
        decoder_layers = []
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        
        for i in range(len(decoder_dims)-1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims)-2:  # 不在最后一层添加激活函数和dropout
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout))
            else:
                # 在最后一层使用Sigmoid激活函数，确保输出在[0,1]范围内
                decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """
        参数:
            x: 输入特征 (batch_size, input_dim)
        
        返回:
            重构的特征 (batch_size, input_dim)
            潜在表示 (batch_size, latent_dim)
        """
        # 编码
        latent = self.encoder(x)
        # 解码
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent 