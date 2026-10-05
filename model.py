#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    
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
    SNF
    
    
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, dropout=0.5):
        super(SNFEmbedding, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, embedding_dim)
        self.dropout = dropout
        self.decoder = nn.Linear(embedding_dim, input_dim)
        
    def encode(self, x, adj):
        """：GCN"""
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
    def decode(self, z):
        """："""
        return self.decoder(z)
    
    def forward(self, x, adj):
        """"""
        # 
        z = self.encode(x, adj)
        
        # 
        x_reconstructed = self.decode(z)
        
        return z, x_reconstructed


class SNFCluster(nn.Module):
    """
    SNF
    
    SNF
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, n_clusters, dropout=0.5):
        super(SNFCluster, self).__init__()
        # 
        self.embedding = SNFEmbedding(input_dim, hidden_dim, embedding_dim, dropout)
        
        # 
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        
    def encode(self, x, adj):
        """"""
        return self.embedding.encode(x, adj)
    
    def forward(self, x, adj):
        """"""
        # 
        z = self.encode(x, adj)
        
        # 
        q = self._soft_assign(z)
        
        # ：
        x_reconstructed = self.embedding.decode(z)
        
        return z, q, x_reconstructed
    
    def _soft_assign(self, z):
        """（）"""
        # 
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2))
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        return q


class MultiviewSNF(nn.Module):
    """
    SNF
    
    
    """
    def __init__(self, input_dims, hidden_dim, embedding_dim, n_clusters=None, dropout=0.5):
        super(MultiviewSNF, self).__init__()
        self.n_views = len(input_dims)
        
        # 
        self.view_encoders = nn.ModuleList([
            SNFEmbedding(input_dims[i], hidden_dim, embedding_dim, dropout)
            for i in range(self.n_views)
        ])
        
        # 
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * self.n_views, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ，
        if n_clusters is not None:
            self.cluster_layer = Parameter(torch.Tensor(n_clusters, embedding_dim))
            torch.nn.init.xavier_normal_(self.cluster_layer.data)
        else:
            self.register_parameter('cluster_layer', None)
    
    def encode_views(self, x_list, adj_list):
        """"""
        z_list = []
        for i in range(self.n_views):
            z = self.view_encoders[i].encode(x_list[i], adj_list[i])
            z_list.append(z)
        return z_list
    
    def fuse_embeddings(self, z_list):
        """"""
        # 
        z_concat = torch.cat(z_list, dim=1)
        
        # 
        z_fused = self.fusion_layer(z_concat)
        
        return z_fused
    
    def forward(self, x_list, adj_list):
        """"""
        # 
        z_list = self.encode_views(x_list, adj_list)
        
        # 
        z_fused = self.fuse_embeddings(z_list)
        
        # ，
        if self.cluster_layer is not None:
            # 
            q = 1.0 / (1.0 + torch.sum(
                torch.pow(z_fused.unsqueeze(1) - self.cluster_layer, 2), 2))
            q = q.pow((1 + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            
            return z_list, z_fused, q
        
        return z_list, z_fused, None


def loss_function(x, x_recon, z, q=None, p=None, adj=None, pos_weight=None):
    """
    
    
    
    
    :
        x: 
        x_recon: 
        z: 
        q: 
        p: 
        adj: 
        pos_weight: 
    """
    # 
    recon_loss = F.mse_loss(x_recon, x)
    
    # ，KL
    if q is not None and p is not None:
        kl_loss = torch.sum(p * torch.log(p / q))
        
        #  =  + 
        return recon_loss + kl_loss
    
    return recon_loss


def target_distribution(q):
    """
    
    
    :
        q: 
        
    :
        p: 
    """
    weight = q ** 2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()
    return p 