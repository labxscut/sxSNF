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
    SNF
    """
    def __init__(self, input_dim, hidden_dims, embedding_dim, dropout=0.5, activation='relu'):
        """
        SNF
        
        :
            input_dim: 
            hidden_dims: ， [128, 64]
            embedding_dim: 
            dropout: Dropout
            activation: ， 'relu', 'elu', 'leaky_relu'
        """
        super(SNFEmbedding, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # 
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f": {activation}")
        
        # 
        dims = [input_dim] + hidden_dims + [embedding_dim]
        encoder_layers = []
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # dropout
                encoder_layers.append(self.activation)
                encoder_layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        
        """
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        
        
        :
            x: ， (batch_size, input_dim)
            
        :
            embeddings: ， (batch_size, embedding_dim)
        """
        embeddings = self.encoder(x)
        return embeddings
    
    def get_embeddings(self, x):
        """
        
        
        :
            x: ， (batch_size, input_dim)
            
        :
            embeddings: ， (batch_size, embedding_dim)
        """
        self.eval()  # 
        with torch.no_grad():
            embeddings = self.forward(x)
        return embeddings

class MultiViewSNFEmbedding(nn.Module):
    """
    SNF
    """
    def __init__(self, input_dims, hidden_dims, embedding_dim, dropout=0.5, activation='relu', fusion_type='concat'):
        """
        SNF
        
        :
            input_dims: 
            hidden_dims: ， [128, 64]
            embedding_dim: 
            dropout: Dropout
            activation: 
            fusion_type: ， 'concat', 'attention', 'mean'
        """
        super(MultiViewSNFEmbedding, self).__init__()
        
        self.num_views = len(input_dims)
        self.fusion_type = fusion_type
        self.embedding_dim = embedding_dim
        
        # 
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
        
        # ，
        if fusion_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # ，
        if fusion_type == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(embedding_dim * self.num_views, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
    
    def forward(self, inputs):
        """
        
        
        :
            inputs: ， (batch_size, input_dim_i)
            
        :
            fused_embedding: ， (batch_size, embedding_dim)
        """
        # 
        assert len(inputs) == self.num_views, f" {len(inputs)}  {self.num_views} "
        
        # 
        view_embeddings = []
        for i in range(self.num_views):
            view_embeddings.append(self.view_encoders[i](inputs[i]))
        
        # 
        if self.fusion_type == 'concat':
            # 
            concat_embeddings = torch.cat(view_embeddings, dim=1)
            fused_embedding = self.fusion_layer(concat_embeddings)
        
        elif self.fusion_type == 'attention':
            # 
            attentions = []
            for emb in view_embeddings:
                att = self.attention(emb)
                attentions.append(att)
            
            # 
            attention_weights = F.softmax(torch.cat(attentions, dim=1), dim=1)
            
            # 
            fused_embedding = torch.zeros_like(view_embeddings[0])
            for i, emb in enumerate(view_embeddings):
                fused_embedding += emb * attention_weights[:, i].unsqueeze(1)
        
        elif self.fusion_type == 'mean':
            # 
            fused_embedding = torch.stack(view_embeddings).mean(dim=0)
        
        else:
            raise ValueError(f": {self.fusion_type}")
        
        return fused_embedding
    
    def get_view_embeddings(self, inputs):
        """
        
        
        :
            inputs: 
            
        :
            view_embeddings: 
        """
        self.eval()  # 
        with torch.no_grad():
            view_embeddings = []
            for i in range(self.num_views):
                view_embeddings.append(self.view_encoders[i](inputs[i]))
            
            return view_embeddings
    
    def get_fused_embedding(self, inputs):
        """
        
        
        :
            inputs: 
            
        :
            fused_embedding: 
        """
        self.eval()  # 
        with torch.no_grad():
            fused_embedding = self.forward(inputs)
            
            return fused_embedding

class SNFLoss(nn.Module):
    """
    SNF
    """
    def __init__(self, temperature=0.5, lambda_reg=0.1):
        """
        SNF
        
        :
            temperature: ，
            lambda_reg: 
        """
        super(SNFLoss, self).__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg
    
    def forward(self, embeddings, adj_matrix):
        """
        SNF
        
        :
            embeddings: ， (batch_size, embedding_dim)
            adj_matrix: ， (batch_size, batch_size)
            
        :
            loss: 
        """
        batch_size = embeddings.shape[0]
        
        # 
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # softmax
        exp_sim = torch.exp(sim_matrix)
        
        # 
        mask = torch.eye(batch_size, device=embeddings.device)
        exp_sim = exp_sim * (1 - mask)
        
        # 
        exp_sum = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # 
        p_matrix = exp_sim / (exp_sum + 1e-10)
        
        # 
        loss = -torch.sum(adj_matrix * torch.log(p_matrix + 1e-10)) / batch_size
        
        # 
        reg = torch.norm(embeddings, p=2) * self.lambda_reg
        
        return loss + reg

class MultiViewSNFLoss(nn.Module):
    """
    SNF
    """
    def __init__(self, temperature=0.5, lambda_reg=0.1, lambda_consistency=0.5):
        """
        SNF
        
        :
            temperature: 
            lambda_reg: 
            lambda_consistency: 
        """
        super(MultiViewSNFLoss, self).__init__()
        self.snf_loss = SNFLoss(temperature, lambda_reg)
        self.lambda_consistency = lambda_consistency
    
    def forward(self, view_embeddings, fused_embedding, adj_matrices, fused_adj_matrix):
        """
        SNF
        
        :
            view_embeddings: 
            fused_embedding: 
            adj_matrices: 
            fused_adj_matrix: 
            
        :
            loss: 
        """
        # SNF
        fused_loss = self.snf_loss(fused_embedding, fused_adj_matrix)
        
        # SNF
        view_losses = 0
        for i, (emb, adj) in enumerate(zip(view_embeddings, adj_matrices)):
            view_losses += self.snf_loss(emb, adj)
        
        view_losses /= len(view_embeddings)
        
        # 
        consistency_loss = 0
        for emb in view_embeddings:
            consistency_loss += F.mse_loss(emb, fused_embedding)
        
        consistency_loss /= len(view_embeddings)
        
        # 
        total_loss = fused_loss + view_losses + self.lambda_consistency * consistency_loss
        
        return total_loss

class AutoEncoder(nn.Module):
    """
    
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.1):
        """
        
        
        :
            input_dim: 
            hidden_dims: 
            latent_dim: 
            dropout: Dropout
        """
        super(AutoEncoder, self).__init__()
        
        # 
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 
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
        
        
        :
            x: 
            
        :
            x_hat: 
            z: 
        """
        # 
        z = self.encoder(x)
        # 
        x_hat = self.decoder(z)
        
        return x_hat, z
    
    def encode(self, x):
        """
        
        
        :
            x: 
            
        :
            z: 
        """
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
        return z

class ContrastiveLoss(nn.Module):
    """
    
    """
    def __init__(self, temperature=0.5):
        """
        
        
        :
            temperature: ，
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features):
        """
        
        
        :
            features:  [batch_size, feature_dim]
            
        :
            loss: 
        """
        batch_size = features.shape[0]
        
        # 
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # （）
        mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        
        # ，
        # ，
        positives = similarity_matrix[mask].view(batch_size, 1)
        negatives = similarity_matrix[~mask].view(batch_size, -1)
        
        # softmax
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss 

class SNFEncoder(nn.Module):
    """
    SNF：，
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(SNFEncoder, self).__init__()
        
        # 
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:  # dropout
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        :
            x:  (batch_size, input_dim)
        
        :
             (batch_size, output_dim)
        """
        return self.encoder(x)

class MultiViewSNFModel(nn.Module):
    """
    SNF: ，
    """
    def __init__(self, view_dims, hidden_dims, output_dim, fusion='concat', dropout=0.1):
        """
        :
            view_dims: 
            hidden_dims: 
            output_dim: 
            fusion:  ('concat', 'sum', 'mean')
            dropout: dropout
        """
        super(MultiViewSNFModel, self).__init__()
        
        self.fusion = fusion
        self.n_views = len(view_dims)
        
        # 
        self.encoders = nn.ModuleList()
        for view_dim in view_dims:
            self.encoders.append(SNFEncoder(view_dim, hidden_dims, output_dim, dropout))
        
        # ，
        if fusion == 'concat':
            self.fusion_layer = nn.Linear(output_dim * self.n_views, output_dim)
    
    def forward(self, views):
        """
        :
            views: ，(batch_size, view_dim)
        
        :
             (batch_size, output_dim)
        """
        # 
        view_embeddings = [encoder(view) for encoder, view in zip(self.encoders, views)]
        
        # 
        if self.fusion == 'concat':
            # 
            combined = torch.cat(view_embeddings, dim=1)
            # 
            return self.fusion_layer(combined)
        
        elif self.fusion == 'sum':
            # 
            return torch.sum(torch.stack(view_embeddings), dim=0)
        
        elif self.fusion == 'mean':
            # 
            return torch.mean(torch.stack(view_embeddings), dim=0)
        
        else:
            raise ValueError(f": {self.fusion}")

class SingleViewSNFModel(nn.Module):
    """
    SNF: 
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        """
        :
            input_dim: 
            hidden_dims: 
            output_dim: 
            dropout: dropout
        """
        super(SingleViewSNFModel, self).__init__()
        
        self.encoder = SNFEncoder(input_dim, hidden_dims, output_dim, dropout)
    
    def forward(self, x):
        """
        :
            x:  (batch_size, input_dim)
        
        :
             (batch_size, output_dim)
        """
        return self.encoder(x)

class ReconstructionLoss(nn.Module):
    """
    : 
    """
    def __init__(self, loss_type='mse'):
        """
        :
            loss_type:  ('mse', 'mae')
        """
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, reconstructed, original):
        """
        :
            reconstructed: 
            original: 
        
        :
            
        """
        if self.loss_type == 'mse':
            return F.mse_loss(reconstructed, original)
        elif self.loss_type == 'mae':
            return F.l1_loss(reconstructed, original)
        else:
            raise ValueError(f": {self.loss_type}")

class SNFAutoEncoder(nn.Module):
    """
    SNF: SNF
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.1):
        """
        :
            input_dim: 
            hidden_dims: 
            latent_dim: 
            dropout: dropout
        """
        super(SNFAutoEncoder, self).__init__()
        
        # 
        encoder_layers = []
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        
        for i in range(len(encoder_dims)-1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            if i < len(encoder_dims)-2:  # dropout
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # （）
        decoder_layers = []
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        
        for i in range(len(decoder_dims)-1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims)-2:  # dropout
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout))
            else:
                # Sigmoid，[0,1]
                decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """
        :
            x:  (batch_size, input_dim)
        
        :
             (batch_size, input_dim)
             (batch_size, latent_dim)
        """
        # 
        latent = self.encoder(x)
        # 
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent 