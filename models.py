import torch.nn as nn
import numpy as np
from torchvision import models
import torch

class Actor(nn.Module):
    def __init__(self, latent_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logstd = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x):
        h = self.net(x)
        std = self.logstd.exp()
        return self.mu(h), std


class Critic(nn.Module):
    def __init__(self, latent_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class ResNetEmbedder(nn.Module):
    def __init__(self, output_dim=128, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        self.dim_adjuster = nn.Linear(2048, output_dim)
        self.output_dim = output_dim

    def forward(self, images):
        features = self.backbone(images)
        features = features.squeeze()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = self.dim_adjuster(features)
        return features

class CNNAutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,512,4,2,1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.encoder2latent = nn.Linear(512*16*16+1, latent_dim)
        self.latent2decoder = nn.Linear(latent_dim+1, 512*16*16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,2,1), nn.Sigmoid()
        )

    def encode(self, x, y):
        h = self.encoder(x)
        flat = h.view(x.size(0), -1)
        xy = torch.cat([flat, y], dim=1)
        z = self.encoder2latent(xy)
        return z

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        flat = self.latent2decoder(zy)
        h = flat.view(z.size(0),512,16,16)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x, y):
        z = self.encode(x, y)
        return self.decode(z, y)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self,x):
        L = x.size(1)
        return x + self.pe[:,:L,:]

class ViTransformerDetector(nn.Module):
    def __init__(self, embedder, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedder = embedder
        self.embedding_dim = embedder.output_dim
        self.projection = nn.Linear(self.embedding_dim,d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model,128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,1), nn.Sigmoid()
        )

    def forward(self,x):
        emb = self.embedder(x).view(x.size(0),1,-1)
        emb = self.projection(emb)
        emb = self.positional_encoding(emb)
        h = self.transformer_encoder(emb)
        pooled = h.mean(dim=1)
        return self.fc(pooled).squeeze(1)

