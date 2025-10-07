"""
model_gan/model.py
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=10, code_dim=5, data_dim=24, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid()
        )
    def forward(self, z, c_onehot):
        return self.net(torch.cat([z, c_onehot], dim=1))

class Discriminator(nn.Module):
    def __init__(self, data_dim=24, hidden_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim, 1)  # logits
    def forward(self, x, return_features=False):
        h = self.features(x)
        if return_features:
            return h, self.head(h)
        return self.head(h)

class QNetwork(nn.Module):
    def __init__(self, hidden_dim=64, code_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim)  # logits
        )
    def forward(self, h):
        return self.net(h)
