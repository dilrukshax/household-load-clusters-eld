"""
model_gan/model.py

Defines the neural network components for the GAN:
- Generator
- Discriminator
- QNetwork (for cluster prediction)
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=10, code_dim=5, data_dim=24, hidden_dim=64):
        """
        Generator network: G(noise, code) -> fake data sample.
        noise_dim: dimension of random noise vector z.
        code_dim: number of cluster categories (one-hot length).
        data_dim: dimensionality of output data (24 features).
        hidden_dim: hidden layer size.
        """
        super(Generator, self).__init__()
        self.input_dim = noise_dim + code_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, data_dim)
        self.activation = nn.ReLU()
        # We use Sigmoid at the output since data is normalized [0,1].
        self.output_act = nn.Sigmoid()
    def forward(self, noise, code):
        """
        noise: tensor of shape (batch_size, noise_dim)
        code: tensor of shape (batch_size, code_dim) (one-hot encoding of cluster)
        Returns: fake_data of shape (batch_size, data_dim)
        """
        x = torch.cat([noise, code], dim=1)  # concatenate noise and one-hot code
        x = self.activation(self.fc1(x))
        x = self.output_act(self.fc2(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, data_dim=24, hidden_dim=64):
        """
        Discriminator: D(x) -> probability that x is real (not generated).
        """
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        # We'll apply Sigmoid at the output to get a probability in (0,1).
        self.output_act = nn.Sigmoid()
    def forward(self, x):
        """
        x: tensor of shape (batch_size, data_dim)
        Returns: probability (batch_size, 1)
        """
        x = self.activation(self.fc1(x))
        x = self.output_act(self.fc2(x))
        return x

class QNetwork(nn.Module):
    def __init__(self, data_dim=24, hidden_dim=64, code_dim=5):
        """
        Q-network: Q(x) -> logits for cluster codes.
        This network tries to predict the latent cluster code from the data.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, code_dim)
        self.activation = nn.ReLU()
        # No softmax here; we'll use CrossEntropyLoss which applies softmax to logits.
    def forward(self, x):
        """
        x: tensor of shape (batch_size, data_dim)
        Returns: logits of shape (batch_size, code_dim)
        """
        x = self.activation(self.fc1(x))
        logits = self.fc2(x)
        return logits
