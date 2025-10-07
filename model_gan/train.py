"""
model_gan/train.py

Trains the GAN model and performs clustering evaluation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Import our modules
from preprocess import load_and_preprocess
from model import Generator, Discriminator, QNetwork

# Hyperparameters
NUM_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
NOISE_DIM = 10
CODE_DIM = 5    # number of clusters to discover
HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_INTERVAL = 100  # print losses every 100 epochs

# Load and prepare data
X = load_and_preprocess()  # shape (N, 24)
N, data_dim = X.shape[0], X.shape[1]

# Convert data to torch tensor
X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

# We will use a DataLoader for real data batches (useful if N is large; here N=370, small but still).
dataset = torch.utils.data.TensorDataset(X_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Initialize models
G = Generator(noise_dim=NOISE_DIM, code_dim=CODE_DIM, data_dim=data_dim, hidden_dim=HIDDEN_DIM).to(DEVICE)
D = Discriminator(data_dim=data_dim, hidden_dim=HIDDEN_DIM).to(DEVICE)
Q = QNetwork(data_dim=data_dim, hidden_dim=HIDDEN_DIM, code_dim=CODE_DIM).to(DEVICE)

# Optimizers
optim_D = optim.Adam(D.parameters(), lr=LEARNING_RATE)
# Note: we combine G and Q parameters for optimizing generator step, because generator's loss includes Q's prediction.
# However, in InfoGAN, typically Q and G are optimized together for the mutual info loss.
optim_G = optim.Adam(list(G.parameters()) + list(Q.parameters()), lr=LEARNING_RATE)

# Loss functions
criterion_gan = nn.BCELoss()          # binary cross-entropy for discriminator/generator
criterion_info = nn.CrossEntropyLoss()  # for Q's cluster prediction (discrete code)

# Labels
real_label = 1.0
fake_label = 0.0

# Training loop
for epoch in range(1, NUM_EPOCHS+1):
    for real_batch, in dataloader:
        # real_batch is a batch of real data points
        batch_size = real_batch.size(0)
        #######################
        # 1. Train Discriminator
        #######################
        D.zero_grad()
        # Real data loss
        real_data = real_batch.to(DEVICE)  # shape (batch, data_dim)
        output_real = D(real_data).view(-1)          # D outputs (batch,1) -> flatten to (batch,)
        labels_real = torch.full((batch_size,), real_label, device=DEVICE)
        loss_real = criterion_gan(output_real, labels_real)
        # Fake data loss
        # Sample random latent vectors
        noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
        # Sample random cluster codes (one-hot)
        code_indices = torch.randint(0, CODE_DIM, (batch_size,), device=DEVICE)
        code_onehot = torch.zeros(batch_size, CODE_DIM, device=DEVICE)
        code_onehot[torch.arange(batch_size), code_indices] = 1.0
        fake_data = G(noise, code_onehot)
        output_fake = D(fake_data.detach()).view(-1)  # detach generator so only D is updated in this block
        labels_fake = torch.full((batch_size,), fake_label, device=DEVICE)
        loss_fake = criterion_gan(output_fake, labels_fake)
        # Combine D loss and update
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optim_D.step()

        #######################
        # 2. Train Generator and Q (together)
        #######################
        G.zero_grad()
        Q.zero_grad()
        # Generate another batch of fakes (or reuse the ones above, but let's resample for diversity)
        noise2 = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
        code_indices2 = torch.randint(0, CODE_DIM, (batch_size,), device=DEVICE)
        code_onehot2 = torch.zeros(batch_size, CODE_DIM, device=DEVICE)
        code_onehot2[torch.arange(batch_size), code_indices2] = 1.0
        fake_data2 = G(noise2, code_onehot2)
        # Generator adversarial loss (we want D to think these fakes are real)
        output_fake2 = D(fake_data2).view(-1)
        # Label as real for generator loss
        labels_gen = torch.full((batch_size,), real_label, device=DEVICE)
        loss_G_adv = criterion_gan(output_fake2, labels_gen)
        # Info loss: make Q correctly predict the code used
        q_logits = Q(fake_data2)            # shape (batch, CODE_DIM)
        loss_G_info = criterion_info(q_logits, code_indices2)
        # Combined generator loss
        loss_G_total = loss_G_adv + loss_G_info
        loss_G_total.backward()
        optim_G.step()
    # End of epoch, optionally print losses
    if epoch % PRINT_INTERVAL == 0 or epoch == 1 or epoch == NUM_EPOCHS:
        print(f"Epoch {epoch}/{NUM_EPOCHS}  Loss_D: {loss_D.item():.4f}  Loss_G_adv: {loss_G_adv.item():.4f}  Loss_G_info: {loss_G_info.item():.4f}")

# Training finished
print("Training complete.")

# Clustering using Q-network on real data
with torch.no_grad():
    q_logits_real = Q(X_tensor.to(DEVICE))        # shape (N, CODE_DIM)
    q_probs = torch.softmax(q_logits_real, dim=1) # probabilities for each cluster
    cluster_assignments = q_probs.argmax(dim=1).cpu().numpy()  # predicted cluster for each data point

# Evaluate clustering metrics
sil_score = silhouette_score(X, cluster_assignments)
db_score = davies_bouldin_score(X, cluster_assignments)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")

# Visualization: t-SNE scatter plot of clusters
tsne = TSNE(n_components=2, init='pca', random_state=0)
X_2d = tsne.fit_transform(X)  # project original data to 2D for visualization
plt.figure(figsize=(6,5))
for c in range(CODE_DIM):
    pts = X_2d[cluster_assignments == c]
    plt.scatter(pts[:,0], pts[:,1], label=f"Cluster {c}", alpha=0.7)
plt.title("GAN-based Clustering (t-SNE visualization)")
plt.legend()
plt.tight_layout()
plt.savefig("tsne_clusters_gan.png")
plt.close()
print("Cluster plot saved to tsne_clusters_gan.png")
