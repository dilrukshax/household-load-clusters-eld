"""
model_gan/train.py
"""
import json, os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess
from model import Generator, Discriminator, QNetwork

# ------------------ config ------------------
SEED = 7
torch.manual_seed(SEED); np.random.seed(SEED)
NUM_EPOCHS   = 800
BATCH_SIZE   = 64
LR           = 2e-4
NOISE_DIM    = 10
CODE_DIM     = 5      # number of clusters you want
HIDDEN_DIM   = 64
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY  = 100
SAVE_PREFIX  = "gan"
# --------------------------------------------

X = load_and_preprocess()                 # (N, 24)
N, D = X.shape
X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)

dataset  = torch.utils.data.TensorDataset(X_tensor)
loader   = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

G = Generator(NOISE_DIM, CODE_DIM, D, HIDDEN_DIM).to(DEVICE)
Dnet = Discriminator(D, HIDDEN_DIM).to(DEVICE)
Q = QNetwork(HIDDEN_DIM, CODE_DIM).to(DEVICE)

optD = optim.Adam(Dnet.parameters(), lr=LR)
optGQ = optim.Adam(list(G.parameters()) + list(Q.parameters()), lr=LR)

bce_logits = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()

for epoch in range(1, NUM_EPOCHS+1):
    for (xb,) in loader:
        bs = xb.size(0)

        # ---- Train D ----
        Dnet.zero_grad()
        h_real, d_real = Dnet(xb, return_features=True)
        loss_real = bce_logits(d_real, torch.ones(bs,1, device=DEVICE))

        z = torch.randn(bs, NOISE_DIM, device=DEVICE)
        c_idx = torch.randint(0, CODE_DIM, (bs,), device=DEVICE)
        c_onehot = torch.zeros(bs, CODE_DIM, device=DEVICE); c_onehot[torch.arange(bs), c_idx] = 1.
        x_fake = G(z, c_onehot)
        _, d_fake = Dnet(x_fake.detach(), return_features=True)
        loss_fake = bce_logits(d_fake, torch.zeros(bs,1, device=DEVICE))

        (loss_real + loss_fake).backward(); optD.step()

        # ---- Train G + Q ----
        G.zero_grad(); Q.zero_grad()
        h_fake, d_fake2 = Dnet(x_fake, return_features=True)
        g_loss = bce_logits(d_fake2, torch.ones(bs,1, device=DEVICE))
        q_logits = Q(h_fake)
        q_loss = xent(q_logits, c_idx)
        (g_loss + q_loss).backward(); optGQ.step()

    if epoch % PRINT_EVERY == 0 or epoch in (1, NUM_EPOCHS):
        print(f"Epoch {epoch}/{NUM_EPOCHS}  D: {(loss_real+loss_fake).item():.4f}  G: {g_loss.item():.4f}  Q: {q_loss.item():.4f}")

# ---- Inference: cluster assignments on real data ----
with torch.no_grad():
    h, _ = Dnet(X_tensor, return_features=True)
    q_logits = Q(h)
    probs = torch.softmax(q_logits, dim=1).cpu().numpy()
labels = probs.argmax(axis=1)

sil = float(silhouette_score(X, labels))
db  = float(davies_bouldin_score(X, labels))
print(f"Silhouette={sil:.4f}  DBI={db:.4f}")

# ---- Save results ----
np.save("clusters.npy", labels)
with open("results.json","w") as f:
    json.dump({"model":"GAN","clusters":CODE_DIM,"silhouette":sil,"dbi":db}, f, indent=2)

# ---- Plot ----
X2 = TSNE(n_components=2, init='pca', random_state=SEED).fit_transform(X)
plt.figure(figsize=(6,5))
for k in range(CODE_DIM):
    pts = X2[labels==k]
    plt.scatter(pts[:,0], pts[:,1], label=f"Cluster {k}", alpha=0.8)
plt.title("GAN-based Clustering (t-SNE)")
plt.legend(); plt.tight_layout(); plt.savefig("tsne_gan.png"); plt.close()
print("Saved: clusters.npy, results.json, tsne_gan.png")
