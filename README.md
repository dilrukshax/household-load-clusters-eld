# Brain Tumor MRI Classification – Assignment Package

This package contains **four Jupyter notebooks** that train different supervised models on the *Brain Tumor MRI Dataset – Merged* (≈13k images, 4 classes: `glioma`, `meningioma`, `pituitary`, `no_tumor`). 
Models included: **CNN**, **RNN**, **LSTM**, **GNN**.

> Designed for beginners: clear structure, heavy comments, and identical preprocessing across models to enable fair comparison.

## Folder Structure
```
brain_tumor_project/
├─ notebooks/
│  ├─ cnn_brain_tumor_classification.ipynb
│  ├─ rnn_brain_tumor_classification.ipynb
│  ├─ lstm_brain_tumor_classification.ipynb
│  └─ gnn_brain_tumor_classification.ipynb
├─ data/               # put dataset folders here (glioma/, meningioma/, pituitary/, no_tumor/)
├─ README.md
├─ report_template.docx
└─ requirements.txt
```

## 1) Setup a Python environment
Recommended: Python 3.10+ with a fresh virtual environment.

**Option A – pip**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B – Conda**
```bash
conda create -n btumor python=3.10 -y
conda activate btumor
pip install -r requirements.txt
```

> If you have an NVIDIA GPU, install CUDA-enabled builds of TensorFlow or PyTorch according to their official instructions.

## 2) Download & Place the Dataset
Download the **Brain Tumor MRI Dataset – Merged (4 classes)** from Kaggle (or your chosen source).
Organize it like this inside `data/`:

```
data/
├─ glioma/
├─ meningioma/
├─ pituitary/
└─ no_tumor/
```

> Each subfolder should contain the corresponding class images (JPG/PNG).

## 3) Run the notebooks
Launch Jupyter:
```bash
jupyter notebook
```
Open each notebook in `notebooks/`:

- **cnn_brain_tumor_classification.ipynb** (recommended to start)
- **rnn_brain_tumor_classification.ipynb**
- **lstm_brain_tumor_classification.ipynb**
- **gnn_brain_tumor_classification.ipynb** (advanced; requires PyTorch + PyTorch Geometric)

Each notebook will:
- Load/visualize the dataset (shared preprocessing, 64×64 grayscale, normalization)
- Train its model (with basic augmentation for the image-based models)
- Evaluate with Accuracy, F1-score, **Confusion Matrix**, and **Learning Curves**

## 4) Notes
- Default epochs are kept modest so you can get results quickly; increase if you have time/GPU.
- If you get out-of-memory (OOM) errors, try reducing `batch_size` or image size.
- **GNN** requires `torch` and **PyTorch Geometric**. In case of install issues, run the CNN/RNN/LSTM first.

## 5) Report
Use **report_template.docx** as a scaffold for your final report: fill in figures (CM, curves), metric tables, and your discussion.
