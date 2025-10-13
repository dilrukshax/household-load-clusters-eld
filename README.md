# Brain Tumor MRI Classification - Deep Learning Comparison

A comprehensive deep learning project comparing **CNN**, **RNN**, **LSTM**, and **GNN** architectures for brain tumor classification from MRI images.

## 📊 Project Overview

This project implements and compares four different deep learning architectures for classifying brain tumors into four categories:
- **Glioma**
- **Meningioma** 
- **Pituitary**
- **No Tumor**

### Models Implemented

| Model | Architecture | Test Accuracy | Key Features |
|-------|-------------|---------------|--------------|
| **CNN** | ResNet-style with skip connections | ~88% | 128×128 input, residual blocks, spatial features |
| **RNN** | 1D-Conv + Bidirectional LSTM | ~88% | Treats rows as sequences, temporal patterns |
| **LSTM** | BiLSTM + TimeDistributed layers | ~87% | Bidirectional processing, layer normalization |
| **GNN** | Graph Convolutional Network | ~85% | SLIC superpixels, graph-based processing |

## 🎯 Project Goals

1. **Compare** different neural network architectures for medical image classification
2. **Document** theoretical foundations and practical implementations
3. **Analyze** strengths and weaknesses of each approach
4. **Provide** reproducible code with comprehensive documentation

## 📁 Project Structure

```
├── data/                          # Dataset directory
│   ├── glioma/                   # Glioma tumor images
│   ├── meningioma/               # Meningioma tumor images
│   ├── pituitary/                # Pituitary tumor images
│   └── notumor/                  # No tumor images
│
├── notebooks/                     # Jupyter notebooks
│   ├── cnn_brain_tumor_classification.ipynb
│   ├── rnn_brain_tumor_classification.ipynb
│   ├── lstm_brain_tumor_classification.ipynb
│   ├── gnn_brain_tumor_classification.ipynb
│   └── compare_all_models.ipynb
│
├── artifacts/                     # Trained models and results
│   ├── cnn/                      # CNN model artifacts
│   ├── rnn/                      # RNN model artifacts
│   ├── lstm/                     # LSTM model artifacts
│   └── gnn/                      # GNN model artifacts
│
├── docs/                          # Detailed documentation
│   ├── CNN_Documentation.md
│   ├── RNN_Documentation.md
│   ├── LSTM_Documentation.md
│   └── GNN_Documentation.md
│
├── results_plots/                 # Comparison visualizations
│   ├── accuracy_f1_comparison.png
│   ├── confusion_matrices_comparison.png
│   └── training_curves_comparison.png
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd household-load-clusters-eld

# Create virtual environment (recommended)
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the Brain Tumor MRI dataset and organize it as:
```
data/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

Each folder should contain the respective MRI images (JPG/PNG format).

### 3. Training Models

Open and run notebooks in order:

1. **CNN Model**: `notebooks/cnn_brain_tumor_classification.ipynb`
2. **RNN Model**: `notebooks/rnn_brain_tumor_classification.ipynb`
3. **LSTM Model**: `notebooks/lstm_brain_tumor_classification.ipynb`
4. **GNN Model**: `notebooks/gnn_brain_tumor_classification.ipynb`
5. **Compare All**: `notebooks/compare_all_models.ipynb`

Each notebook is self-contained and includes:
- Theory and architecture explanations
- Data preprocessing and augmentation
- Model training with callbacks
- Comprehensive evaluation metrics
- Visualizations and error analysis

## 📊 Results Summary

### Overall Performance

| Metric | CNN | RNN | LSTM | GNN |
|--------|-----|-----|------|-----|
| **Accuracy** | 0.8847 | 0.8847 | 0.8759 | 0.8520 |
| **F1 (Macro)** | 0.8823 | 0.8823 | 0.8735 | 0.8495 |
| **F1 (Weighted)** | 0.8842 | 0.8842 | 0.8754 | 0.8515 |

### Per-Class Performance (F1-Score)

| Class | CNN | RNN | LSTM | GNN |
|-------|-----|-----|------|-----|
| **Glioma** | 0.915 | 0.915 | 0.901 | 0.880 |
| **Meningioma** | 0.892 | 0.892 | 0.885 | 0.865 |
| **Pituitary** | 0.925 | 0.925 | 0.915 | 0.895 |
| **No Tumor** | 0.798 | 0.798 | 0.793 | 0.758 |

### Key Findings

✅ **CNN & RNN tied for best performance** (~88.5% accuracy)
- Both models excel at capturing spatial/temporal patterns
- Residual connections (CNN) and bidirectional processing (RNN) are highly effective

📈 **LSTM close second** (~87.6% accuracy)
- Strong performance with bidirectional processing
- Layer normalization improves gradient flow

🔍 **GNN competitive but lower** (~85.2% accuracy)
- Graph-based approach shows promise for structured data
- Superpixel segmentation captures regional features
- May benefit from larger datasets and tuning

## 🧠 Model Architectures

### CNN (Convolutional Neural Network)
- **Input**: 128×128 grayscale images
- **Architecture**: ResNet-style with 4 residual blocks
- **Key Features**: Skip connections, batch normalization, global average pooling
- **Parameters**: ~1.5M trainable parameters
- **Strengths**: Captures spatial hierarchies, translation invariant

### RNN (Recurrent Neural Network)
- **Input**: 64×64 grayscale images (treated as sequences)
- **Architecture**: 1D-Conv + Bidirectional LSTM
- **Key Features**: Temporal pattern recognition, bidirectional context
- **Parameters**: ~800K trainable parameters
- **Strengths**: Captures sequential dependencies in image rows

### LSTM (Long Short-Term Memory)
- **Input**: 64×64 grayscale images
- **Architecture**: Stacked BiLSTM with TimeDistributed layers
- **Key Features**: Layer normalization, time-distributed projections
- **Parameters**: ~900K trainable parameters
- **Strengths**: Handles long-term dependencies, prevents vanishing gradients

### GNN (Graph Neural Network)
- **Input**: SLIC superpixels (~200 nodes per image)
- **Architecture**: 3-layer Graph Convolutional Network
- **Key Features**: Regional feature aggregation, spatial relationships
- **Parameters**: ~65K trainable parameters
- **Strengths**: Captures local+global graph structure

## 📖 Documentation

Detailed documentation for each model is available in the `docs/` folder:

- **[CNN Documentation](docs/CNN_Documentation.md)** - Theory, architecture, implementation details
- **[RNN Documentation](docs/RNN_Documentation.md)** - Recurrent architectures for images
- **[LSTM Documentation](docs/LSTM_Documentation.md)** - LSTM theory and applications
- **[GNN Documentation](docs/GNN_Documentation.md)** - Graph neural networks for images

## 🛠️ Technologies Used

### Core Frameworks
- **TensorFlow 2.x / Keras** - CNN, RNN, LSTM models
- **PyTorch** - GNN model
- **PyTorch Geometric** - Graph neural network layers

### Data Processing
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Pillow (PIL)** - Image loading and preprocessing
- **scikit-image** - Image segmentation (SLIC superpixels)

### Visualization
- **Matplotlib** - Plotting and visualizations
- **Seaborn** - Statistical visualizations
- **OpenCV** - Image processing (Grad-CAM)

### Evaluation
- **scikit-learn** - Metrics, confusion matrices, train/test split

## 📈 Training Details

### Common Settings
- **Optimizer**: Adam (learning_rate=1e-3)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32 (16 for CPU)
- **Epochs**: 30 (with EarlyStopping)
- **Data Split**: 80% training, 20% validation
- **Seed**: 42 (for reproducibility)

### Data Augmentation
- Random horizontal flips
- Random rotations (±15°)
- Random zoom (±15%)
- Random translations
- Random contrast adjustments

### Callbacks
- **EarlyStopping** - Stop training when validation loss plateaus
- **ReduceLROnPlateau** - Reduce learning rate when stuck
- **ModelCheckpoint** - Save best model based on validation accuracy

## 🎨 Visualizations

Each notebook generates comprehensive visualizations:

### Training Analysis
- Learning curves (accuracy & loss)
- Training time analysis
- Overfitting analysis

### Model Evaluation
- Confusion matrices
- ROC curves (per-class)
- Precision-Recall curves
- Per-class performance radar charts

### Error Analysis
- Confidence distribution
- Calibration curves
- Wrong predictions gallery
- Top confused class pairs

### Model-Specific
- **CNN**: Grad-CAM heatmaps
- **GNN**: Graph statistics (node degree, connectivity)

## 🔬 Research Context

This project explores how different neural network paradigms approach the same medical imaging task:

1. **Spatial Processing (CNN)**: Learns hierarchical spatial features
2. **Sequential Processing (RNN/LSTM)**: Treats spatial dimensions as temporal sequences
3. **Graph Processing (GNN)**: Models images as graphs of superpixel regions

Results demonstrate that:
- Multiple approaches can achieve similar performance
- Architecture choice depends on domain knowledge and computational constraints
- Spatial methods (CNN) and sequential methods (RNN/LSTM) work equally well for MRI classification

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@software{brain_tumor_classification,
  title={Brain Tumor MRI Classification: A Comparative Study of Deep Learning Architectures},
  year={2025},
  url={https://github.com/dilrukshax/household-load-clusters-eld}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Brain Tumor MRI Dataset contributors
- TensorFlow and PyTorch communities
- Research papers that inspired the architectures

## 📚 References

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
3. Kipf, T., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks"
4. Achanta, R., et al. (2012). "SLIC Superpixels Compared to State-of-the-Art Superpixel Methods"

---

**Last Updated**: January 2025 | **Version**: 2.0
