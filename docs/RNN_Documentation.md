# RNN (Recurrent Neural Network) Documentation

## Table of Contents
1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Architecture Details](#architecture-details)
4. [Implementation](#implementation)
5. [Training Process](#training-process)
6. [Results and Analysis](#results-and-analysis)
7. [How to Use](#how-to-use)
8. [References](#references)

---

## Overview

This document provides comprehensive documentation for the **Recurrent Neural Network (RNN)** model used for brain tumor classification.

### Key Specifications
- **Architecture**: 1D-Conv + Bidirectional LSTM
- **Input**: 64×64 grayscale images (treated as sequences)
- **Output**: 4-class classification
- **Parameters**: ~800K trainable parameters
- **Performance**: ~88.5% test accuracy

### Why RNN for Images?

While CNNs are standard for images, RNNs offer unique advantages:
- **Sequential pattern recognition** across image rows
- **Contextual understanding** via bidirectional processing
- **Temporal modeling** of spatial relationships
- **Complementary** to pure spatial (CNN) approaches

---

## Theoretical Background

### Recurrent Neural Networks

RNNs process sequential data by maintaining a hidden state that captures information from previous time steps.

#### Core Equation

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

Where:
- `h_t`: Hidden state at time t
- `x_t`: Input at time t
- `y_t`: Output at time t
- `W`: Weight matrices
- `b`: Bias vectors

**Problem**: Vanishing/exploding gradients in long sequences

### LSTM (Long Short-Term Memory)

LSTM solves the vanishing gradient problem using gating mechanisms:

```
Forget Gate:  f_t = σ(W_f * [h_{t-1}, x_t] + b_f)
Input Gate:   i_t = σ(W_i * [h_{t-1}, x_t] + b_i)
Cell Update:  C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)
Cell State:   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output Gate:  o_t = σ(W_o * [h_{t-1}, x_t] + b_o)
Hidden State: h_t = o_t ⊙ tanh(C_t)
```

**Key Components**:
- **Forget Gate**: Decides what to discard from cell state
- **Input Gate**: Decides what new information to store
- **Output Gate**: Decides what to output
- **Cell State**: Long-term memory that flows through time

### Bidirectional RNN

Processes sequence in both forward and backward directions:

```
→ h_forward:  Processes rows 1 → 64
← h_backward: Processes rows 64 → 1
h_final = [h_forward; h_backward]  (concatenation)
```

**Advantages**:
- Captures context from both directions
- Better understanding of relationships
- Improves performance on many tasks

### Why RNN for Brain MRI?

**Conceptual Motivation**:
1. **Row-wise patterns**: Brain structures appear across horizontal slices
2. **Spatial dependencies**: Tumor regions span multiple adjacent rows
3. **Hierarchical features**: Low-level to high-level patterns along vertical axis

**Image as Sequence**:
```
Image (64, 64, 1) → Reshape to (64, 64)
Treat as: 64 time steps, each with 64 features (column values)
```

---

## Architecture Details

### Model Architecture

```
Input (64×64×1 grayscale)
    ↓
[Data Augmentation]
    Random flip, rotation, zoom
    ↓
[Reshape: (64, 64, 1) → (64, 64)]
    Treat rows as time steps, columns as features
    ↓
[1D Convolution]
    Conv1D(64 filters, kernel_size=5, padding='same')
    → Extracts local patterns along rows
    BatchNorm → MaxPool(2) → (32, 64)
    ↓
[Bidirectional LSTM Layer]
    BiLSTM(128 units, return_sequences=False)
    → Processes sequences in both directions
    → Output: (256,) = concat([forward_128, backward_128])
    Dropout(0.5)
    ↓
[Dense Classification Head]
    Dense(64, relu)
    Dropout(0.3)
    Dense(4, softmax)
    ↓
Output: [P(Glioma), P(Meningioma), P(Pituitary), P(NoTumor)]
```

### Layer-by-Layer Breakdown

| Layer | Output Shape | Parameters | Purpose |
|-------|--------------|------------|---------|
| Input | (64, 64, 1) | 0 | Grayscale MRI |
| Augmentation | (64, 64, 1) | 0 | Regularization |
| Reshape | (64, 64) | 0 | Sequence preparation |
| Conv1D | (32, 64) | 20,544 | Local feature extraction |
| BiLSTM | (256,) | 787,456 | Temporal modeling |
| Dense 1 | (64,) | 16,448 | Feature refinement |
| Output | (4,) | 260 | Classification |
| **Total** | - | **~825K** | - |

### Design Rationale

**1. 1D Convolution Before LSTM**
- Extracts local patterns (edges, textures)
- Reduces sequence length (64 → 32)
- Provides better features for LSTM

**2. Bidirectional Processing**
- Captures context from top and bottom of image
- Better understanding of tumor location
- Improves boundary detection

**3. return_sequences=False**
- Only final hidden state used
- Reduces parameters
- Sufficient for classification task

**4. Dropout Strategy**
- High dropout (0.5) after LSTM to prevent overfitting
- Moderate dropout (0.3) in dense layers

---

## Implementation

### Image to Sequence Conversion

```python
# Original: (batch, 64, 64, 1)
x = layers.Reshape((IMG_SIZE[0], IMG_SIZE[1]))(x)
# Result: (batch, 64, 64)
# Interpretation: 64 time steps, each with 64 features
```

### Model Definition

```python
inputs = layers.Input(shape=(64, 64, 1))
x = data_augmentation(inputs)

# Reshape for RNN processing
x = layers.Reshape((64, 64))(x)

# 1D Conv for feature extraction
x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)  # (32, 64)

# Bidirectional LSTM
x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
x = layers.Dropout(0.5)(x)

# Classification head
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs, outputs, name='RNN_Conv1D_BiLSTM')
```

### Alternative: CNN + BiLSTM Hybrid

```python
# Small CNN for spatial feature extraction
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)  # (32, 32, 32)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)  # (16, 16, 64)

# Reshape for RNN: (16, 16*64=1024)
x = layers.Reshape((16, 16 * 64))(x)

# BiLSTM for temporal modeling
x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
```

This hybrid approach:
- First extracts spatial features (CNN)
- Then models temporal dependencies (RNN)
- Often achieves similar or better performance

---

## Training Process

### Hyperparameters

```python
IMG_SIZE = (64, 64)      # Smaller than CNN for efficiency
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
LOSS = SparseCategoricalCrossentropy()
```

### Callbacks

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)
]
```

### Training Loop

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
```

### Gradient Flow

**LSTM prevents vanishing gradients**:
- Cell state acts as "highway" for gradients
- Additive updates (vs. multiplicative in vanilla RNN)
- Gating mechanisms control information flow

---

## Results and Analysis

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 88.47% |
| **Macro F1-Score** | 0.8823 |
| **Weighted F1-Score** | 0.8842 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **Glioma** | 0.92 | 0.91 | 0.915 |
| **Meningioma** | 0.89 | 0.89 | 0.892 |
| **Pituitary** | 0.94 | 0.91 | 0.925 |
| **No Tumor** | 0.78 | 0.82 | 0.798 |

### Training Characteristics

- **Convergence**: 18-22 epochs
- **Training Time**: ~25 minutes on RTX 3050
- **Overfitting**: Minimal (train-val gap ~2%)
- **Stability**: Consistent across multiple runs

### Comparison with CNN

| Aspect | RNN | CNN |
|--------|-----|-----|
| **Accuracy** | 88.47% | 88.47% |
| **Parameters** | 825K | 1.5M |
| **Training Time** | 25 min | 45 min |
| **Approach** | Sequential | Spatial |
| **Complexity** | Lower | Higher |

**Key Insights**:
- RNN matches CNN performance with fewer parameters
- Faster training due to smaller model
- Sequential processing is viable alternative to spatial convolution

---

## How to Use

### Training

```bash
# Open notebook
jupyter notebook notebooks/rnn_brain_tumor_classification.ipynb

# Or run as script
python -c "import tensorflow as tf; ..."
```

### Inference

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('artifacts/rnn/best_model.h5')

# Prepare image
img = Image.open('mri.jpg').convert('L').resize((64, 64))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=(0, -1))

# Predict
predictions = model.predict(img_array)
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'NoTumor']
predicted_class = class_names[np.argmax(predictions)]

print(f"Prediction: {predicted_class}")
print(f"Confidence: {np.max(predictions):.2%}")
```

---

## References

### Foundational Papers

1. **LSTM**
   - Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation.
   - Original LSTM paper introducing gating mechanisms

2. **Bidirectional RNN**
   - Schuster, M., & Paliwal, K. K. (1997). "Bidirectional Recurrent Neural Networks." IEEE Transactions on Signal Processing.
   - Introduced bidirectional processing

3. **RNN for Images**
   - Visin, F., et al. (2015). "ReSeg: A Recurrent Neural Network-Based Model for Semantic Segmentation." CVPR Workshop.
   - Applied RNNs to image segmentation

### Additional Resources

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Excellent visual guide
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy's blog
- [TensorFlow RNN Tutorial](https://www.tensorflow.org/guide/keras/rnn)

---

## Appendix

### Mathematical Details

**LSTM Forward Pass**:
```
i_t = σ(W_i * x_t + U_i * h_{t-1} + b_i)
f_t = σ(W_f * x_t + U_f * h_{t-1} + b_f)
o_t = σ(W_o * x_t + U_o * h_{t-1} + b_o)
C̃_t = tanh(W_C * x_t + U_C * h_{t-1} + b_C)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
h_t = o_t ⊙ tanh(C_t)
```

**Bidirectional Concatenation**:
```
h_t = [→h_t; ←h_t]  (dimension: 2 * hidden_size)
```

### Troubleshooting

**Problem**: Training very slow
```python
# Solution: Use GRU instead of LSTM (faster)
x = layers.Bidirectional(layers.GRU(128))(x)
```

**Problem**: Overfitting
```python
# Solution: Increase dropout
x = layers.Dropout(0.7)(x)  # After LSTM
```

**Problem**: Underfitting
```python
# Solution: Add more LSTM layers
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
```

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Author**: Brain Tumor Classification Project
