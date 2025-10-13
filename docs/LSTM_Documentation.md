# LSTM (Long Short-Term Memory) Documentation

## Overview

This document covers the **LSTM model** for brain tumor classification, featuring bidirectional processing and TimeDistributed layers.

### Key Specifications
- **Architecture**: Stacked BiLSTM with TimeDistributed projections
- **Input**: 64×64 grayscale images
- **Output**: 4-class classification
- **Parameters**: ~900K trainable parameters
- **Performance**: ~87.6% test accuracy

---

## Theoretical Background

### LSTM Architecture

LSTM networks address the vanishing gradient problem in standard RNNs through:

1. **Cell State**: Long-term memory pathway
2. **Gates**: Control information flow
   - **Forget Gate**: What to forget
   - **Input Gate**: What to remember
   - **Output Gate**: What to output

### Equations

```
Forget Gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate:    C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell State:   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output Gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State: h_t = o_t ⊙ tanh(C_t)
```

### TimeDistributed Layers

Applies the same Dense layer independently to each time step:

```
For sequence (t_1, t_2, ..., t_n):
    Apply Dense(d) to each t_i independently
    Maintains temporal structure while projecting features
```

### Layer Normalization

Normalizes across features (not batch):

```
μ = mean(x)  (across features)
σ = std(x)   (across features)
y = γ * (x - μ) / σ + β
```

**Benefits over Batch Norm for RNNs**:
- Works with variable sequence lengths
- No dependency on batch statistics
- Stabilizes recurrent activations

---

## Architecture Details

```
Input (64×64×1)
    ↓
[Data Augmentation]
    ↓
[Reshape: (64, 64)]
    Treat as 64 time steps, 64 features each
    ↓
[BiLSTM Layer 1]
    BiLSTM(128 units, return_sequences=True)
    → Output: (64, 256)  [64 timesteps, 256 features]
    ↓
[TimeDistributed Dense]
    TimeDistributed(Dense(64, relu))
    → Applies same Dense layer to each of 64 timesteps
    → Output: (64, 64)
    ↓
[Layer Normalization]
    Normalizes across features for each timestep
    Dropout(0.3)
    ↓
[BiLSTM Layer 2]
    BiLSTM(64 units, return_sequences=False)
    → Aggregates all timesteps into single vector
    → Output: (128)
    Dropout(0.5)
    ↓
[Classification Head]
    Dense(32, relu)
    Dropout(0.3)
    Dense(4, softmax)
    ↓
Output Probabilities
```

### Layer Breakdown

| Layer | Output Shape | Parameters | Purpose |
|-------|--------------|------------|---------|
| Input | (64, 64, 1) | 0 | Image input |
| Reshape | (64, 64) | 0 | Sequence format |
| BiLSTM 1 | (64, 256) | 529,408 | Temporal features |
| TimeDistributed | (64, 64) | 16,448 | Feature projection |
| LayerNorm | (64, 64) | 128 | Stabilization |
| BiLSTM 2 | (128,) | 360,960 | Aggregation |
| Dense 1 | (32,) | 4,128 | Classification |
| Output | (4,) | 132 | Probabilities |
| **Total** | - | **~911K** | - |

### Key Design Choices

1. **Stacked BiLSTMs**: Hierarchical temporal modeling
2. **TimeDistributed**: Per-timestep feature transformation
3. **Layer Normalization**: Better than BatchNorm for sequences
4. **return_sequences**: True for first LSTM, False for second

---

## Implementation

```python
inputs = layers.Input(shape=(64, 64, 1))
x = data_augmentation(inputs)
x = layers.Reshape((64, 64))(x)

# First BiLSTM with return_sequences=True
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

# TimeDistributed projection
x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)

# Layer Normalization (better for RNNs)
x = layers.LayerNormalization()(x)
x = layers.Dropout(0.3)(x)

# Second BiLSTM (aggregation)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
x = layers.Dropout(0.5)(x)

# Classification
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs, outputs, name='BiLSTM_TimeDistributed')
```

---

## Training Process

### Hyperparameters

```python
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
```

### Callbacks

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint(filepath='best_model.h5', save_best_only=True)
]
```

---

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 87.59% |
| **Macro F1** | 0.8735 |
| **Weighted F1** | 0.8754 |

### Per-Class Performance

| Class | F1-Score |
|-------|----------|
| Glioma | 0.901 |
| Meningioma | 0.885 |
| Pituitary | 0.915 |
| No Tumor | 0.793 |

### Training Characteristics

- **Convergence**: 20-25 epochs
- **Training Time**: ~30 minutes (RTX 3050)
- **Stability**: High (consistent results)

---

## How to Use

### Training

```bash
jupyter notebook notebooks/lstm_brain_tumor_classification.ipynb
```

### Inference

```python
model = tf.keras.models.load_model('artifacts/lstm/best_model.h5')

img = Image.open('mri.jpg').convert('L').resize((64, 64))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=(0, -1))

predictions = model.predict(img_array)
class_idx = np.argmax(predictions)
print(f"Predicted: {classes[class_idx]}")
```

---

## References

1. **LSTM**: Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
2. **Layer Normalization**: Ba et al. (2016). "Layer Normalization"
3. **TimeDistributed**: Keras documentation

---

**Last Updated**: January 2025
