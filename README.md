# Sign Language Recognition with GRU

A PyTorch-based sign language recognition system that classifies American Sign Language (ASL) signs from video sequences. The model uses a bidirectional GRU to process temporal hand and pose keypoint sequences extracted via MediaPipe.

## Project Overview

This project tackles **video classification** — a sequential learning problem where temporal dynamics matter. Instead of processing raw video frames (which is computationally expensive), we extract keypoint features once and train a lightweight temporal model.

**Key idea:** Extract hand positions, orientations, and body pose from each frame → normalize the keypoints → feed them into a GRU that learns temporal patterns → classify the sign.

```
Video Frames → MediaPipe Keypoints → Normalize → GRU → Classifier → Sign Label
```

---

## Architecture

### 1. Feature Extraction (MediaPipe)
Each video frame is processed by two MediaPipe models:

- **Hand Landmarker:** Detects 21 keypoints per hand (x, y, z coordinates)
  - Output per hand: 63 dimensions (21 points × 3 coordinates)
  - Both hands: 126 dimensions

- **Pose Landmarker:** Detects 33 body keypoints (shoulders, hips, elbows, knees, etc.)
  - Output: 99 dimensions (33 points × 3 coordinates)

**Total input per frame:** 225 dimensions = (63 + 63 + 99)

### 2. Model Architecture (SignGRU)

```python
Input: (batch_size, num_frames, 225)
         ↓
    GRU Layer 1 (bidirectional)
         ↓
    GRU Layer 2 (bidirectional)
    Output: (batch_size, num_frames, 128)  # 64 hidden × 2 directions
         ↓
    Attention Pooling
    Output: (batch_size, 128)
         ↓
    Classifier
    - Linear(128 → 128) + ReLU + Dropout
    - Linear(128 → num_classes)
         ↓
    Logits: (batch_size, num_classes)
```

**Key components:**

| Component | Purpose |
|-----------|---------|
| **Bidirectional GRU** | Processes sequence forward AND backward to capture temporal context |
| **Attention Pooling** | Learns which frames matter most; computes weighted average across time |
| **Classifier MLP** | Maps GRU output to class logits |

---

## Data Pipeline

### Dataset Structure

```
Dataset/
├── nslt_300.json                 # Metadata: video splits, class labels, frame ranges
├── keypoints/
│   ├── video_001.npy            # Pre-computed features: shape (T, 225)
│   ├── video_002.npy
│   └── ...
└── videos/
    ├── video_001.mp4
    ├── video_002.mp4
    └── ...
```

### Dataset Class (`dataset.py`)

The `SignLangDataSet` class:
1. **Loads metadata** from the JSON file (which videos are in train/val/test)
2. **Maps class indices** from global indices to local 0-99 indices
3. **Loads pre-computed features** as numpy arrays
4. **Applies augmentation** only during training:
   - **Horizontal flip:** Mirror left/right hands and invert x-coordinates
   - **Temporal reverse:** Play video backwards

```python
# Example usage
train_set = SignLangDataSet('nslt_300.json', 'keypoints', split='train')
features, label = train_set[0]
print(features.shape)  # (T, 225) - variable length sequence
print(label)           # class index 0-99
```

**Key insight:** Pre-computing keypoints decouples feature extraction from model training, making iterations much faster.

---

## Training

### Hyperparameters (`train.py`)

```python
BATCH_SIZE = 10          # Small batches for low data regime
HIDDEN_SIZE = 64         # GRU hidden dimension per direction
NUM_LAYERS = 2           # Stacked GRU layers
DROPOUT = 0.3            # Regularization
EPOCHS = 1000            # Max epochs (early stopping kicks in first)
LR = 1e-4                # Learning rate
PATIENCE = 30            # Early stopping: stop if no improvement for 30 epochs
```

### Training Loop

**For each epoch:**
1. **Train phase:** Forward pass → compute loss → backward → update weights
   - Add small Gaussian noise to features (regularization)
2. **Validation phase:** Evaluate on held-out data without updating
3. **Learning rate scheduling:** If validation accuracy plateaus, reduce LR by factor of 0.5
4. **Early stopping:** Stop if validation accuracy hasn't improved for 30 epochs

**Metrics tracked:**
- Cross-entropy loss
- Top-1 accuracy (correct class in top prediction)
- Top-5 accuracy (correct class in top 5 predictions)

### Running Training

```bash
python train.py
```

Output example:
```
epoch 00/1000 | train loss 4.5832 top1: %1.250 top5: %5.000 | val loss 4.5912 top1: %1.000 top5: %5.000
  --> saved best model (val top1: 0.010)
epoch 01/1000 | train loss 4.5721 top1: %2.500 top5: %7.500 | val loss 4.5801 top1: %2.000 top5: %8.000
  --> saved best model (val top1: 0.020)
...
```

**Note:** At very small dataset scales (~14 samples per class), validation metrics are extremely noisy. Treat single-epoch improvements with skepticism — focus on trends over 5-10 epochs.

---

## Inference

### Real-time Sign Recognition (`main.py`)

The system runs a live webcam loop:

1. **Capture frame** from webcam
2. **Extract keypoints** using MediaPipe
3. **Normalize** per-frame (z-score normalization)
4. **Buffer frames** in a sliding window (32 frames)
5. **Run inference** once buffer is full
6. **Display results** with confidence score

```python
python main.py
```

**What you'll see:**
- Live video feed with hand/pose landmarks drawn
- Running prediction updated every 32 frames
- Confidence percentage in top-left corner

Press `q` to quit.

---

## Key Files

| File | Purpose |
|------|---------|
| **model.py** | SignGRU architecture + AttentionPooling |
| **dataset.py** | PyTorch Dataset class + data augmentation |
| **train.py** | Training loop, validation, early stopping |
| **main.py** | Real-time inference from webcam |
| **plot_metrics.py** | Training curve visualization |
| **save_params.py** | Model checkpoint saving/loading |
| **video_utils.py** | Video frame loading utilities |
| **avg_frames.py** | Dataset analysis (frame count statistics) |

---

## Understanding the Model: Concrete Example

Let's walk through what happens when you classify a sign:

### Input: A 32-frame video of the sign "HELLO"

```
Frame 0:  hand_x=0.45, hand_y=0.32, hand_z=0.01, ... (225 dims total)
Frame 1:  hand_x=0.46, hand_y=0.31, hand_z=0.02, ...
...
Frame 31: hand_x=0.50, hand_y=0.30, hand_z=0.05, ...

Input tensor shape: (1, 32, 225)
                     └─ batch │  frames │  keypoints
```

### Through the GRU

**Layer 1 (forward):** Processes frames 0 → 31, outputs 64-dim hidden state
**Layer 1 (backward):** Processes frames 31 → 0, outputs 64-dim hidden state
→ Concatenate: 128 dimensions

**Layer 2:** Repeats the process on the 128-dim sequence
→ Output: (1, 32, 128)

Each of the 32 timesteps now has a 128-dimensional representation learned from context.

### Through Attention Pooling

The model learns which frames matter:
- Maybe frames 10-20 are most discriminative for "HELLO" (peak hand movement)
- Attention learns to weight those frames higher
- Result: weighted sum of all 32 frames → (1, 128)

### Through Classifier

```
(1, 128) → Linear → ReLU → Dropout → Linear → (1, 300)
                                               └─ logits for 300 classes
```

### Softmax → Prediction

```
softmax(logits) → probabilities sum to 1
argmax → class index (e.g., 42 = "HELLO")
max probability → confidence (e.g., 0.87 = 87%)
```

---

## Known Limitations & Debugging

### Data Constraint Issue

The current dataset has **~1,400 training samples** across **100 classes** → roughly **14 samples per class**. This is very small.

**Symptom:** Loss stuck at ~4.605 (random chance for 100 classes)

**Root causes identified:**
1. **Architecture overfits easily** with so few samples
2. **Validation metrics are noisy** — single samples shift accuracy significantly
3. **Hyperparameter mismatches** between `model.py` defaults and `train.py` overrides

**Diagnostic:** Single-batch overfit test shows the model *can* learn (gradient flow is fine) — the issue is **generalization with little data**, not the architecture.

### Next Steps

1. **Reduce class count:** Train on top 20-50 classes with more samples each
2. **Data augmentation:** Expand the current geometric transforms (flip, reverse)
3. **Simpler baseline:** Replace GRU with mean-pooling + MLP to reduce parameters
4. **Collect more data:** More samples per class dramatically improves generalization

---

## Useful Commands

```bash
# Run training
python train.py

# Plot training curves from saved metrics
python plot_metrics.py

# Run real-time inference
python main.py

# Analyze dataset statistics
python avg_frames.py

# Check model architecture and parameter count
python model.py

# Verify dataset loading
python dataset.py
```

---

## Understanding the Math: Attention Pooling

Here's the mechanism under the hood:

```python
# Input: (batch=2, seq_len=32, hidden_size=128)
x = torch.randn(2, 32, 128)

# Learn attention scores for each timestep
attn = nn.Linear(128, 1)
scores = attn(x)  # (2, 32, 1)

# Normalize across time
weights = torch.softmax(scores, dim=1)  # (2, 32, 1), sum to 1 across frames

# Weight each frame by its importance
weighted = x * weights  # broadcast: (2, 32, 128)

# Sum across time
pooled = weighted.sum(dim=1)  # (2, 128)
```

**Intuition:** Instead of always taking the last frame (or mean), the model learns which frames are most informative and weights them accordingly.

---

## Loss Function: Cross-Entropy

For multiclass classification:

```
L = -log(p_true_class)
```

Where `p_true_class` is the softmax probability assigned to the correct class.

- **Perfect prediction:** p = 1.0 → L = 0
- **Random guess (100 classes):** p ≈ 0.01 → L ≈ 4.605

If loss is stuck at 4.605, the model is predicting uniformly random — it's learning nothing.

---

## References

- **GRU:** Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014)
- **Attention:** Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
- **MediaPipe:** Google's real-time perception library (https://mediapipe.dev/)
- **PyTorch:** Deep learning framework (https://pytorch.org/)

---

## Questions to Deepen Understanding

1. **Why bidirectional?** What information does the backward pass provide that forward alone doesn't?
2. **Why attention pooling instead of just the last hidden state?** What about mean pooling?
3. **Why is dropout applied during training but not inference?** What would happen if you applied it during inference?
4. **What happens if you remove attention and use `hidden[:, -1, :]` (last frame)?** How does accuracy change?
5. **How would you handle variable-length sequences?** (Hint: padding + masking)

---

## License

This is an educational project for learning deep learning fundamentals.
