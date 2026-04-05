import torch
import torch.nn as nn
from torch import Tensor


class SignGRU(nn.Module):
    """Sign Language GRU model for video classification.
    
    Architecture: GRU layers process sequential frame features, followed by a linear
    classifier to map to sign language classes.
    """
    
    def __init__(self, input_size: int = 225, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 300, dropout: float = 0.3, bidirectional:bool = True) -> None:
        """Initialize the SignGRU model.
        
        Args:
            input_size: Dimension of input features (keypoint+pose embedding or backbone features). In this project, precomputed keypoints are 225 per frame.
            hidden_size: Dimension of GRU hidden state per direction. The output features are doubled when bidirectional=True.
            num_layers: Number of stacked GRU layers (2)
            num_classes: Number of sign language classes to classify (300)
            dropout: Dropout probability for regularization (0.3)
        """
        super().__init__()
        
        # GRU layers for processing sequential frame features
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        
        # Since GRU is bidirectional, the output hidden dimension is doubled
        effective_hidden = hidden_size * 2
        
        self.attn_pool = AttentionPooling(effective_hidden)
        # Linear classifier: maps final GRU hidden state to class logits
        # Note: Using logits without softmax because nn.CrossEntropyLoss applies softmax internally
        self.classifier = nn.Sequential(
            nn.Linear(effective_hidden, effective_hidden),  # Project hidden representation to classifier feature size
            nn.ReLU(),                            # Non-linear activation
            nn.Dropout(dropout),                  # Regularization to prevent overfitting
            nn.Linear(effective_hidden, num_classes))  # Project to class logits
        
    def forward(self, X: Tensor) -> Tensor:
        """Forward pass through the model.
        
        Args:
            X: Input tensor of shape (batch_size, num_frames, keypoints_dim)
               Example: (8, 32, 225) for batch of 8 videos, 32 frames each, 2048-dim features
        
        Returns:
            logits: Class logits of shape (batch_size, num_classes)
                   Example: (8, 100) for batch of 8 samples, 100 sign classes
        """
        out, hn = self.GRU(X)        # out: (batch, seq_len, hidden_size * 2) for bidirectional GRU
        pooled = self.attn_pool(out)                # (batch, hidden_size * 2) after attention pooling over time
        logits = self.classifier(pooled)
        return logits

class AttentionPooling(nn.Module):
    """Temporal attention pooling over GRU output sequences.

    This layer computes a scalar attention score for each timestep, normalizes
    the scores over the sequence dimension, and returns a weighted sum of the
    input sequence features.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor):
        """Apply attention pooling to a sequence tensor.

        Args:
            x: Tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Tensor of shape (batch, hidden_size) after weighted sum over time.
        """

        # Compute unnormalized attention scores for each timestep
        scores = self.attn(x)              # (batch, seq_len, 1)

        # Normalize attention scores across the temporal dimension
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)

        # Weight each timestep by its attention weight
        weighted = x * weights             # (batch, seq_len, hidden_size)

        # Sum over time to produce a single pooled vector per sample
        pooled = weighted.sum(dim=1)       # (batch, hidden_size)

        return pooled
   
if __name__ == '__main__':
    # Test script to verify model architecture and functionality
    model = SignGRU()
    print("Model Architecture:")
    print(model)

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Sanity check with dummy input to verify shapes
    x = torch.randn(8, 32, 225)   # Fake batch: 8 videos, 32 frames, 225-dim features
    logits = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")  # Should be (8, 100) for 8 samples, 100 classes
    

