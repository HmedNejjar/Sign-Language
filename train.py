import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from plot_metrics import MetricsLogger
from dataset import SignLangDataSet
from model import SignGRU
from pathlib import Path
from save_params import save_params, load_params

# ==================== Paths ====================
PARENT = Path(r'G:\Projects\Python\SignLanguage\Dataset')
NSLT_PATH    = PARENT / Path('nslt_300.json')      # JSON file with dataset split information
KEYPOINTS_DIR = PARENT / Path('keypoints')            # Directory containing pre-computed video features
MODEL_PATH = Path('models')
MODEL_NAME = "SignLang_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# ==================== Hyperparameters ====================
BIDIRECTIONAL = True   # Enable bidirectional GRU
BATCH_SIZE = 10        # Number of samples per batch
PATIENCE = 30          # Number of epochs to wait for improvement before early stopping
INPUT_SIZE = 225       # Feature dimension from pre-computed video features
HIDDEN_SIZE = 64       # GRU hidden state dimension
NUM_LAYERS = 2         # Number of GRU layers
NUM_CLASSES = 300      # Number of sign language classes
DROPOUT = 0.3          # Dropout probability
EPOCHS = 1000          # Maximum number of training epochs
LR = 1e-4              # Initial learning rate
FACTOR = 0.5           # Factor to reduce learning rate by on plateau

# Use GPU if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

def accuracy(logits: Tensor, labels: Tensor, topk=(1, 5)) -> list:
    """Calculate top-k accuracy for the given logits and labels.
    
    Args:
        logits: Model predictions with shape (batch_size, num_classes)
        labels: Ground truth class indices with shape (batch_size,)
        topk: Tuple of k values to evaluate (default: top-1 and top-5 accuracy)
        
    Returns:
        List of accuracy values for each k in topk
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        # Get top-k predicted class indices
        _, pred = logits.topk(maxk, dim=1)         # (batch, maxk)
        pred = pred.t()                            # (maxk, batch) - transpose for comparison
        correct = pred.eq(labels.view(1, -1).expand_as(pred))  # (maxk, batch) - boolean correctness matrix

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append((correct_k / batch_size).item())
        return results   # [top1_acc, top5_acc]
    
def train_one_epoch(model: SignGRU, loader: DataLoader, loss_fn: nn.Module, optimizer: Optimizer) -> tuple:
    """Train the model for one epoch.
    
    Args:
        model: SignGRU model to train
        loader: DataLoader for training data
        loss_fn: Loss function (CrossEntropyLoss)
        optimizer: Optimizer for updating model parameters
        
    Returns:
        Tuple of (average_loss, average_top1_accuracy, average_top5_accuracy)
    """
    model.train()  # Set model to training mode (enables dropout, etc.)
    total_loss  = 0
    top1_total  = 0
    top5_total  = 0
    n_batches   = len(loader)
    
    for features, labels in loader:
        # Move batch to device (GPU/CPU)
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()  # Clear previous gradients
        features += 0.01 * torch.randn_like(features)   # Add noise
        logits = model(features)
        loss = loss_fn(logits, labels)
        
        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        
        top1, top5   = accuracy(logits, labels)
        total_loss  += loss.item()
        top1_total  += top1
        top5_total  += top5
    
    return (total_loss / n_batches, top1_total / n_batches, top5_total / n_batches)
    
def evaluate(model: SignGRU, loader: DataLoader, loss_fn: nn.Module) -> tuple:
    """Evaluate the model on validation or test data.
    
    Args:
        model: SignGRU model to evaluate
        loader: DataLoader for validation/test data
        loss_fn: Loss function (CrossEntropyLoss)
        
    Returns:
        Tuple of (average_loss, average_top1_accuracy, average_top5_accuracy)
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    total_loss  = 0
    top1_total  = 0
    top5_total  = 0
    n_batches   = len(loader)

    # Disable gradient computation for efficiency during evaluation
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels   = labels.to(device)

            logits = model(features)
            loss = loss_fn(logits, labels)
            top1, top5 = accuracy(logits, labels)

            total_loss  += loss.item()
            top1_total  += top1
            top5_total  += top5

    return (total_loss / n_batches, top1_total / n_batches, top5_total / n_batches)

if __name__ == "__main__":
    # ==================== Load Data ====================
    # Create datasets for training and validation splits
    train_set = SignLangDataSet(NSLT_PATH, KEYPOINTS_DIR, split='train')
    print(f'Train Length: {len(train_set)}')
    val_set = SignLangDataSet(NSLT_PATH, KEYPOINTS_DIR, split='val')
    print(f'Val Length: {len(val_set)}')
    
    # Create data loaders with batching
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # ==================== Initialize Model ====================
    # Create SignGRU model with specified architecture
    model = SignGRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES, dropout=DROPOUT, bidirectional= BIDIRECTIONAL)
    model = model.to(device)  # Move model to device
    
    # ==================== Setup Training ====================
    # Load pre-trained model if available, otherwise start from scratch
    
    if MODEL_SAVE_PATH.exists():
        print(f"Loading pre-trained model from {MODEL_SAVE_PATH}...")
        model_state = load_params(MODEL_SAVE_PATH)
        model.load_state_dict(model_state)
        print("Model loaded successfully!")
    else:
        print("No pre-trained model found. Starting training from scratch.")
    
    # Loss function and optimizer
    CELoss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # Learning rate scheduler: reduce LR when validation accuracy plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=FACTOR)
    
    best_val_top1  = 0.0    # Track best validation accuracy
    epochs_no_imp  = 0      # Counter for early stopping
    
    # ==================== Training Loop ====================
    logger = MetricsLogger()
    for epoch in range(EPOCHS):
        # Train for one epoch
        train_loss, train_top1, train_top5 = train_one_epoch(model, train_loader, CELoss, optimizer)
        # Evaluate on validation set
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, CELoss)
        
        # Adjust learning rate if validation accuracy plateaus
        scheduler.step(val_top1)
        logger.update(train_loss, train_top1, train_top5, val_loss, val_top1, val_top5)
        # Log training and validation metrics
        print(f"epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} top1: %{100*train_top1:.3f} top5: %{100*train_top5:.3f} | val loss {val_loss:.4f} top1: %{100*val_top1:.3f} top5: %{100*val_top5:.3f}")
        
        # Save model if validation accuracy improved
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            epochs_no_imp = 0  # Reset counter
            save_params(model)
            print(f"  --> saved best model (val top1: {best_val_top1:.3f})")
            
        else:
            # Increment counter if no improvement
            epochs_no_imp += 1
            print(f"  --> no improvement ({epochs_no_imp}/{PATIENCE})")
        
        # Early stopping: stop training if no improvement for PATIENCE epochs
        if epochs_no_imp >= PATIENCE:
            print(f"\nearly stopping at epoch {epoch}")
            break
        
    # ==================== Training Complete ====================
    print(f"\ntraining done. best val top1: {best_val_top1:.3f}")
    print(f"model saved to {MODEL_SAVE_PATH}")
    logger.save(); logger.plot()