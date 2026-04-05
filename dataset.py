import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Custom PyTorch Dataset for sign language recognition
# Loads pre-computed video features and their corresponding class labels
class SignLangDataSet(Dataset):
    def __init__(self, nslt_path: str | Path, features_dir: str | Path, split: str) -> None:
        """Initialize the dataset.
        
        Args:
            nslt_path: Path to the JSON file containing dataset metadata and splits
            features_dir: Directory containing pre-computed .npy feature files
            split: Dataset split - 'train', 'val', or 'test'
        """
        super().__init__()

        self.features_dir = Path(features_dir)
        self.split = split
        
        # Load the JSON metadata file
        with open(nslt_path) as f:
            nslt = json.load(f)
        
        # Create a mapping from global class indices to local 0-99 indices
        # This ensures consistent class label encoding across samples
        all_classes = sorted(set(meta['action'][0] for _, meta in nslt.items()))
        # Map global index -> local 0-99
        self.class_map = {global_idx: local_idx for local_idx, global_idx in enumerate(all_classes)}

        # Build list of (video_id, class_label) pairs for the specified split
        self.sample = [(video_id, meta['action'][0]) for video_id, meta in nslt.items() if meta['subset'] == split]
        
    def __len__(self):
        """Return the total number of samples in this split."""
        return len(self.sample)
    
    def __getitem__(self, idx: int) -> tuple:
        """Retrieve a sample and its label.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (features_tensor, class_label)
                - features_tensor: Pre-computed video features loaded from .npy file
                - class_label: Mapped class index (0-99)
        """
        video_id, class_idx = self.sample[idx]
        features = self.features_dir / f'{video_id}.npy'
        features = torch.tensor(np.load(features), dtype=torch.float32)
    
        return (features, self.class_map[class_idx])
    
if __name__ == '__main__':
    # Test script to verify dataset loading and shapes
    NSLT_PATH    = Path(r'G:\Projects\Python\SignLanguage\Dataset\nslt_300.json')
    FEATURES_DIR = Path(r'G:\Projects\Python\SignLanguage\Dataset\keypoints')

    # Create dataset splits
    train_set = SignLangDataSet(NSLT_PATH, FEATURES_DIR, split='train')
    val_set   = SignLangDataSet(NSLT_PATH, FEATURES_DIR, split='val')
    test_set  = SignLangDataSet(NSLT_PATH, FEATURES_DIR, split='test')

    # Print dataset sizes
    print(f"train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=8, shuffle=False)

    # Sanity check: verify batch shapes and data types
    features, labels = next(iter(train_loader))
    print(f"features shape: {features.shape}")   # (8, 32, 2048) - batch_size, frames, feature_dim
    print(f"labels shape:   {labels.shape}")     # (8,) - batch of class indices
    print(f"labels:         {labels}")
        
    
