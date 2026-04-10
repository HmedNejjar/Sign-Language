import torch
from pathlib import Path

def save_params(model, path: Path | str):
    model_path = Path(path)

    if model_path.suffix == '':
        model_dir = model_path
        model_path = model_dir / "SignLang_model.pth"
    else:
        model_dir = model_path.parent

    model_dir.mkdir(parents=True, exist_ok=True)

    for params in model.GRU.parameters():
        params.requires_grad = False

    torch.save(obj=model.state_dict(), f=model_path)
    return model_path
    
def load_params(path: Path | str):
    return torch.load(path)

def freeze(model, train_GRU: bool = False, train_attn_pool: bool = False, train_classifier: bool = True):
    for param in model.GRU.parameters():
        param.requires_grad = train_GRU
    
    for param in model.attn_pool.parameters():
        param.requires_grad = train_attn_pool
    
    for param in model.classifier.parameters():
        param.requires_grad = train_classifier
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}")