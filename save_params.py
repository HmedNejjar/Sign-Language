import torch
from model import SignGRU
from pathlib import Path

def save_params(model: SignGRU):
    MODEL_PATH = Path('models')
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    MODEL_NAME = "SignLang_model.pth"   
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH) 
    
def load_params(path: Path):
    return torch.load(f= path,weights_only= True)