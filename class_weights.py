# class_weights.py
import torch
from pcseg.dataset_toronto3d import Toronto3DDataset

def compute_class_weights(split="train"):
    """
    Compute class weights based on inverse frequency.
    Returns normalized weights as a tensor.
    """
    ds = Toronto3DDataset(split=split, augment=False)
    _, _, labels = ds[0]
    
    counts = torch.bincount(labels)
    weights = 1.0 / (counts.float() + 1e-6)   # inverse frequency
    weights = weights / weights.max()         # normalize
    
    print("Class weights:", weights)
    return weights


if __name__ == "__main__":
    # Test the function
    weights = compute_class_weights("train")
    print("Computed weights:", weights)
