from typing import Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_dataset_npz(path: str = "dataset.npz", batch_size: int = 128, shuffle: bool = True) -> Tuple[DataLoader, np.ndarray, np.ndarray]:
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    tensor_x = torch.from_numpy(X).float()
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, X, y
