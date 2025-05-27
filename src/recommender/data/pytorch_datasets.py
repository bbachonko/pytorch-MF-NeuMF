import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class NeuMFDataset(Dataset):
    """Binary implicit feedback dataset that also yields *item content vectors*."""

    def __init__(self, interactions: pd.DataFrame, content_matrix: np.ndarray):
        self.users: torch.LongTensor = torch.as_tensor(interactions.user.values, dtype=torch.long)
        self.items: torch.LongTensor = torch.as_tensor(interactions.item.values, dtype=torch.long)
        self.labels: torch.FloatTensor = torch.as_tensor(interactions.label.values, dtype=torch.float32)
        # Whole matrix as a (n_items × content_dim) tensor for fast indexing
        self._content: torch.FloatTensor = torch.from_numpy(content_matrix)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item_id = int(self.items[idx])
        return (
            self.users[idx],                     # user index
            self.items[idx],                     # item index
            self._content[item_id],              # content vector (no copy!)
            self.labels[idx],
        )
