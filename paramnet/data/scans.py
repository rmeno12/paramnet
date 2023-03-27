from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from paramnet.data.util import BagData


class ScanDataset(Dataset):
    def __init__(self, bag_file: Path) -> None:
        self.bagdata = BagData(bag_file)

        self.X, self.y = self._load_data()

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __len__(self) -> int:
        return len(self.y)
