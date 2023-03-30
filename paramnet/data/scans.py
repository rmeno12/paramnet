from pathlib import Path
from typing import List, Optional, Tuple

import torch
from rosbag import Bag
from torch.utils.data import Dataset


class RosbagScanDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        bag_files: List[Path],
        scan_topic: str = "/scan",
        joystick_topic: str = "/joystick",
    ) -> None:
        self.bags: List[Bag] = [Bag(bag_file) for bag_file in bag_files]
        self.scan_topic = scan_topic
        self.joystick_topic = joystick_topic

        self.X, self.y = self._load_data()

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.cat(
            [
                self._msgs_to_tensor(
                    bag.read_messages(topics=[self.scan_topic, self.joystick_topic])
                )
                for bag in self.bags
            ]
        )
        return data[:, 1:], data[:, 0]

    def _msgs_to_tensor(self, msgs) -> torch.Tensor:
        last_class: Optional[int] = None
        last_scan: Optional[List[float]] = None
        rows: List[torch.Tensor] = []
        for topic, msg, _ in msgs:
            if topic == self.scan_topic:
                last_scan = msg.ranges
                if last_class is not None:
                    rows.append(
                        torch.tensor([[last_class, *last_scan]], dtype=torch.float32)
                    )
            elif topic == self.joystick_topic:
                last_class = msg.buttons[0] == 1

        return torch.cat(rows)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]
