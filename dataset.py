import rosbag
import torch
from torch.utils.data import Dataset


def rosbag_to_tensor(rosbag_file) -> torch.Tensor:
    bag = rosbag.Bag(rosbag_file)
    last_param = None
    last_scan = None
    rows = []
    for topic, msg, t in bag.read_messages(topics=["/scan", "/joystick"]):
        if topic == "/scan":
            last_scan = msg.ranges
            if last_param is not None:
                rows.append(
                    torch.tensor([[last_param, *last_scan]], dtype=torch.float32)
                )
        elif topic == "/joystick":
            last_param = msg.buttons[0] == 1

    bag.close()
    return torch.cat(rows, 0)


class CurveParamClassifierDataset(Dataset):
    def __init__(self, rosbag_files, transform=None):
        raw = torch.cat([rosbag_to_tensor(f) for f in rosbag_files], 0)
        pos = raw[raw[:, 0] == 1]
        neg = raw[raw[:, 0] == 0]
        print(f"Pos: {len(pos)}, Neg: {len(neg)}")

        self.data = raw[:, 1:]
        self.labels = raw[:, 0][:, None]

        # self.data = torch.nn.functional.normalize(self.data)
        # m = torch.mean(self.data)
        # s = torch.std(self.data)
        # print(m, s)
        # print(max(self.data[1]))
        # self.data = (self.data - m) / s
        # print(max(self.data[1]))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        if self.transform:
            d = self.transform(d)
        return d, self.labels[idx]
