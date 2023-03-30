import torch
from torch.nn import Module, Sequential, Conv1d, ReLU, MaxPool1d, Linear


class Conv1dClassifier(Module):
    def __init__(self) -> None:
        super(Conv1dClassifier, self).__init__()
        self.conv = Sequential(
            Conv1d(1, 16, 11, stride=4, padding=1),
            ReLU(),
            MaxPool1d(2),
            Conv1d(16, 16, 5, stride=2, padding=2),
            ReLU(),
            MaxPool1d(2),
        )
        self.fc = Sequential(
            Linear(512, 64),
            ReLU(),
            Linear(64, 1),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], 1, -1)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
