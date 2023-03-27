import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import Conv1d, Linear, MaxPool1d, Module, ReLU, Sequential, Sigmoid
from torcheval.metrics import BinaryF1Score, Mean

import dataset


class Conv1dClassifier(Module):
    def __init__(self, input_size):
        super(Conv1dClassifier, self).__init__()
        self.conv = Sequential(
            Conv1d(1, 4, 3),
            ReLU(),
            MaxPool1d(2),
            Conv1d(4, 16, 3),
            ReLU(),
            MaxPool1d(2),
        )
        self.fc = Sequential(
            Linear(4096, 128),
            ReLU(),
            Linear(128, 1),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    d = dataset.CurveParamClassifierDataset(
        [
            "ahgtrack2.bag",
            "ahgtrack3.bag",
            "ahgtrack4.bag",
            "ahgtrack5.bag",
            "ahgtrack6.bag",
            "ahgtrack7.bag",
            "ahgtrack8.bag",
        ]
    )
    train, val = torch.utils.data.random_split(d, [0.8, 0.2])
    tloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=False)
    print(f"Train: {len(train)}, Val: {len(val)}")

    learning_rate = 0.005
    epochs = 500

    model = Conv1dClassifier(d[0][0].shape[0]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]).to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.25, patience=10, verbose=True
    )
    early_stopper = EarlyStopper(patience=25)

    tlosses = []
    vlosses = []
    tf1s = []
    vf1s = []
    for t in range(epochs):
        # if t == 300:
        #     learning_rate = 0.005
        #     optimizer.param_groups[0]["lr"] = learning_rate

        model.train(True)
        tf1 = BinaryF1Score(device=device)
        tfl = Mean(device=device)
        for batch, (X, y) in enumerate(tloader):
            X, y = X.to(device), y.to(device)

            # predict
            pred = model(X)
            loss = loss_fn(pred, y)
            tf1.update(torch.sigmoid(pred[:, 0]), y[:, 0])
            tfl.update(loss)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tlosses.append(tfl.compute().item())
        tf1s.append(tf1.compute().item())

        model.train(False)
        vf1 = BinaryF1Score(device=device)
        vl = Mean(device=device)
        for batch, (X, y) in enumerate(vloader):
            X, y = X.to(device), y.to(device)

            # predict
            pred = model(X)
            loss = loss_fn(pred, y)
            vf1.update(torch.sigmoid(pred[:, 0]), y[:, 0])
            vl.update(loss)
        vlosses.append(vl.compute().item())
        vf1s.append(vf1.compute().item())

        scheduler.step(vlosses[-1])

        if t % 10 == 9:
            print(f"Epoch {t+1}")
            print(f"Val Loss: {vlosses[-1]}")
            print(f"Val F1 Score: {vf1s[-1]}")

        if early_stopper.early_stop(vlosses[-1]):
            print(f"Early stopping at epoch {t+1}")
            break

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(tlosses, label="Train")
    ax1.plot(vlosses, label="Val")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(tf1s, label="Train")
    ax2.plot(vf1s, label="Val")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    plt.savefig(f"graphs/W_Conv1D_4_16_4096_238__{vf1s[-1]:.2f}.png")

    torch.save(model.state_dict(), f"models/W_Conv1D_4_16_4096_238__{vf1s[-1]:.2f}.pth")
