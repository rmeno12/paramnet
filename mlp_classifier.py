import torch
from matplotlib import pyplot as plt
from torch.nn import Linear, Module, ReLU, Sequential, Sigmoid
from torcheval.metrics import BinaryF1Score, Mean

import dataset



class MLPClassifier(Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.seq = Sequential(
            Linear(input_size, 512),
            ReLU(),
            Linear(512, 128),
            ReLU(),
            Linear(128, 32),
            ReLU(),
            Linear(32, 1),
        )

    def forward(self, x):
        return self.seq(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    d = dataset.CurveParamClassifierDataset("curve_params.bag")
    train, val = torch.utils.data.random_split(
        d, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )
    tloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    vloader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False)

    learning_rate = 0.01
    epochs = 500

    model = MLPClassifier(d[0][0].shape[0]).to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs")
    #     model = torch.nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))

    tlosses = []
    vlosses = []
    tf1s = []
    vf1s = []
    for t in range(epochs):
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

        if t % 10 == 9:
            print(f"Epoch {t+1}")
            print(f"Val Loss: {vlosses[-1]}")
            print(f"Val F1 Score: {vf1s[-1]}")


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(tlosses, label="Train")
    ax1.plot(vlosses, label="Val")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(tf1s, label="Train")
    ax2.plot(vf1s, label="Val")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    plt.savefig(f"graphs/W_MLP_512_128_32__{vf1s[-1]:.2f}.png")

    torch.save(model.state_dict(), f"models/W_MLP_512_128_32__{vf1s[-1]:.2f}.pth")
