from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassF1Score, Mean, MulticlassAccuracy
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from paramnet.data.scans import RosbagScanDataset
from paramnet.models.conv1d import Conv1dMultiClassifier
from paramnet.run.util import EarlyStopper


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("bag_folder", type=Path)
    parser.add_argument("--run_log_folder", type=Path, default=Path("runs"))
    parser.add_argument("--num_classes", type=int, default=3)
    args = parser.parse_args()

    bag_folder = args.bag_folder
    bag_files = list(bag_folder.glob("*.bag"))

    logger.info("Loading data...")
    dataset = RosbagScanDataset(bag_files)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_indices, val_indices = next(split.split(dataset.X, dataset.y))
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # stratified random sampling for training data
    train_class_counts = [
        sum(train_dataset.dataset.y[train_indices] == float(i))
        for i in range(args.num_classes)
    ]
    train_class_weights = 1.0 / torch.tensor(train_class_counts, dtype=torch.float)
    train_weights = train_class_weights[train_dataset.dataset.y[train_indices].int()]
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # stratified random sampling for validation data
    val_class_counts = [
        sum(val_dataset.dataset.y[val_indices] == float(i))
        for i in range(args.num_classes)
    ]
    val_class_weights = 1.0 / torch.tensor(val_class_counts, dtype=torch.float)
    val_weights = val_class_weights[val_dataset.dataset.y[val_indices].int()]
    val_sampler = WeightedRandomSampler(val_weights, len(val_weights))

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler)
    logger.info(
        f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}"
    )
    logger.info("Done loading data.")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    learning_rate = 0.01
    epochs = 500
    model = Conv1dMultiClassifier(args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, verbose=True
    )
    early_stopper = EarlyStopper(min_delta=0.02)

    stamp = str(datetime.now()).replace(" ", "_")
    savedir = Path(args.run_log_folder / stamp)
    savedir.mkdir(parents=True, exist_ok=True)
    ckptdir = savedir.joinpath("ckpts")
    ckptdir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(savedir) + "/tb_logs")
    tb_writer.add_text("Model type", str(model.__class__.__name__))

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()

        tl = Mean(device=device)
        tf = MulticlassF1Score(device=device)
        ta = MulticlassAccuracy(device=device)
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).long()

            pred = model(X).squeeze()
            loss = loss_fn(pred, y)
            tl.update(loss)
            tf.update(pred, y)
            ta.update(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tb_writer.add_scalar("train/loss", tl.compute().item(), epoch)
        tb_writer.add_scalar("train/f1", tf.compute().item(), epoch)
        tb_writer.add_scalar("train/accuracy", ta.compute().item(), epoch)

        model.eval()
        torch.jit.script(model).save(ckptdir / f"convclassifier_{epoch}.pt")
        with torch.no_grad():
            vl = Mean(device=device)
            vf = MulticlassF1Score(device=device)
            va = MulticlassAccuracy(device=device)
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).long()

                pred = model(X).squeeze()
                loss = loss_fn(pred, y)
                vl.update(loss)
                vf.update(pred, y)
                va.update(pred, y)

            val_loss = vl.compute().item()

        scheduler.step(val_loss)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.add_scalar("val/f1", vf.compute().item(), epoch)
        tb_writer.add_scalar("val/accuracy", va.compute().item(), epoch)

        if early_stopper.early_stop(val_loss):
            logger.info(f"Early stopping at epoch {epoch}.")
            break


if __name__ == "__main__":
    main()
