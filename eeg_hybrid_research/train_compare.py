#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from models import CNNBaseline, HybridCNNLSTMGNN, LSTMBaseline
from models.graph_utils import (
    build_correlation_graph,
    build_distance_graph,
    channels_to_positions,
    load_electrode_positions,
    normalize_adjacency,
)


class EEGDataset(Dataset):
    def __init__(self, x, y, n_channels, n_times):
        self.x = torch.tensor(x.reshape(-1, n_channels, n_times), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv_dataset(data_dir: Path):
    x_train = pd.read_csv(data_dir / "training_set.csv", header=None).values.astype(np.float32)
    x_test = pd.read_csv(data_dir / "test_set.csv", header=None).values.astype(np.float32)

    y_train = pd.read_csv(data_dir / "training_label.csv", header=None).values
    y_test = pd.read_csv(data_dir / "test_label.csv", header=None).values

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    if y_train.ndim > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return x_train, y_train, x_test, y_test


def get_shape(data_dir: Path, feature_dim: int, n_channels_arg: int):
    meta_path = data_dir / "dataset_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta["n_channels"]), int(meta["n_times"]), meta

    if n_channels_arg <= 0:
        raise ValueError("Could not infer shape. Provide --n-channels when dataset_meta.json is missing.")
    if feature_dim % n_channels_arg != 0:
        raise ValueError("Feature dimension {} not divisible by n_channels {}".format(feature_dim, n_channels_arg))
    return n_channels_arg, feature_dim // n_channels_arg, {}


def build_adjacency(data_dir: Path, electrode_path: Path, x_train_shaped: np.ndarray, graph_k: int, sigma: float):
    ch_names_path = data_dir / "channel_names.csv"
    ch_names = None
    if ch_names_path.exists():
        ch_names = pd.read_csv(ch_names_path, header=None).values.squeeze().tolist()
        ch_names = [str(x) for x in ch_names]

    positions = load_electrode_positions(str(electrode_path))
    if ch_names and positions:
        coords = channels_to_positions(ch_names, positions)
    else:
        coords = None

    if coords is not None:
        adj = build_distance_graph(coords=coords, sigma=sigma, k=graph_k)
        source = "electrode_positions"
    else:
        adj = build_correlation_graph(train_x=x_train_shaped, k=graph_k)
        source = "train_data_correlation"

    adj = normalize_adjacency(adj)
    return torch.tensor(adj, dtype=torch.float32), source


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
    return float(np.mean(losses)), accuracy_score(ys, preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
    return float(np.mean(losses)), accuracy_score(ys, preds), ys, preds


def plot_history(history_df: pd.DataFrame, model_name: str, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("{} Loss".format(model_name))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val")
    axes[1].set_title("{} Accuracy".format(model_name))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "{}_curves.png".format(model_name), dpi=160)
    plt.close(fig)


def plot_confusion(y_true, y_pred, model_name: str, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title("{} Confusion Matrix".format(model_name))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_dir / "{}_confusion_matrix.png".format(model_name), dpi=180)
    plt.close(fig)


def train_one_model(
    model_name: str,
    n_channels: int,
    n_times: int,
    n_classes: int,
    train_loader,
    val_loader,
    test_loader,
    adjacency,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
):
    if model_name == "cnn":
        model = CNNBaseline(n_channels=n_channels, n_classes=n_classes).to(device)
    elif model_name == "lstm":
        model = LSTMBaseline(n_channels=n_channels, n_classes=n_classes).to(device)
    elif model_name == "hybrid":
        model = HybridCNNLSTMGNN(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            adjacency=adjacency.to(device),
        ).to(device)
    else:
        raise ValueError("Unsupported model: {}".format(model_name))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, patience // 3), verbose=False
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    no_improve = 0
    history_rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(
            "[{}] Epoch {:03d} | train_acc={:.4f} | val_acc={:.4f} | lr={:.2e}".format(
                model_name, epoch, train_acc, val_acc, optimizer.param_groups[0]["lr"]
            )
        )

        if no_improve >= patience:
            print("[{}] Early stopping at epoch {}.".format(model_name, epoch))
            break

    model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    test_f1 = f1_score(y_true, y_pred, average="macro")

    history_df = pd.DataFrame(history_rows)
    return model, history_df, best_val_acc, test_acc, test_f1, y_true, y_pred, test_loss


def parse_models(spec: str):
    if spec.strip().lower() == "all":
        return ["cnn", "lstm", "hybrid"]
    return [x.strip().lower() for x in spec.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train CNN/LSTM/Hybrid CNN+LSTM+GNN and compare results.")
    parser.add_argument("--data-dir", default="data/processed/EEG-Motor-Movement-Imagery-Dataset")
    parser.add_argument(
        "--electrode-path",
        default="../sources/EEG-DL-master/Download_Raw_EEG_Data/electrode_positions.txt",
        help="Electrode coordinates file from the referenced repos",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models", default="all", help="all or comma list: cnn,lstm,hybrid")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--graph-k", type=int, default=10)
    parser.add_argument("--graph-sigma", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-channels", type=int, default=0, help="Only needed if dataset_meta.json is absent")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_csv_dataset(data_dir)
    n_channels, n_times, meta = get_shape(data_dir, x_train.shape[1], args.n_channels)
    n_classes = int(max(np.max(y_train), np.max(y_test)) + 1)

    split = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    tr_idx, val_idx = next(split.split(x_train, y_train))
    x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
    x_val, y_val = x_train[val_idx], y_train[val_idx]

    train_ds = EEGDataset(x_tr, y_tr, n_channels=n_channels, n_times=n_times)
    val_ds = EEGDataset(x_val, y_val, n_channels=n_channels, n_times=n_times)
    test_ds = EEGDataset(x_test, y_test, n_channels=n_channels, n_times=n_times)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    x_tr_shaped = x_tr.reshape(-1, n_channels, n_times)
    electrode_path = Path(args.electrode_path).resolve()
    adjacency, graph_source = build_adjacency(
        data_dir=data_dir,
        electrode_path=electrode_path,
        x_train_shaped=x_tr_shaped,
        graph_k=args.graph_k,
        sigma=args.graph_sigma,
    )
    np.save(results_dir / "adjacency_matrix.npy", adjacency.numpy())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = parse_models(args.models)

    summary = []
    for model_name in model_names:
        (
            model,
            history_df,
            best_val_acc,
            test_acc,
            test_f1,
            y_true,
            y_pred,
            test_loss,
        ) = train_one_model(
            model_name=model_name,
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            adjacency=adjacency,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )

        history_path = results_dir / "{}_history.csv".format(model_name)
        history_df.to_csv(history_path, index=False)
        torch.save(model.state_dict(), results_dir / "{}_best.pt".format(model_name))
        plot_history(history_df, model_name=model_name, out_dir=results_dir)
        plot_confusion(y_true, y_pred, model_name=model_name, out_dir=results_dir)

        summary.append(
            {
                "model": model_name,
                "best_val_acc": float(best_val_acc),
                "test_acc": float(test_acc),
                "test_f1_macro": float(test_f1),
                "test_loss": float(test_loss),
                "graph_source": graph_source if model_name == "hybrid" else "",
            }
        )
        print("[{}] Test accuracy={:.4f}, macro-F1={:.4f}".format(model_name, test_acc, test_f1))

    summary_df = pd.DataFrame(summary).sort_values(by="test_acc", ascending=False)
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary_df["model"], summary_df["test_acc"], color=["#5B8FF9", "#61DDAA", "#F6BD16"][: len(summary_df)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison")
    for i, val in enumerate(summary_df["test_acc"]):
        ax.text(i, val + 0.01, "{:.2%}".format(val), ha="center")
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    run_info = {
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "graph_source": graph_source,
        "models": model_names,
        "seed": args.seed,
    }
    if meta:
        run_info["dataset_meta"] = meta
    with (results_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print("\nTraining complete. Results saved in {}".format(results_dir))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

