from __future__ import annotations

import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from imgclassvalidation.engines import evaluate_classification
from imgclassvalidation.types import EvalConfig
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

console = Console()
app = typer.Typer()


def set_logger(
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
) -> None:
    """ロガーを設定する関数。RichHandler を使用して、コンソール出力とファイル出力の両方を設定する。

    Args:
        console_level (int, optional): コンソール出力のログレベル。 Defaults to logging.INFO.
        file_level (int, optional): ファイル出力のログレベル。 Defaults to logging.INFO.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # ← 全体は最も低く

    # 既存handlerがある場合の多重登録防止
    if root_logger.handlers:
        root_logger.handlers.clear()

    # --- console handler: Rich ---
    console_handler = RichHandler(
        console=console,
        level=console_level,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=False,
    )

    # RichHandler 側で時刻・レベル・パスを表示するので message だけ
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # --- file handler ---
    file_fmt = logging.Formatter(
        "[%(levelname)s] [%(asctime)s]: [%(filename)s:%(lineno)s] %(funcName)s : %(message)s"
    )
    file_handler = logging.FileHandler(
        f"{str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))}/.log", encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_fmt)

    # 登録
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


set_logger()
log = logging.getLogger(__name__)


# -------------------------
#  CIFAR10の前処理
# -------------------------
def cifar10_train_trsfm():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def cifar10_valid_trsfm():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return correct / total, loss / total


# -------------------------
# training loop (Pruning用)
# -------------------------
def train(
    model,
    train_loader,
    test_loader,
    epochs,
    optimizer,
    criterion,
    device,
    lr_milestones=[30, 60],
    lr_gamma=0.1,
):
    model.to(device)
    model.train()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=lr_gamma
    )

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in track(
            train_loader, description=f"Epoch {epoch + 1}/{epochs}", transient=True
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total

        scheduler.step()

        log.info(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} Acc: {acc:.4f}")

        model.eval()
        with torch.no_grad():
            val_acc, val_loss = eval(model, test_loader, device=device)
        log.info(f"Validation Loss: {val_loss:.4f} Validation Acc: {val_acc:.4f}")


def get_cifar10_dataloaders(
    batch_size: int,
    device: torch.device,
    train_trsfm: transforms.Compose,
    valid_trsfm: transforms.Compose,
    data_dir: str = "/ldisk/DeepLearning/Dataset/CIFAR10/",
) -> tuple[DataLoader, DataLoader]:
    """cifar10のDataLoaderを作成するユーティリティ関数

    Args:
        batch_size (int): バッチサイズ
        device (torch.device): 使用するデバイス
        train_trsfm (transforms.Compose): 訓練用の前処理
        valid_trsfm (transforms.Compose): 評価用の前処理
        data_dir (str, optional): データディレクトリのパス。 Defaults to "/ldisk/DeepLearning/Dataset/CIFAR10/".

    Returns:
        tuple[DataLoader, DataLoader]: 訓練用、評価用のDataLoaderのタプル
    """
    # -------------------------
    # 2) Dataset / DataLoader (Evaluation用)
    # -------------------------
    test_ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=valid_trsfm,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,  # 評価なので基本False
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,  # 大規模評価寄り（環境によりFalseでも可）
    )

    # -------------------------
    # 3) train dataloader (Pruning用)
    # -------------------------
    train_ds = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_trsfm,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # 学習なのでTrue
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,  # 大規模評価寄り（環境によりFalseでも可）
    )
    return train_loader, test_loader


@app.command()
def train_cifar10(
    device: str = typer.Option("cuda", "-d", "--device", help="使用するデバイス"),
    epochs: int = typer.Option(100, "-e", "--epochs", help="訓練するエポック数"),
    lr: float = typer.Option(0.1, "-l", "--lr", help="学習率"),
):
    device_ = torch.device(device if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=128,
        device=device_,
        train_trsfm=cifar10_train_trsfm(),
        valid_trsfm=cifar10_valid_trsfm(),
    )
    train(
        model,
        train_loader,
        test_loader,
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device_,
    )

    config = EvalConfig(
        device=device_,
        amp=(device_.type == "cuda"),  # CUDAならAMP on（任意）
        non_blocking=True,
        topk=(1, 5),
        criterion=nn.CrossEntropyLoss(),
    )

    # -------------------------
    # 1) Run evaluation
    # -------------------------
    result = evaluate_classification(
        model=model,
        dataloader=test_loader,
        config=config,
        # output_transform=None,  # 通常は不要（logits, yがそのまま）
    )

    # -------------------------
    # 2) Print report
    # -------------------------
    log.info("\n=== Metrics ===")
    for k, v in result.metrics.items():
        log.info(f"{k}: {v:.6f}")

    log.info("\n=== Extras ===")
    for k, v in result.extras.items():
        log.info(f"{k}: {v}")
