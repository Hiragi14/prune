from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
import typer
from imgclassvalidation.engines import evaluate_classification
from imgclassvalidation.types import EvalConfig
from rich.progress import track
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

app = typer.Typer()
log = getLogger(__name__)


# -------------------------
#  ImageNet1kの前処理
# -------------------------
def cifar100_train_trsfm():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def cifar100_valid_trsfm():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


# -------------------------
# training loop (Pruning用)
# -------------------------
def train(model, train_loader, epochs, optimizer, criterion, device):
    model.to(device)
    model.train()

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

        log.info(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} Acc: {acc:.4f}")


def get_cifar100_dataloaders(
    batch_size: int,
    device: torch.device,
    train_trsfm: transforms.Compose,
    valid_trsfm: transforms.Compose,
    data_dir: str = "/ldisk/DeepLearning/Dataset/ImageNet_torchvision/",
) -> tuple[DataLoader, DataLoader]:
    """cifar100のDataLoaderを作成するユーティリティ関数

    Args:
        batch_size (int): バッチサイズ
        device (torch.device): 使用するデバイス
        train_trsfm (transforms.Compose): 訓練用の前処理
        valid_trsfm (transforms.Compose): 評価用の前処理
        data_dir (str, optional): データディレクトリのパス。 Defaults to "/ldisk/DeepLearning/Dataset/ImageNet_torchvision/".

    Returns:
        tuple[DataLoader, DataLoader]: 訓練用、評価用のDataLoaderのタプル
    """
    # -------------------------
    # 2) Dataset / DataLoader (Evaluation用)
    # -------------------------
    test_ds = datasets.CIFAR100(
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
    train_ds = datasets.CIFAR100(
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


def get_resnet18_model(device: torch.device) -> nn.Module:
    """ResNet-18モデルを取得するユーティリティ関数

    Args:
        device (torch.device): 使用するデバイス

    Returns:
        nn.Module: ResNet-18モデル
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    return model


def save_prune_summary(
    summary: PruneSummary | PruneReport, path: str | Path = "prune_summary.json"
) -> None:
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)


@dataclass(frozen=True)
class PruneResult:
    """
    Container for results returned by the pruning API.
    """

    iteration: int
    loss: float
    top1_acc: float
    top5_acc: float
    fvcore_flop: float
    params: float


@dataclass
class PruneReport:
    """
    Container for the final report of the pruning process.
    """

    density: float
    results: list[PruneResult] = field(default_factory=list)


@dataclass
class PruneSummary:
    """
    Container for the summary of the pruning process.
    """

    report: list[PruneReport] = field(default_factory=list)


@app.command()
def taylor_prune(
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "-d",
        "--device",
        help="Device to use",
    ),
    iterative_steps: int = typer.Option(
        5, "-i", "--iter", help="Number of iterative pruning steps"
    ),
    density: float = typer.Option(
        0.6, "-r", "--density", help="Target density after pruning"
    ),
    epochs: int = typer.Option(
        1, "-e", "--epochs", help="Number of fine-tuning epochs"
    ),
):

    # -------------------------
    # 1) Device / Config
    # -------------------------
    device_ = torch.device(device)

    config = EvalConfig(
        device=device_,
        amp=(device_.type == "cuda"),  # CUDAならAMP on（任意）
        non_blocking=True,
        topk=(1, 5),
        criterion=nn.CrossEntropyLoss(),
    )

    train_loader, test_loader = get_cifar100_dataloaders(
        batch_size=256,
        device=device_,
        train_trsfm=cifar100_train_trsfm(),
        valid_trsfm=cifar100_valid_trsfm(),
    )

    pruned_model = get_resnet18_model(device_)
    example_input = torch.rand(128, 3, 224, 224).to(device)
    ignored_layers = []
    ignored_layers.append(pruned_model.conv1)
    for m in pruned_model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)

    pruner = tp.pruner.BasePruner(
        model=pruned_model,
        example_inputs=example_input,
        importance=tp.importance.TaylorImportance(),
        pruning_ratio=float(1 - density),
        iterative_steps=iterative_steps,
        ignored_layers=ignored_layers,
        round_to=2,
    )
    report = PruneReport(density=density, results=[])

    for iter in range(iterative_steps):
        print(f"\n=== Iteration {iter + 1}/{iterative_steps} ===")

        # 勾配の計算
        pruned_model.zero_grad()
        for k, (images, labels) in track(
            enumerate(train_loader),
            description="Calculating importance scores...",
            transient=True,
        ):
            if k >= 20:  # 最初の20バッチだけ使用（任意）
                break
            images = images.to(device)
            labels = labels.to(device)

            output = pruned_model(images)
            loss = torch.nn.functional.cross_entropy(output, labels)
            loss.backward()

        # Prune
        pruner.step()

        # fine-tuning (任意、今回は無し)
        optimizer = torch.optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        train(
            pruned_model,
            train_loader,
            epochs=epochs,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # -------------------------
        # 1) Run evaluation
        # -------------------------
        result = evaluate_classification(
            model=pruned_model,
            dataloader=test_loader,
            config=config,
            # output_transform=None,  # 通常は不要（logits, yがそのまま）
        )

        # -------------------------
        # 2) Print report
        # -------------------------
        log.info(f"=== Iteration {iter + 1}/{iterative_steps} ===")
        log.info("\n=== Metrics ===")
        for k, v in result.metrics.items():
            log.info(f"{k}: {v:.6f}")

        log.info("\n=== Extras ===")
        for k, v in result.extras.items():
            log.info(f"{k}: {v}")
        report.results.append(
            PruneResult(
                iteration=iter + 1,
                loss=result.metrics["loss"],
                top1_acc=result.metrics["acc1"],
                top5_acc=result.metrics["acc5"],
                fvcore_flop=result.extras["fvcore_flops_total"],
                params=result.extras["params_total"],
            )
        )

    torch.save(
        pruned_model.state_dict(),
        f"pruned_taylor_resnet18_{density:.2f}_{iterative_steps}_iter_ft{epochs}epoch.pth",
    )
    save_prune_summary(
        report,
        path=f"prune_report_taylor_{density:.2f}_{iterative_steps}_iter_ft{epochs}epoch.json",
    )
    return report


DEFAULT_DENSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@app.command()
def taylor_prune_loop(
    densities: list[float] = typer.Option(
        DEFAULT_DENSITIES,
        "-r",
        "--densities",
        help="List of target densities for pruning",
    ),  # noqa: B008
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "-d",
        "--device",
        help="Device to use",
    ),
    iterative_steps: int = typer.Option(
        5, "-i", "--iter", help="Number of iterative pruning steps"
    ),
    epochs: int = typer.Option(
        1, "-e", "--epochs", help="Number of fine-tuning epochs"
    ),
):
    summary = PruneSummary(report=[])
    for density in densities:
        report = taylor_prune(
            device=device,
            iterative_steps=iterative_steps,
            density=density,
            epochs=epochs,
        )
        summary.report.append(report)
        # 結果の集約などは必要に応じて実装
    save_prune_summary(summary, path="taylor_prune_summary.json")
    return summary


if __name__ == "__main__":
    app()
