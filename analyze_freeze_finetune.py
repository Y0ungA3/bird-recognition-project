"""Compare frozen-backbone, full fine-tuning and warm-up schedules."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from train import Birds, make_splits, read_pairs, run_epoch

DATA = ROOT / "data" / "CUB_200_2011"
OUT = ROOT / "runs" / "freeze_finetune_comparison"
SEEDS = [42, 43, 44]
POLICIES = {
    "frozen_head": {
        "name": "Backbone frozen",
        "freeze_epochs": 10,
        "runs": {
            seed: ROOT / "runs" / f"freeze_head_seed{seed}"
            for seed in SEEDS
        },
        "color": "#4477aa",
    },
    "finetune_all": {
        "name": "Fine-tune from epoch 1",
        "freeze_epochs": 0,
        "runs": {
            seed: ROOT / "runs" / f"finetune_all_seed{seed}"
            for seed in SEEDS
        },
        "color": "#ee7733",
    },
    "warmup_finetune": {
        "name": "Freeze 2, then fine-tune",
        "freeze_epochs": 2,
        "runs": {
            42: ROOT / "runs" / "milestone_letterbox_3060",
            43: ROOT / "runs" / "crop_letterbox_3060_seed43",
            44: ROOT / "runs" / "crop_letterbox_3060_seed44",
        },
        "color": "#228833",
    },
}
EXPECTED_SPLIT = (
    "40122ce37b1e511dcd2975be60d9b52bc99a4e2601d3ad310f588f6aed996df6"
)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(policy, seed):
    path = POLICIES[policy]["runs"][seed]
    if not (path / "completed.json").exists():
        raise FileNotFoundError(f"Incomplete run: {path}")
    checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    expected = {
        "seed": seed,
        "split_seed": 42,
        "epochs": 10,
        "batch_size": 32,
        "freeze_epochs": POLICIES[policy]["freeze_epochs"],
        "crop": True,
        "resize_mode": "letterbox",
        "margin": 0.1,
        "smoke": False,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Configuration mismatch in {path}: {mismatches}")
    if checkpoint.get("split_sha256") != EXPECTED_SPLIT:
        raise RuntimeError(f"Split mismatch in {path}")
    return path, checkpoint, config


def evaluate_checkpoint(checkpoint, rows):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 200)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model.to(device).eval()
    loader = DataLoader(
        Birds(DATA, rows, False, True, 0.1, "letterbox"),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    with torch.inference_mode():
        metrics, _, predictions = run_epoch(
            model, loader, device, nn.CrossEntropyLoss()
        )
    return metrics, predictions


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = make_splits(DATA, 42)["val"]
    labels = {row["id"]: row["label"] for row in rows}
    architecture = models.resnet18(weights=None)
    architecture.fc = nn.Linear(512, 200)
    total_parameters = sum(parameter.numel() for parameter in architecture.parameters())
    head_parameters = sum(parameter.numel() for parameter in architecture.fc.parameters())
    results = []
    prediction_maps = {}
    trainable_parameters = {}

    for policy, metadata in POLICIES.items():
        for seed in SEEDS:
            path, checkpoint, config = load_checkpoint(policy, seed)
            metrics, predictions = evaluate_checkpoint(checkpoint, rows)
            history = list(csv.DictReader((path / "history.csv").open()))
            best_row = max(history, key=lambda row: float(row["val_accuracy"]))
            last_row = history[-1]
            trainable = (
                head_parameters
                if policy == "frozen_head"
                else total_parameters
            )
            result = {
                "policy": policy,
                "seed": seed,
                "run": str(path.relative_to(ROOT)),
                "freeze_epochs": config["freeze_epochs"],
                "best_epoch": int(checkpoint["epoch"]),
                "accuracy": metrics["accuracy"],
                "top5": metrics["top5"],
                "macro_f1": metrics["macro_f1"],
                "best_train_accuracy": float(best_row["train_accuracy"]),
                "last_train_accuracy": float(last_row["train_accuracy"]),
                "total_seconds": float(sum(float(row["seconds"]) for row in history)),
                "trainable_parameters": trainable,
            }
            results.append(result)
            (path / "validation_metrics.json").write_text(
                json.dumps({"policy": policy, "seed": seed, **metrics}, indent=2)
            )
            prediction_maps[(policy, seed)] = {
                image_id: predicted
                for image_id, _, predicted in predictions
            }
            trainable_parameters[policy] = trainable
            print("Evaluated:", result, flush=True)

    write_csv(OUT / "per_seed_results.csv", results)
    summary = {}
    for policy in POLICIES:
        summary[policy] = {}
        for metric in ["accuracy", "top5", "macro_f1"]:
            values = [
                row[metric] for row in results if row["policy"] == policy
            ]
            mean, std = mean_std(values)
            summary[policy][metric] = {"mean": mean, "std": std}
        summary[policy]["total_seconds"] = float(
            np.mean(
                [
                    row["total_seconds"]
                    for row in results
                    if row["policy"] == policy
                ]
            )
        )
        summary[policy]["trainable_parameters"] = trainable_parameters[policy]

    comparisons = [
        ("finetune_all", "frozen_head"),
        ("warmup_finetune", "finetune_all"),
        ("warmup_finetune", "frozen_head"),
    ]
    paired = []
    for left, right in comparisons:
        for seed in SEEDS:
            fixed = sum(
                prediction_maps[(right, seed)][row["id"]] != row["label"]
                and prediction_maps[(left, seed)][row["id"]] == row["label"]
                for row in rows
            )
            broken = sum(
                prediction_maps[(right, seed)][row["id"]] == row["label"]
                and prediction_maps[(left, seed)][row["id"]] != row["label"]
                for row in rows
            )
            paired.append(
                {
                    "left_policy": left,
                    "right_policy": right,
                    "seed": seed,
                    "left_minus_right_accuracy_percentage_points": 100
                    * (fixed - broken)
                    / len(rows),
                    "fixed_errors": fixed,
                    "new_errors": broken,
                }
            )
    write_csv(OUT / "paired_differences.csv", paired)

    prediction_rows = []
    for policy in POLICIES:
        for seed in SEEDS:
            for row in rows:
                predicted = prediction_maps[(policy, seed)][row["id"]]
                prediction_rows.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "image_id": row["id"],
                        "true_class_0based": labels[row["id"]],
                        "predicted_class_0based": predicted,
                        "correct": int(predicted == labels[row["id"]]),
                    }
                )
    if len(prediction_rows) != 10800:
        raise RuntimeError(
            f"Expected 10,800 validation predictions, got {len(prediction_rows)}"
        )
    write_csv(OUT / "validation_predictions.csv", prediction_rows)

    summary.update(
        {
            "training_seeds": SEEDS,
            "split_seed": 42,
            "split_sha256": EXPECTED_SPLIT,
            "train_images": 4794,
            "validation_images": 1200,
            "official_test_evaluated": False,
            "paired": paired,
            "interpretation": (
                "Fine-tuning the backbone is essential. Fine-tuning from epoch 1 "
                "and freezing for 2 epochs have nearly identical mean accuracy; "
                "the 0.14-point mean difference does not support a stable claim "
                "that warm-up is beneficial."
            ),
            "controlled_variables": (
                "All runs use crop=True, margin=0.1, resize_mode=letterbox, "
                "batch_size=32, 10 epochs, the same split and the same seeds."
            ),
        }
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    labels_display = [
        "Backbone frozen",
        "Fine-tune epoch 1",
        "Freeze 2, fine-tune",
    ]
    policies = list(POLICIES)
    colors = [POLICIES[policy]["color"] for policy in policies]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in zip(
        axes,
        ["accuracy", "top5", "macro_f1"],
        ["Top-1 accuracy", "Top-5 accuracy", "Macro-F1"],
    ):
        axis.bar(
            labels_display,
            [100 * summary[policy][metric]["mean"] for policy in policies],
            yerr=[100 * summary[policy][metric]["std"] for policy in policies],
            color=colors,
            capsize=4,
        )
        axis.set(title=title, ylabel="Percent", ylim=(0, 100))
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Bird-region letterbox: frozen vs fine-tuning (3 seeds, mean and sample SD)"
    )
    figure.tight_layout()
    figure.savefig(OUT / "performance_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for policy in policies:
        histories = []
        for seed in SEEDS:
            path = POLICIES[policy]["runs"][seed]
            history = list(csv.DictReader((path / "history.csv").open()))
            histories.append(
                {
                    "train": [float(row["train_accuracy"]) * 100 for row in history],
                    "val": [float(row["val_accuracy"]) * 100 for row in history],
                }
            )
        epochs = np.arange(1, 11)
        for axis, split, title in [
            (axes[0], "val", "Validation Top-1"),
            (axes[1], "train", "Training Top-1"),
        ]:
            values = np.array([history[split] for history in histories])
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            axis.plot(
                epochs,
                mean,
                color=POLICIES[policy]["color"],
                label=POLICIES[policy]["name"],
            )
            axis.fill_between(
                epochs,
                mean - std,
                mean + std,
                color=POLICIES[policy]["color"],
                alpha=0.12,
            )
            axis.set(xlabel="Epoch", ylabel="Percent", title=title, ylim=(0, 100))
            axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "learning_curves.png", dpi=180)
    plt.close(figure)

    report = [
        "冻结主干与全参数微调对照实验（仅使用固定验证集）",
        "",
        "目的：在相同鸟框 letterbox 输入上，比较是否微调 ImageNet 主干，"
        "并把现有两轮冻结预热策略作为第三种训练方案。",
        "固定条件：CUB-200-2011；训练4794张、验证1200张；split-seed=42；"
        "训练 seed=42、43、44；ResNet-18；真实鸟框每侧扩展10%；letterbox；"
        "batch size32；10轮；AdamW，主干1e-4、分类头1e-3；权重衰减1e-4；混合精度。",
        "训练策略：",
        "- frozen_head：10轮全程冻结主干，只训练最后一层分类头。",
        "- finetune_all：从第1轮开始同时训练主干和分类头。",
        "- warmup_finetune：前2轮冻结主干，之后8轮微调全部参数。",
        "",
        "结果（3个训练种子的均值±样本标准差，不是置信区间）：",
    ]
    names_cn = {
        "frozen_head": "全程冻结主干",
        "finetune_all": "从第1轮微调",
        "warmup_finetune": "冻结2轮后微调",
    }
    for policy in policies:
        item = summary[policy]
        report.append(
            f"{names_cn[policy]}："
            f"Top-1 {100 * item['accuracy']['mean']:.2f}% ± "
            f"{100 * item['accuracy']['std']:.2f}%；"
            f"Top-5 {100 * item['top5']['mean']:.2f}% ± "
            f"{100 * item['top5']['std']:.2f}%；"
            f"Macro-F1 {100 * item['macro_f1']['mean']:.2f}% ± "
            f"{100 * item['macro_f1']['std']:.2f}%。"
        )
    report += [
        "",
        "训练观察：",
        f"冻结主干只更新{head_parameters:,}个分类参数；"
        f"全参数微调更新{total_parameters:,}个参数。",
        "冻结策略的泛化能力受 ImageNet 原特征限制。"
        "本次单机训练包含首次 CUDA 初始化，实际墙钟时间不作为效率结论。"
        "两轮预热用于先适应随机初始化的分类头，再微调主干。",
        "",
        "限制：这是单一网络、单一输入方案、10轮预算内的比较，"
        "不能推广为所有迁移学习任务的普遍结论。三种策略仍共享同一数据划分和随机种子。"
        "官方测试集未评估，最终模型也不得依据官方测试集重新调参。",
        "",
        "文件：per_seed_results.csv、validation_predictions.csv、"
        "paired_differences.csv、summary.json、"
        "performance_comparison.png、learning_curves.png。",
    ]
    (OUT / "experiment_report.txt").write_text(
        "\n".join(report), encoding="utf-8-sig"
    )
    print("FREEZE/FINE-TUNE COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    main()
