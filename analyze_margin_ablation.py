"""Ablate the amount of context retained around the annotated bird box."""
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
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from train import Birds, make_splits, read_pairs, run_epoch

DATA = ROOT / "data" / "CUB_200_2011"
OUT = ROOT / "runs" / "margin_ablation"
SEEDS = [42, 43, 44]
MARGINS = [0.0, 0.05, 0.1, 0.2]
LABELS = ["0%", "5%", "10%", "20%"]
COLORS = ["#4477aa", "#66ccee", "#228833", "#ee7733"]
RUNS = {
    0.0: {
        seed: ROOT / "runs" / f"margin000_seed{seed}" for seed in SEEDS
    },
    0.05: {
        seed: ROOT / "runs" / f"margin005_seed{seed}" for seed in SEEDS
    },
    0.1: {
        seed: ROOT / "runs" / f"finetune_all_seed{seed}" for seed in SEEDS
    },
    0.2: {
        seed: ROOT / "runs" / f"margin020_seed{seed}" for seed in SEEDS
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


def load_checkpoint(margin, seed):
    path = RUNS[margin][seed]
    if not (path / "completed.json").exists():
        raise FileNotFoundError(f"Incomplete run: {path}")
    checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint.get("split_sha256") != EXPECTED_SPLIT:
        raise RuntimeError(f"Split mismatch in {path}")
    expected = {
        "seed": seed,
        "split_seed": 42,
        "epochs": 10,
        "batch_size": 32,
        "freeze_epochs": 0,
        "crop": True,
        "resize_mode": "letterbox",
        "smoke": False,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if abs(config.get("margin", -1) - margin) > 1e-12:
        mismatches["margin"] = (config.get("margin"), margin)
    if mismatches:
        raise RuntimeError(f"Configuration mismatch in {path}: {mismatches}")
    return path, checkpoint


def evaluate(path, checkpoint, rows):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 200)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model.to(device).eval()
    loader = DataLoader(
        Birds(
            DATA,
            rows,
            False,
            True,
            float(checkpoint["config"]["margin"]),
            "letterbox",
        ),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    with torch.inference_mode():
        metrics, _, predictions = run_epoch(
            model, loader, device, nn.CrossEntropyLoss()
        )
    (path / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    return metrics, predictions


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def crop_statistics(rows, margin):
    crop_fractions = []
    bird_fractions = []
    for row in rows:
        with Image.open(DATA / "images" / row["path"]) as image:
            x, y, width, height = row["bbox"]
            left = max(0, x - width * margin)
            top = max(0, y - height * margin)
            right = min(image.width, x + width * (1 + margin))
            bottom = min(image.height, y + height * (1 + margin))
            crop_area = (right - left) * (bottom - top)
            image_area = image.width * image.height
            crop_fractions.append(crop_area / image_area)
            bird_fractions.append(min(1.0, width * height / crop_area))
    return {
        "crop_area_fraction_mean": float(np.mean(crop_fractions)),
        "crop_area_fraction_std": float(np.std(crop_fractions, ddof=1)),
        "bird_area_within_crop_mean": float(np.mean(bird_fractions)),
        "bird_area_within_crop_std": float(np.std(bird_fractions, ddof=1)),
    }


def make_input_grid(rows, areas):
    ordered = sorted(rows, key=lambda row: areas[row["id"]])
    selected = [
        ordered[len(ordered) // 10],
        ordered[len(ordered) // 2],
        ordered[-max(1, len(ordered) // 10)],
    ]
    role_names = ["Smallest box", "Median box", "Largest box"]
    figure, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    for i, (row, role) in enumerate(zip(selected, role_names)):
        for j, margin in enumerate(MARGINS):
            tensor, _, _ = Birds(
                DATA, [row], False, True, margin, "letterbox"
            )[0]
            array = tensor.numpy().transpose(1, 2, 0)
            array = np.clip(
                array * np.array([0.229, 0.224, 0.225])
                + np.array([0.485, 0.456, 0.406]),
                0,
                1,
            )
            axes[i, j].imshow(array)
            axes[i, j].axis("off")
            axes[i, j].set_title(f"margin {100 * margin:.0f}%", fontsize=9)
    figure.suptitle(
        "Actual network inputs at four bird-box expansion margins\n"
        "Representative smallest, median and largest validation bird boxes"
    )
    figure.tight_layout(rect=(0.17, 0, 1, 0.95))
    for i, (row, role) in enumerate(zip(selected, role_names)):
        position = axes[i, 0].get_position()
        figure.text(
            0.16,
            (position.y0 + position.y1) / 2,
            f"{role}\nbox area {100 * areas[row['id']]:.1f}%",
            ha="right",
            va="center",
            fontsize=9,
        )
    figure.savefig(OUT / "margin_inputs.png", dpi=170)
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = make_splits(DATA, 42)["val"]
    labels = {row["id"]: row["label"] for row in rows}
    areas = {}
    for row in rows:
        with Image.open(DATA / "images" / row["path"]) as image:
            areas[row["id"]] = (
                row["bbox"][2] * row["bbox"][3] / (image.width * image.height)
            )

    results = []
    prediction_rows = []
    prediction_maps = {}
    for margin in MARGINS:
        for seed in SEEDS:
            path, checkpoint = load_checkpoint(margin, seed)
            metrics, predictions = evaluate(path, checkpoint, rows)
            result = {
                "margin": margin,
                "margin_percent": int(round(100 * margin)),
                "seed": seed,
                "run": str(path.relative_to(ROOT)),
                "best_epoch": int(checkpoint["epoch"]),
                **metrics,
            }
            results.append(result)
            prediction_maps[(margin, seed)] = {
                image_id: predicted
                for image_id, _, predicted in predictions
            }
            for image_id, true_class, predicted_class in predictions:
                prediction_rows.append(
                    {
                        "margin": margin,
                        "seed": seed,
                        "image_id": image_id,
                        "true_class_0based": true_class,
                        "predicted_class_0based": predicted_class,
                        "correct": int(true_class == predicted_class),
                    }
                )
            print("Evaluated:", result, flush=True)

    write_csv(OUT / "per_seed_results.csv", results)
    write_csv(OUT / "validation_predictions.csv", prediction_rows)
    if len(prediction_rows) != 4 * 3 * 1200:
        raise RuntimeError(f"Unexpected prediction count: {len(prediction_rows)}")

    summary = {}
    for margin in MARGINS:
        summary[str(margin)] = {}
        for metric in ["accuracy", "top5", "macro_f1"]:
            values = [
                row[metric] for row in results if row["margin"] == margin
            ]
            mean, std = mean_std(values)
            summary[str(margin)][metric] = {"mean": mean, "std": std}

    paired = []
    for margin in MARGINS:
        if margin == 0.1:
            continue
        for seed in SEEDS:
            fixed = sum(
                prediction_maps[(0.1, seed)][row["id"]] != row["label"]
                and prediction_maps[(margin, seed)][row["id"]] == row["label"]
                for row in rows
            )
            broken = sum(
                prediction_maps[(0.1, seed)][row["id"]] == row["label"]
                and prediction_maps[(margin, seed)][row["id"]] != row["label"]
                for row in rows
            )
            paired.append(
                {
                    "margin": margin,
                    "margin_percent": int(round(100 * margin)),
                    "seed": seed,
                    "margin_minus_10pct_accuracy_percentage_points": 100
                    * (fixed - broken)
                    / len(rows),
                    "fixed_errors": fixed,
                    "new_errors": broken,
                }
            )
    write_csv(OUT / "paired_differences.csv", paired)

    area_rows = []
    for margin in MARGINS:
        area_rows.append(
            {
                "margin": margin,
                "margin_percent": int(round(100 * margin)),
                **crop_statistics(rows, margin),
            }
        )
    write_csv(OUT / "area_summary.csv", area_rows)

    bird_size_rows = []
    for low, high, name in [
        (0, 0.1, "<10%"),
        (0.1, 0.3, "10-30%"),
        (0.3, 0.6, "30-60%"),
        (0.6, float("inf"), ">=60%"),
    ]:
        image_ids = [
            row["id"] for row in rows if low <= areas[row["id"]] < high
        ]
        group = {"bird_box_area_group": name, "images": len(image_ids)}
        for margin in MARGINS:
            values = [
                np.mean(
                    [
                        prediction_maps[(margin, seed)][image_id] == labels[image_id]
                        for image_id in image_ids
                    ]
                )
                for seed in SEEDS
            ]
            mean, std = mean_std(values)
            key = f"margin_{int(round(100 * margin))}"
            group[f"{key}_mean"] = mean
            group[f"{key}_std"] = std
        bird_size_rows.append(group)
    write_csv(OUT / "bird_size_analysis.csv", bird_size_rows)

    best_margin = max(
        MARGINS, key=lambda margin: summary[str(margin)]["accuracy"]["mean"]
    )
    summary.update(
        {
            "training_seeds": SEEDS,
            "split_seed": 42,
            "split_sha256": EXPECTED_SPLIT,
            "train_images": 4794,
            "validation_images": 1200,
            "official_test_evaluated": False,
            "best_margin_by_validation_top1": best_margin,
            "paired": paired,
            "area_summary": area_rows,
            "bird_size_analysis": bird_size_rows,
            "controlled_variables": (
                "ResNet-18 ImageNet; crop=True; letterbox; fine-tune from "
                "epoch 1; batch size 32; 10 epochs; same split and seeds."
            ),
        }
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in zip(
        axes,
        ["accuracy", "top5", "macro_f1"],
        ["Top-1 accuracy", "Top-5 accuracy", "Macro-F1"],
    ):
        axis.bar(
            LABELS,
            [100 * summary[str(margin)][metric]["mean"] for margin in MARGINS],
            yerr=[100 * summary[str(margin)][metric]["std"] for margin in MARGINS],
            color=COLORS,
            capsize=4,
        )
        axis.set(title=title, ylabel="Percent", ylim=(0, 100))
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Bird-box margin ablation (3 seeds, mean and sample SD)"
    )
    figure.tight_layout()
    figure.savefig(OUT / "performance_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(MARGINS))
    for index, row in enumerate(area_rows):
        axis.scatter(
            index,
            100 * row["bird_area_within_crop_mean"],
            color=COLORS[index],
            s=70,
        )
        axis.text(
            index,
            min(
                105,
                100 * row["bird_area_within_crop_mean"] + 1.5,
            ),
            f"{100 * row['bird_area_within_crop_mean']:.1f}%",
            ha="center",
            fontsize=9,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(LABELS)
    axis.set(
        xlabel="Box expansion margin",
        ylabel="Bird area within network input (%)",
        title="How much of the input remains bird vs context",
        ylim=(0, 108),
    )
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUT / "context_composition.png", dpi=180)
    plt.close(figure)

    positions = np.arange(len(bird_size_rows))
    width = 0.19
    figure, axis = plt.subplots(figsize=(11, 5))
    for index, margin in enumerate(MARGINS):
        key = f"margin_{int(round(100 * margin))}"
        axis.bar(
            positions + (index - 1.5) * width,
            [100 * row[f"{key}_mean"] for row in bird_size_rows],
            width,
            yerr=[100 * row[f"{key}_std"] for row in bird_size_rows],
            color=COLORS[index],
            label=f"Margin {LABELS[index]}",
            capsize=2,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{row['bird_box_area_group']}\n(n={row['images']})"
            for row in bird_size_rows
        ]
    )
    axis.set(
        xlabel="Bird box area / original image area",
        ylabel="Validation Top-1 (%)",
        ylim=(0, 100),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUT / "bird_size_comparison.png", dpi=180)
    plt.close(figure)

    make_input_grid(rows, areas)

    report = [
        "鸟框扩展范围消融实验（仅使用固定验证集）",
        "",
        "目的：在相同网络、数据划分和训练预算下，比较鸟框每侧扩展0%、5%、10%、20%时保留的鸟体与背景比例及识别性能。",
        "固定条件：ResNet-18 ImageNet预训练；训练4794张、验证1200张；split-seed=42；"
        "训练seed=42、43、44；从第1轮开始全参数微调；letterbox；水平翻转；"
        "batch size32；10轮；AdamW；主干1e-4、分类头1e-3；混合精度。",
        "margin定义：在真实边界框的每一侧额外扩展原框宽/高的给定比例，再裁出完整区域。",
        "",
        "结果（3个训练种子的均值±样本标准差，不是置信区间）：",
    ]
    for margin, label in zip(MARGINS, LABELS):
        item = summary[str(margin)]
        report.append(
            f"margin {label}："
            f"Top-1 {100 * item['accuracy']['mean']:.2f}% ± "
            f"{100 * item['accuracy']['std']:.2f}%；"
            f"Top-5 {100 * item['top5']['mean']:.2f}% ± "
            f"{100 * item['top5']['std']:.2f}%；"
            f"Macro-F1 {100 * item['macro_f1']['mean']:.2f}% ± "
            f"{100 * item['macro_f1']['std']:.2f}%。"
        )
    report += [
        "",
        f"验证集Top-1平均最高的范围为 margin {100 * best_margin:.0f}%。",
        "相对10%条件的逐种子Top-1差值：",
    ]
    for margin, label in zip(MARGINS, LABELS):
        if margin == 0.1:
            continue
        values = [
            row["margin_minus_10pct_accuracy_percentage_points"]
            for row in paired
            if row["margin"] == margin
        ]
        report.append(
            f"margin {label}："
            + "、".join(
                f"seed {seed} {value:+.2f}"
                for seed, value in zip(SEEDS, values)
            )
            + f"；平均 {np.mean(values):+.2f} 个百分点。"
        )
    report += [
        "",
        "输入构成（验证集平均）：",
    ]
    for row in area_rows:
        report.append(
            f"margin {row['margin_percent']}%："
            f"裁剪区域占原图 {100 * row['crop_area_fraction_mean']:.2f}%，"
            f"鸟框占网络输入 {100 * row['bird_area_within_crop_mean']:.2f}%。"
        )
    report += [
        "",
        "限制：不同margin同时改变目标尺度、背景面积和有效像素，不能把差异单独归因于背景。"
        "选择范围时使用验证集；官方测试集尚未评估。三个种子仅提供初步稳定性证据。"
        "margin 0%也可能切掉边界框外的头、尾等部位，因为CUB边界框并非逐像素完美前景掩码。",
        "",
        "文件：summary.json、per_seed_results.csv、validation_predictions.csv、"
        "paired_differences.csv、area_summary.csv、bird_size_analysis.csv、"
        "performance_comparison.png、context_composition.png、"
        "bird_size_comparison.png、margin_inputs.png。",
    ]
    (OUT / "experiment_report.txt").write_text(
        "\n".join(report), encoding="utf-8-sig"
    )
    print("MARGIN ABLATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
