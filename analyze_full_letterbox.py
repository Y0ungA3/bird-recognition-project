"""Compare whole-image and bird-region letterbox runs on identical local hardware."""
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
from torch.utils.data import DataLoader
from torchvision import models

from train import Birds, make_splits, read_pairs, run_epoch

OUT = ROOT / "runs" / "local_geometry_comparison"
SEEDS = [42, 43, 44]
MODES = ["full_letterbox", "crop_letterbox"]
NAMES = ["Whole-image letterbox", "Bird-region letterbox"]
COLORS = ["#4477aa", "#228833"]
CROP_RUNS = {
    42: "milestone_letterbox_3060",
    43: "crop_letterbox_3060_seed43",
    44: "crop_letterbox_3060_seed44",
}


def run_dir(mode, seed):
    if mode == "full_letterbox":
        return ROOT / "runs" / f"full_letterbox_seed{seed}"
    return ROOT / "runs" / CROP_RUNS[seed]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data" / "CUB_200_2011"
    rows = make_splits(data, 42)["val"]
    labels = {row["id"]: row["label"] for row in rows}
    classes = list(read_pairs(data / "classes.txt").values())
    expected_split = "40122ce37b1e511dcd2975be60d9b52bc99a4e2601d3ad310f588f6aed996df6"
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the controlled comparison")

    results, predictions, environments = [], [], []
    for mode in MODES:
        for seed in SEEDS:
            path = run_dir(mode, seed)
            if not (path / "completed.json").exists():
                raise FileNotFoundError(f"Incomplete run: {path}")
            checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            environment = json.loads((path / "environment.json").read_text())
            if environment["split_sha256"] != expected_split:
                raise RuntimeError(f"Split mismatch: {path}")
            if checkpoint.get("split_sha256") != expected_split:
                raise RuntimeError(f"Checkpoint split mismatch: {path}")
            expected = {
                "seed": seed,
                "split_seed": 42,
                "resize_mode": "letterbox",
                "epochs": 10,
                "batch_size": 32,
                "margin": 0.1,
            }
            if any(config[key] != value for key, value in expected.items()):
                raise RuntimeError(f"Config mismatch: {path}")
            if config["crop"] != (mode == "crop_letterbox"):
                raise RuntimeError(f"Crop mode mismatch: {path}")

            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(512, 200)
            model.load_state_dict(checkpoint["model"])
            model.to(device).eval()
            loader = DataLoader(
                Birds(data, rows, False, config["crop"], 0.1, "letterbox"),
                batch_size=32,
                num_workers=0,
                pin_memory=True,
            )
            metrics, _, image_predictions = run_epoch(
                model, loader, device, torch.nn.CrossEntropyLoss()
            )
            result = {
                "seed": seed,
                "mode": mode,
                "best_epoch": checkpoint["epoch"],
                **metrics,
            }
            results.append(result)
            for image_id, true_class, predicted_class in image_predictions:
                predictions.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "image_id": image_id,
                        "true_class_0based": true_class,
                        "predicted_class_0based": predicted_class,
                        "correct": int(true_class == predicted_class),
                    }
                )
            environments.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "gpu": environment["gpu"],
                    "torch": environment["torch"],
                    "cuda": environment["cuda"],
                    "split_sha256": environment["split_sha256"],
                }
            )
            (path / "validation_metrics.json").write_text(json.dumps(result, indent=2))
            print("Validated:", result, flush=True)
            del model, loader
            torch.cuda.empty_cache()

    write_csv(OUT / "validation_results.csv", results)
    write_csv(OUT / "validation_predictions.csv", predictions)
    summary = {
        mode: {
            metric: {
                "mean": float(
                    np.mean([row[metric] for row in results if row["mode"] == mode])
                ),
                "std": float(
                    np.std(
                        [row[metric] for row in results if row["mode"] == mode],
                        ddof=1,
                    )
                ),
            }
            for metric in ["accuracy", "top5", "macro_f1"]
        }
        for mode in MODES
    }

    predicted = {
        (seed, mode): {
            row["image_id"]: row["predicted_class_0based"]
            for row in predictions
            if row["seed"] == seed and row["mode"] == mode
        }
        for seed in SEEDS
        for mode in MODES
    }
    paired = []
    for seed in SEEDS:
        fixed = sum(
            predicted[(seed, "full_letterbox")][row["id"]] != row["label"]
            and predicted[(seed, "crop_letterbox")][row["id"]] == row["label"]
            for row in rows
        )
        broken = sum(
            predicted[(seed, "full_letterbox")][row["id"]] == row["label"]
            and predicted[(seed, "crop_letterbox")][row["id"]] != row["label"]
            for row in rows
        )
        paired.append(
            {
                "seed": seed,
                "crop_minus_full_delta_percentage_points": 100
                * (fixed - broken)
                / len(rows),
                "fixed_errors": fixed,
                "new_errors": broken,
            }
        )
    write_csv(OUT / "paired_differences.csv", paired)

    areas = {}
    for row in rows:
        with Image.open(data / "images" / row["path"]) as image:
            areas[row["id"]] = row["bbox"][2] * row["bbox"][3] / (
                image.width * image.height
            )
    bird_size = []
    for low, high, name in [
        (0, 0.1, "<10%"),
        (0.1, 0.3, "10-30%"),
        (0.3, 0.6, "30-60%"),
        (0.6, float("inf"), ">=60%"),
    ]:
        image_ids = [row["id"] for row in rows if low <= areas[row["id"]] < high]
        size_row = {"bird_box_area_group": name, "images": len(image_ids)}
        for mode in MODES:
            values = [
                np.mean(
                    [
                        predicted[(seed, mode)][image_id] == labels[image_id]
                        for image_id in image_ids
                    ]
                )
                for seed in SEEDS
            ]
            size_row[f"{mode}_mean"] = float(np.mean(values))
            size_row[f"{mode}_std"] = float(np.std(values, ddof=1))
        bird_size.append(size_row)
    write_csv(OUT / "bird_size_analysis.csv", bird_size)

    summary.update(
        {
            "paired": paired,
            "bird_size_analysis": bird_size,
            "training_seeds": SEEDS,
            "split_seed": 42,
            "split_sha256": expected_split,
            "official_test_evaluated": False,
            "train_images": 4794,
            "validation_images": 1200,
            "environments": environments,
            "experiment": (
                "Both conditions use whole-region letterbox geometry and only "
                "horizontal-flip augmentation. The crop condition additionally "
                "uses the ground-truth box expanded by 10%."
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
            NAMES,
            [100 * summary[mode][metric]["mean"] for mode in MODES],
            yerr=[100 * summary[mode][metric]["std"] for mode in MODES],
            color=COLORS,
            capsize=4,
        )
        axis.set(title=title, ylabel="Percent", ylim=(0, 100))
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "RTX 3060 local control: mean and sample SD across three training seeds"
    )
    figure.tight_layout()
    figure.savefig(OUT / "performance_comparison.png", dpi=180)
    plt.close(figure)

    lines = [
        "整图与鸟框 letterbox 的本地控制实验",
        "",
        "设置：RTX 3060 Laptop GPU 6GB；固定 split-seed=42；训练 seed=42、43、44；"
        "ResNet-18；训练4794张、验证1200张；10轮，前2轮冻结主干；batch32；"
        "两种条件都使用最长边224等比例缩放、居中填充和水平翻转，验证不再中心裁剪。",
        "唯一输入差异：鸟框条件使用真实边界框并每侧扩展10%；整图条件直接使用完整原图。",
        "",
        "结果（3个训练种子的均值±样本标准差，不是置信区间）：",
    ]
    for mode, name in zip(MODES, ["整图 letterbox", "鸟框 letterbox"]):
        lines.append(
            name
            + "："
            + "；".join(
                f"{title} {100 * summary[mode][metric]['mean']:.2f}% "
                f"± {100 * summary[mode][metric]['std']:.2f}%"
                for metric, title in [
                    ("accuracy", "Top-1"),
                    ("top5", "Top-5"),
                    ("macro_f1", "Macro-F1"),
                ]
            )
        )
    delta = 100 * (
        summary["crop_letterbox"]["accuracy"]["mean"]
        - summary["full_letterbox"]["accuracy"]["mean"]
    )
    lines += [
        f"鸟框相对整图 letterbox 的平均 Top-1 差值：{delta:+.2f}个百分点。",
        "",
        "限制：鸟框条件依赖额外真实位置标注；它同时改变目标尺度和背景内容。"
        "该实验只能说明在当前预处理管线中定位鸟框是否有帮助，不能单独证明背景的因果作用。"
        "官方测试集未评估。",
        "",
        "文件：summary.json、validation_results.csv、validation_predictions.csv、"
        "paired_differences.csv、bird_size_analysis.csv、performance_comparison.png。",
    ]
    (OUT / "experiment_report.txt").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )
    print("LOCAL GEOMETRY COMPARISON COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
