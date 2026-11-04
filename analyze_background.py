"""Validation-only background analysis for the bird-region letterbox models."""
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from train import Birds, Letterbox, make_splits, read_pairs

OUT = ROOT / "runs" / "background_analysis"
DATA = ROOT / "data" / "CUB_200_2011"
SEEDS = [42, 43, 44]
RUNS = {
    42: ROOT / "runs" / "milestone_letterbox_3060",
    43: ROOT / "runs" / "crop_letterbox_3060_seed43",
    44: ROOT / "runs" / "crop_letterbox_3060_seed44",
}
CONDITIONS = OrderedDict(
    [
        ("baseline", "Unmodified bird-region letterbox"),
        ("background_blur", "Bird clear; context Gaussian blurred"),
        ("background_occluded", "Bird clear; context filled"),
        ("background_only", "Bird masked; context retained"),
    ]
)
FILL = (124, 116, 104)
BLUR_RADIUS = 12
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_display(name):
    return name.split(".", 1)[-1].replace("_", " ")


def expanded_bounds(row, image, margin=0.1):
    x, y, width, height = row["bbox"]
    return (
        max(0, x - width * margin),
        max(0, y - height * margin),
        min(image.width, x + width * (1 + margin)),
        min(image.height, y + height * (1 + margin)),
    )


def tensor_to_rgb(tensor):
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(array * STD + MEAN, 0, 1)


class BackgroundDataset(Dataset):
    def __init__(self, rows, condition, margin=0.1):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        self.rows = rows
        self.condition = condition
        self.margin = margin
        self.transform = transforms.Compose(
            [
                Letterbox(224),
                transforms.ToTensor(),
                transforms.Normalize(MEAN.tolist(), STD.tolist()),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def make_image(self, row):
        with Image.open(DATA / "images" / row["path"]) as source:
            image = source.convert("RGB")
        bounds = expanded_bounds(row, image, self.margin)
        expanded = image.crop(bounds)
        if self.condition == "baseline":
            return expanded

        crop_left, crop_top, crop_right, crop_bottom = bounds
        x, y, width, height = row["bbox"]
        tight = (
            max(0.0, x - crop_left),
            max(0.0, y - crop_top),
            min(float(expanded.width), x + width - crop_left),
            min(float(expanded.height), y + height - crop_top),
        )
        tight = (
            int(np.floor(tight[0])),
            int(np.floor(tight[1])),
            max(int(np.floor(tight[0])) + 1, int(np.ceil(tight[2]))),
            max(int(np.floor(tight[1])) + 1, int(np.ceil(tight[3]))),
        )
        bird = expanded.crop(tight)

        if self.condition == "background_blur":
            modified = expanded.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
            modified.paste(bird, tight[:2])
            return modified
        if self.condition == "background_occluded":
            modified = Image.new("RGB", expanded.size, FILL)
            modified.paste(bird, tight[:2])
            return modified
        if self.condition == "background_only":
            modified = expanded.copy()
            ImageDraw.Draw(modified).rectangle(tight, fill=FILL)
            return modified
        raise AssertionError(self.condition)

    def __getitem__(self, index):
        row = self.rows[index]
        image = self.make_image(row)
        return self.transform(image), row["label"], row["id"]


def load_model(seed):
    path = RUNS[seed]
    checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if (
        config["seed"] != seed
        or config["split_seed"] != 42
        or not config["crop"]
        or config["resize_mode"] != "letterbox"
        or config["smoke"]
    ):
        raise RuntimeError(f"Unexpected checkpoint configuration: {path}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 200)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint, config


def calculate_metrics(predictions):
    count = len(predictions)
    confusion = np.zeros((200, 200), dtype=np.int64)
    for row in predictions:
        confusion[row["true_class_0based"], row["predicted_class_0based"]] += 1
    true_positive = np.diag(confusion).astype(float)
    actual = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    f1 = 2 * true_positive / np.maximum(actual + predicted, 1)
    return {
        "accuracy": float(true_positive.sum() / count),
        "top5": float(np.mean([row["top5_correct"] for row in predictions])),
        "macro_f1": float(f1.mean()),
    }


def evaluate(model, rows, condition, device):
    model.eval()
    loader = DataLoader(
        BackgroundDataset(rows, condition),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    output = []
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", enabled=device.type == "cuda"
    ):
        for images, labels, image_ids in loader:
            images = images.to(device, non_blocking=True)
            probabilities = model(images).softmax(dim=1).float().cpu()
            labels_cpu = labels.cpu()
            top_probabilities, top_indices = probabilities.topk(5, dim=1)
            for index, image_id in enumerate(image_ids):
                label = int(labels_cpu[index])
                prediction = int(top_indices[index, 0])
                output.append(
                    {
                        "image_id": int(image_id),
                        "true_class_0based": label,
                        "predicted_class_0based": prediction,
                        "correct": int(label == prediction),
                        "top5_correct": int(
                            label in top_indices[index].tolist()
                        ),
                        "predicted_class_probability": float(
                            top_probabilities[index, 0]
                        ),
                        "true_class_probability": float(probabilities[index, label]),
                        "top5_classes": " ".join(
                            str(int(value)) for value in top_indices[index]
                        ),
                        "top5_probabilities": " ".join(
                            f"{float(value):.8f}"
                            for value in top_probabilities[index]
                        ),
                    }
                )
    metrics = calculate_metrics(output)
    return metrics, output


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def make_performance_figure(summary):
    labels = ["Baseline", "Blur", "Occluded", "Bird masked"]
    keys = list(CONDITIONS)
    colors = ["#4477aa", "#66ccee", "#ee7733", "#aa3377"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in zip(
        axes,
        ["accuracy", "top5", "macro_f1"],
        ["Top-1 accuracy", "Top-5 accuracy", "Macro-F1"],
    ):
        axis.bar(
            labels,
            [100 * summary[key][metric]["mean"] for key in keys],
            yerr=[100 * summary[key][metric]["std"] for key in keys],
            color=colors,
            capsize=4,
        )
        axis.set(title=title, ylabel="Percent", ylim=(0, 100))
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Validation-only background interventions (mean and sample SD, 3 seeds)"
    )
    figure.tight_layout()
    figure.savefig(OUT / "background_performance.png", dpi=180)
    plt.close(figure)


def make_size_figure(group_rows):
    keys = list(CONDITIONS)
    labels = ["Baseline", "Blur", "Occluded", "Bird masked"]
    colors = ["#4477aa", "#66ccee", "#ee7733", "#aa3377"]
    positions = np.arange(len(group_rows))
    width = 0.19
    figure, axis = plt.subplots(figsize=(11, 5))
    for index, (key, label, color) in enumerate(zip(keys, labels, colors)):
        axis.bar(
            positions + (index - 1.5) * width,
            [100 * row[f"{key}_mean"] for row in group_rows],
            width,
            yerr=[100 * row[f"{key}_std"] for row in group_rows],
            color=color,
            label=label,
            capsize=2,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{row['bird_box_area_group']}\n(n={row['images']})"
            for row in group_rows
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
    figure.savefig(OUT / "background_by_bird_size.png", dpi=180)
    plt.close(figure)


def make_confusion_output(baseline, classes):
    confusion = np.zeros((200, 200), dtype=np.int64)
    for row in baseline:
        confusion[row["true_class_0based"], row["predicted_class_0based"]] += 1
    pairs = []
    for true_class in range(200):
        for predicted_class in range(200):
            if true_class == predicted_class:
                continue
            count = int(confusion[true_class, predicted_class])
            if count:
                pairs.append(
                    {
                        "true_class_0based": true_class,
                        "predicted_class_0based": predicted_class,
                        "true_class": classes[true_class],
                        "predicted_class": classes[predicted_class],
                        "count": count,
                    }
                )
    pairs.sort(key=lambda row: (-row["count"], row["true_class"], row["predicted_class"]))
    pairs = pairs[:10]
    write_csv(OUT / "top10_confusion_pairs.csv", pairs)

    figure, axis = plt.subplots(figsize=(10, 5))
    labels = [
        f"{class_display(row['true_class'])} -> "
        f"{class_display(row['predicted_class'])}"
        for row in reversed(pairs)
    ]
    values = [row["count"] for row in reversed(pairs)]
    axis.barh(labels, values, color="#4477aa")
    axis.set(xlabel="Validation errors (seed 42)", title="Top 10 confusion pairs")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUT / "top10_confusion_pairs.png", dpi=180)
    plt.close(figure)
    return pairs


def select_case_rows(rows, baseline_by_id, whole_predictions, areas):
    smallest = min(rows, key=lambda row: areas[row["id"]])
    largest = max(rows, key=lambda row: areas[row["id"]])
    improved = next(
        (
            row
            for row in rows
            if whole_predictions[row["id"]] != row["label"]
            and baseline_by_id[row["id"]]["predicted_class_0based"] == row["label"]
        ),
        None,
    )
    degraded = next(
        (
            row
            for row in rows
            if whole_predictions[row["id"]] == row["label"]
            and baseline_by_id[row["id"]]["predicted_class_0based"] != row["label"]
        ),
        None,
    )
    if improved is None or degraded is None:
        raise RuntimeError("Could not find both improvement and regression cases")
    chosen = [smallest, largest, improved, degraded]
    seen = set()
    unique = []
    for row in chosen:
        if row["id"] not in seen:
            unique.append(row)
            seen.add(row["id"])
    return unique, ["Smallest bird", "Largest bird", "Letterbox improvement", "Letterbox regression"]


def make_case_comparison(rows, labels, predictions, classes):
    figure, axes = plt.subplots(len(rows), 2, figsize=(8, 3.2 * len(rows)))
    for index, (row, role) in enumerate(zip(rows, labels)):
        for j, (mode, title) in enumerate(
            [
                ("full_letterbox", "Whole-image letterbox"),
                ("crop_letterbox", "Bird-region letterbox"),
            ]
        ):
            tensor, _, _ = Birds(
                DATA,
                [row],
                False,
                mode == "crop_letterbox",
                0.1,
                "letterbox",
            )[0]
            predicted = predictions[mode][row["id"]]
            axis = axes[index, j]
            axis.imshow(tensor_to_rgb(tensor))
            axis.axis("off")
            correctness = "correct" if predicted == row["label"] else "incorrect"
            axis.set_title(
                f"{title}\n{class_display(classes[predicted])} ({correctness})",
                fontsize=8,
                color="green" if predicted == row["label"] else "#bb2222",
            )
        axes[index, 0].text(
            -0.1,
            0.5,
            f"{role}\nTrue: {class_display(classes[row['label']])}",
            transform=axes[index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
            wrap=True,
        )
    figure.suptitle(
        "Seed 42: whole-image vs bird-region letterbox\n"
        "Selected smallest, largest, improved, and regressed cases"
    )
    figure.tight_layout(rect=(0.12, 0, 1, 0.95))
    figure.savefig(OUT / "case_comparison.png", dpi=170)
    plt.close(figure)


def make_gradcam(model, rows, labels, baseline_by_id, classes, device):
    dataset = BackgroundDataset(rows, "baseline")
    activations, gradients = [], []

    def forward_hook(module, inputs, output):
        activations.clear()
        gradients.clear()
        output.register_hook(lambda gradient: gradients.append(gradient))
        activations.append(output)

    handle = model.layer4.register_forward_hook(forward_hook)
    metadata = []
    figure, axes = plt.subplots(len(rows), 2, figsize=(8, 3.2 * len(rows)))
    try:
        for index, (row, role) in enumerate(zip(rows, labels)):
            model.zero_grad(set_to_none=True)
            tensor, _, _ = BackgroundDataset([row], "baseline")[0]
            image = tensor.unsqueeze(0).to(device)
            image.requires_grad_(False)
            logits = model(image)
            predicted = int(logits[0].argmax().item())
            logits[0, predicted].backward()
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            cam = functional.relu((weights * activations[0]).sum(dim=1, keepdim=True))
            cam = functional.interpolate(
                cam,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            cam -= cam.min()
            cam /= cam.max().clamp_min(1e-8)
            heatmap = plt.cm.jet(cam.detach().cpu().numpy())[..., :3]
            rgb = tensor_to_rgb(tensor)
            overlay = np.clip(0.52 * rgb + 0.48 * heatmap, 0, 1)
            axes[index, 0].imshow(rgb)
            axes[index, 0].axis("off")
            axes[index, 0].set_title("Network input", fontsize=8)
            axes[index, 1].imshow(overlay)
            axes[index, 1].axis("off")
            axes[index, 1].set_title("Grad-CAM: predicted class", fontsize=8)
            axes[index, 0].text(
                -0.08,
                0.5,
                f"{role}\nTrue: {class_display(classes[row['label']])}\n"
                f"Pred: {class_display(classes[predicted])}",
                transform=axes[index, 0].transAxes,
                ha="right",
                va="center",
                fontsize=8,
            )
            metadata.append(
                {
                    "role": role,
                    "image_id": row["id"],
                    "path": row["path"],
                    "true_class": classes[row["label"]],
                    "predicted_class": classes[predicted],
                    "correct": int(predicted == row["label"]),
                }
            )
    finally:
        handle.remove()
    figure.suptitle(
        "Seed 42 Grad-CAM: bird-region letterbox baseline\n"
        "Heatmaps explain the model's predicted class"
    )
    figure.tight_layout(rect=(0.13, 0, 1, 0.95))
    figure.savefig(OUT / "gradcam_examples.png", dpi=170)
    plt.close(figure)
    write_csv(OUT / "gradcam_examples.csv", metadata)


def make_change_examples(
    rows,
    top_changes,
    all_predictions,
    classes,
):
    selected = []
    for condition in ["background_blur", "background_occluded", "background_only"]:
        selected.extend(
            [
                row
                for row in top_changes
                if row["condition"] == condition
            ][:2]
        )
    row_by_id = {row["id"]: row for row in rows}
    figure, axes = plt.subplots(len(selected), 4, figsize=(11, 2.9 * len(selected)))
    for index, change in enumerate(selected):
        row = row_by_id[change["image_id"]]
        for j, condition in enumerate(CONDITIONS):
            tensor, _, _ = BackgroundDataset([row], condition)[0]
            prediction = next(
                item
                for item in all_predictions
                if item["seed"] == 42
                and item["condition"] == condition
                and item["image_id"] == row["id"]
            )
            axis = axes[index, j]
            axis.imshow(tensor_to_rgb(tensor))
            axis.axis("off")
            color = "green" if prediction["correct"] else "#bb2222"
            axis.set_title(
                f"{condition.replace('_', ' ')}\n"
                f"{class_display(classes[prediction['predicted_class_0based']])}\n"
                f"true p={prediction['true_class_probability']:.3f}",
                fontsize=7,
                color=color,
            )
    figure.suptitle(
        "Seed 42: examples with the largest true-class probability drops\n"
        "Two examples per background intervention"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(OUT / "background_shift_examples.png", dpi=170)
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the background analysis")
    device = torch.device("cuda")
    rows = make_splits(DATA, 42)["val"]
    classes = list(read_pairs(DATA / "classes.txt").values())
    areas = {}
    for row in rows:
        with Image.open(DATA / "images" / row["path"]) as image:
            areas[row["id"]] = (
                row["bbox"][2] * row["bbox"][3] / (image.width * image.height)
            )

    results = []
    all_predictions = []
    prediction_maps = {}
    for seed in SEEDS:
        model, _, _ = load_model(seed)
        model.to(device)
        for condition in CONDITIONS:
            metrics, predictions = evaluate(model, rows, condition, device)
            result = {"seed": seed, "condition": condition, **metrics}
            results.append(result)
            prediction_maps[(seed, condition)] = {
                row["image_id"]: row for row in predictions
            }
            for prediction in predictions:
                all_predictions.append(
                    {"seed": seed, "condition": condition, **prediction}
                )
            print("Validated:", result, flush=True)
        del model
        torch.cuda.empty_cache()

    baseline = prediction_maps[(42, "baseline")]
    for seed in SEEDS:
        saved = json.loads((RUNS[seed] / "validation_metrics.json").read_text())
        evaluated = next(
            row
            for row in results
            if row["seed"] == seed and row["condition"] == "baseline"
        )
        for metric in ["accuracy", "top5", "macro_f1"]:
            if abs(saved[metric] - evaluated[metric]) > 1e-7:
                raise RuntimeError(f"Baseline mismatch for seed {seed}: {metric}")

    write_csv(OUT / "per_seed_results.csv", results)
    write_csv(
        OUT / "validation_predictions.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"top5_classes", "top5_probabilities"}
            }
            for row in all_predictions
        ],
    )

    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for metric in ["accuracy", "top5", "macro_f1"]:
            values = [
                row[metric] for row in results if row["condition"] == condition
            ]
            mean, std = mean_std(values)
            summary[condition][metric] = {"mean": mean, "std": std}
    make_performance_figure(summary)

    paired = []
    for seed in SEEDS:
        for condition in list(CONDITIONS)[1:]:
            baseline_rows = prediction_maps[(seed, "baseline")]
            changed_rows = prediction_maps[(seed, condition)]
            fixed = sum(
                not baseline_rows[row["id"]]["correct"]
                and changed_rows[row["id"]]["correct"]
                for row in rows
            )
            broken = sum(
                baseline_rows[row["id"]]["correct"]
                and not changed_rows[row["id"]]["correct"]
                for row in rows
            )
            paired.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "accuracy_delta_percentage_points": 100 * (fixed - broken) / len(rows),
                    "fixed_errors": fixed,
                    "new_errors": broken,
                    "mean_true_probability_delta": float(
                        np.mean(
                            [
                                changed_rows[row["id"]]["true_class_probability"]
                                - baseline_rows[row["id"]]["true_class_probability"]
                                for row in rows
                            ]
                        )
                    ),
                }
            )
    write_csv(OUT / "paired_differences.csv", paired)

    group_rows = []
    for low, high, name in [
        (0, 0.1, "<10%"),
        (0.1, 0.3, "10-30%"),
        (0.3, 0.6, "30-60%"),
        (0.6, float("inf"), ">=60%"),
    ]:
        ids = [row["id"] for row in rows if low <= areas[row["id"]] < high]
        group = {"bird_box_area_group": name, "images": len(ids)}
        baseline_group_mean = np.mean(
            [
                np.mean(
                    [
                        prediction_maps[(seed, "baseline")][image_id]["correct"]
                        for image_id in ids
                    ]
                )
                for seed in SEEDS
            ]
        )
        for condition in CONDITIONS:
            values = [
                np.mean(
                    [
                        prediction_maps[(seed, condition)][image_id]["correct"]
                        for image_id in ids
                    ]
                )
                for seed in SEEDS
            ]
            mean, std = mean_std(values)
            group[f"{condition}_mean"] = mean
            group[f"{condition}_std"] = std
            group[f"{condition}_delta_vs_baseline"] = (
                mean - baseline_group_mean
            )
        group_rows.append(group)
    write_csv(OUT / "bird_size_analysis.csv", group_rows)
    make_size_figure(group_rows)

    confusion_pairs = make_confusion_output(
        [prediction_maps[(42, "baseline")][row["id"]] for row in rows],
        classes,
    )

    top_changes = []
    for seed in SEEDS:
        for condition in list(CONDITIONS)[1:]:
            for row in rows:
                baseline_row = prediction_maps[(seed, "baseline")][row["id"]]
                changed_row = prediction_maps[(seed, condition)][row["id"]]
                top_changes.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "image_id": row["id"],
                        "true_class_0based": row["label"],
                        "baseline_predicted_class_0based": baseline_row[
                            "predicted_class_0based"
                        ],
                        "changed_predicted_class_0based": changed_row[
                            "predicted_class_0based"
                        ],
                        "prediction_changed": int(
                            baseline_row["predicted_class_0based"]
                            != changed_row["predicted_class_0based"]
                        ),
                        "true_probability_drop": baseline_row[
                            "true_class_probability"
                        ]
                        - changed_row["true_class_probability"],
                        "baseline_true_probability": baseline_row[
                            "true_class_probability"
                        ],
                        "changed_true_probability": changed_row[
                            "true_class_probability"
                        ],
                    }
                )
    top_changes_by_condition = {}
    top_changes_export = []
    for condition in list(CONDITIONS)[1:]:
        candidates = sorted(
            [row for row in top_changes if row["condition"] == condition],
            key=lambda row: row["true_probability_drop"],
            reverse=True,
        )
        top_changes_by_condition[condition] = candidates[:20]
        top_changes_export.extend(candidates[:20])
    write_csv(OUT / "top_prediction_changes.csv", top_changes_export)

    whole_predictions = {}
    with (ROOT / "results" / "local_geometry_comparison" / "validation_predictions.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        for row in csv.DictReader(stream):
            if int(row["seed"]) == 42 and row["mode"] == "full_letterbox":
                whole_predictions[int(row["image_id"])] = int(
                    row["predicted_class_0based"]
                )
    case_rows, case_labels = select_case_rows(
        rows, baseline, whole_predictions, areas
    )
    comparison_predictions = {
        "full_letterbox": whole_predictions,
        "crop_letterbox": {
            image_id: row["predicted_class_0based"]
            for image_id, row in baseline.items()
        },
    }
    make_case_comparison(
        case_rows, case_labels, comparison_predictions, classes
    )

    seed42_model, _, _ = load_model(42)
    seed42_model.to(device).eval()
    make_gradcam(
        seed42_model,
        case_rows,
        case_labels,
        baseline,
        classes,
        device,
    )
    make_change_examples(
        rows, top_changes_export, all_predictions, classes
    )

    summary.update(
        {
            "conditions": CONDITIONS,
            "training_seeds": SEEDS,
            "split_seed": 42,
            "split_sha256": "40122ce37b1e511dcd2975be60d9b52bc99a4e2601d3ad310f588f6aed996df6",
            "train_images": 4794,
            "validation_images": 1200,
            "official_test_evaluated": False,
            "margin": 0.1,
            "blur_radius_original_pixels": BLUR_RADIUS,
            "occlusion_fill_rgb": list(FILL),
            "paired": paired,
            "bird_size_analysis": group_rows,
            "top10_confusion_pairs": confusion_pairs,
            "examples": {
                "case_rows": [
                    {
                        "role": role,
                        "image_id": row["id"],
                        "path": row["path"],
                        "true_class": classes[row["label"]],
                        "bird_box_area": areas[row["id"]],
                    }
                    for row, role in zip(case_rows, case_labels)
                ],
                "largest_prediction_changes": {
                    condition: rows[:10]
                    for condition, rows in top_changes_by_condition.items()
                },
            },
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    report = [
        "背景干预与可视化分析（仅使用固定验证集）",
        "",
        "设置：使用本机 RTX 3060 训练的鸟框 letterbox 模型，训练 seed=42、43、44；"
        "固定 split-seed=42；每次选择验证 Top-1 最佳 checkpoint。官方测试集未评估。",
        "鸟体区域：真实边界框；模型输入视野：边界框每侧扩展10%。所有条件统一 letterbox。",
        "background_blur：扩展框内部、紧贴鸟框外部的背景使用高斯模糊，半径12原图像素；鸟体保持原样。",
        "background_occluded：扩展框内部、紧贴鸟框外部的背景替换为RGB(124,116,104)；鸟体保持原样。",
        "background_only：紧贴鸟框内部的鸟体替换为RGB(124,116,104)，保留周围背景。",
        "",
        "结果（3个训练种子的均值±样本标准差，不是置信区间）：",
    ]
    for condition, description in CONDITIONS.items():
        report.append(
            f"{condition}："
            + "；".join(
                f"{title} {100 * summary[condition][metric]['mean']:.2f}% "
                f"± {100 * summary[condition][metric]['std']:.2f}%"
                for metric, title in [
                    ("accuracy", "Top-1"),
                    ("top5", "Top-5"),
                    ("macro_f1", "Macro-F1"),
                ]
            )
        )
    report += ["", "按鸟框面积分组的 Top-1 结果（3个种子平均）："]
    for group in group_rows:
        report.append(
            f"{group['bird_box_area_group']}（n={group['images']}）："
            f"baseline {100 * group['baseline_mean']:.2f}%；"
            f"blur {100 * group['background_blur_mean']:.2f}%"
            f"（{100 * group['background_blur_delta_vs_baseline']:+.2f}pp）；"
            f"occluded {100 * group['background_occluded_mean']:.2f}%"
            f"（{100 * group['background_occluded_delta_vs_baseline']:+.2f}pp）；"
            f"bird masked {100 * group['background_only_mean']:.2f}%"
            f"（{100 * group['background_only_delta_vs_baseline']:+.2f}pp）。"
        )
    report += [
        "",
        "限制：这些是分布变化测试而非重新训练；背景模糊和遮挡只覆盖模型输入视野内的背景。"
        "鸟体仍使用真实边界框。background-only 是辅助诊断，不能单独证明模型在正常输入上依赖背景。"
        "Top-10混淆类别对来自seed42验证集；每类只有6张验证图片，因此混淆计数很小，需谨慎解释。",
        "",
        "文件：summary.json、per_seed_results.csv、validation_predictions.csv、"
        "paired_differences.csv、bird_size_analysis.csv、top_prediction_changes.csv、"
        "top10_confusion_pairs.csv，以及六张分析图。",
    ]
    (OUT / "background_report.txt").write_text(
        "\n".join(report), encoding="utf-8-sig"
    )
    print("BACKGROUND ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
