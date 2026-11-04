"""Qualitative and confusion analysis for the selected validation candidate."""
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
import torch.nn.functional as functional
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from plot_style import (
    BG,
    CORAL,
    GOLD,
    MUTED,
    PANEL,
    TEAL,
    TEXT,
    apply_report_style,
    page_header,
    save,
    style_axis,
)
from train import Birds, Letterbox, make_splits, read_pairs

DATA = ROOT / "data" / "CUB_200_2011"
RUN = ROOT / "runs" / "margin000_seed43"
OUT = ROOT / "results" / "final_visuals"
EXPECTED_SPLIT = (
    "40122ce37b1e511dcd2975be60d9b52bc99a4e2601d3ad310f588f6aed996df6"
)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_name(class_name):
    return class_name.split(".", 1)[-1].replace("_", " ")


def tensor_rgb(tensor):
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(array * STD + MEAN, 0, 1)


def load_model():
    checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint["split_sha256"] != EXPECTED_SPLIT:
        raise RuntimeError("Checkpoint split mismatch")
    expected = {
        "seed": 43,
        "split_seed": 42,
        "margin": 0.0,
        "crop": True,
        "resize_mode": "letterbox",
        "freeze_epochs": 0,
        "epochs": 10,
        "smoke": False,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Unexpected model config: {mismatches}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 200)
    model.load_state_dict(checkpoint["model"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    return model.to("cuda").eval(), checkpoint


def evaluate(model, rows):
    loader = DataLoader(
        Birds(DATA, rows, False, True, 0.0, "letterbox"),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    count = correct = top5 = 0
    confusion = np.zeros((200, 200), dtype=np.int64)
    predictions = []
    with torch.inference_mode():
        for images, labels, image_ids in loader:
            images = images.to("cuda")
            probabilities = model(images).softmax(dim=1).float().cpu()
            top_probabilities, top_indices = probabilities.topk(5, dim=1)
            for i, image_id in enumerate(image_ids):
                label = int(labels[i])
                predicted = int(top_indices[i, 0])
                correct += label == predicted
                top5 += label in top_indices[i].tolist()
                count += 1
                confusion[label, predicted] += 1
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "true_class": label,
                        "predicted_class": predicted,
                        "correct": int(label == predicted),
                        "true_probability": float(probabilities[i, label]),
                        "predicted_probability": float(top_probabilities[i, 0]),
                        "top5_correct": int(label in top_indices[i].tolist()),
                    }
                )
    accuracy = correct / count
    macro_f1 = float(
        (
            2
            * np.diag(confusion).astype(float)
            / np.maximum(
                confusion.sum(axis=1) + confusion.sum(axis=0), 1
            )
        ).mean()
    )
    return {
        "accuracy": accuracy,
        "top5": top5 / count,
        "macro_f1": macro_f1,
        "confusion": confusion,
        "predictions": predictions,
    }


def select_cases(rows, predictions):
    prediction_by_id = {row["image_id"]: row for row in predictions}
    row_by_id = {row["id"]: row for row in rows}
    areas = {}
    for row in rows:
        with Image.open(DATA / "images" / row["path"]) as image:
            areas[row["id"]] = (
                row["bbox"][2] * row["bbox"][3] / (image.width * image.height)
            )
    smallest = min(rows, key=lambda row: areas[row["id"]])
    largest = max(rows, key=lambda row: areas[row["id"]])
    correct_medium = next(
        row
        for row in sorted(rows, key=lambda row: areas[row["id"]])
        if prediction_by_id[row["id"]]["correct"]
        and row["id"] not in {smallest["id"], largest["id"]}
    )
    errors = sorted(
        (row for row in rows if not prediction_by_id[row["id"]]["correct"]),
        key=lambda row: prediction_by_id[row["id"]]["true_probability"],
    )[:2]
    selected = [
        (smallest, "Smallest bird box"),
        (correct_medium, "Representative correct"),
        (largest, "Largest bird box"),
        (errors[0], "High-confidence error"),
        (errors[1], "Second high-confidence error"),
    ]
    return selected, row_by_id, prediction_by_id


def make_confusion_figure(confusion, classes):
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
    pairs.sort(
        key=lambda row: (
            -row["count"],
            row["true_class_0based"],
            row["predicted_class_0based"],
        )
    )
    pairs = pairs[:10]
    write_csv(OUT / "top10_confusion_pairs.csv", pairs)

    pairs = list(reversed(pairs))
    labels = [
        f"{display_name(row['true_class'])}  →  "
        f"{display_name(row['predicted_class'])}"
        for row in pairs
    ]
    values = [row["count"] for row in pairs]
    figure, axis = plt.subplots(figsize=(9.8, 5.3))
    figure.subplots_adjust(top=0.74, bottom=0.14, left=0.38, right=0.97)
    page_header(
        figure,
        "Selected model: confusion pairs",
        "Seed 43 · margin 0% · fixed validation split",
    )
    bars = axis.barh(labels, values, color=["#BDD1DF"] * len(values), height=0.56)
    bars[-1].set_color(GOLD)
    for bar, value in zip(bars, values):
        axis.text(
            value + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            color=TEXT,
            fontsize=9,
        )
    axis.set_xlabel("Validation errors")
    axis.set_xlim(0, max(values) + 0.6)
    style_axis(axis)
    save(figure, OUT / "top10_confusion_pairs.png")
    plt.close(figure)
    return pairs


def make_gradcam(
    model,
    selected,
    prediction_by_id,
    classes,
    output_name="gradcam_examples.png",
    figure_width=9.4,
    row_height=3.0,
    title="Selected model: what drives the prediction?",
    metadata_name="gradcam_examples.csv",
    top_margin=0.88,
):
    activations, gradients = [], []

    def forward_hook(module, inputs, output):
        activations.clear()
        gradients.clear()
        output.register_hook(lambda gradient: gradients.append(gradient))
        activations.append(output)

    handle = model.layer4.register_forward_hook(forward_hook)
    figure, axes = plt.subplots(
        len(selected), 2, figsize=(figure_width, row_height * len(selected))
    )
    figure.patch.set_facecolor(BG)
    figure.subplots_adjust(
        top=top_margin,
        bottom=0.035,
        left=0.18,
        right=0.97,
        hspace=0.18,
        wspace=0.05,
    )
    page_header(
        figure,
        title,
        "Grad-CAM for the predicted class · exact bird box · seed 43",
    )
    metadata = []
    try:
        for i, (row, role) in enumerate(selected):
            model.zero_grad(set_to_none=True)
            tensor, _, _ = Birds(
                DATA, [row], False, True, 0.0, "letterbox"
            )[0]
            image = tensor.unsqueeze(0).to("cuda")
            logits = model(image)
            predicted = int(logits[0].argmax())
            logits[0, predicted].backward()
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            cam = functional.relu(
                (weights * activations[0]).sum(dim=1, keepdim=True)
            )
            cam = functional.interpolate(
                cam,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            cam -= cam.min()
            cam /= cam.max().clamp_min(1e-8)
            heatmap = plt.cm.inferno(cam.detach().cpu().numpy())[..., :3]
            rgb = tensor_rgb(tensor)
            overlay = np.clip(0.60 * rgb + 0.40 * heatmap, 0, 1)
            axes[i, 0].imshow(rgb)
            axes[i, 1].imshow(overlay)
            for axis in axes[i]:
                axis.axis("off")
            correct = predicted == row["label"]
            status = "Correct" if correct else "Incorrect"
            axes[i, 0].set_title("Network input", fontsize=9, color=MUTED, pad=6)
            axes[i, 1].set_title("Grad-CAM", fontsize=9, color=MUTED, pad=6)
            axes[i, 0].text(
                -0.08,
                0.5,
                f"{role}\n"
                f"{display_name(classes[row['label']])}\n"
                f"{status} · {display_name(classes[predicted])}\n"
                f"{prediction_by_id[row['id']]['predicted_probability']:.3f}",
                transform=axes[i, 0].transAxes,
                ha="right",
                va="center",
                color=TEAL if correct else CORAL,
                fontsize=8.5,
                linespacing=1.3,
            )
            metadata.append(
                {
                    "role": role,
                    "image_id": row["id"],
                    "path": row["path"],
                    "true_class": classes[row["label"]],
                    "predicted_class": classes[predicted],
                    "correct": int(correct),
                    "predicted_probability": prediction_by_id[row["id"]][
                        "predicted_probability"
                    ],
                }
            )
    finally:
        handle.remove()
    save(figure, OUT / output_name)
    plt.close(figure)
    if metadata_name:
        write_csv(OUT / metadata_name, metadata)
    return metadata


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    apply_report_style()
    rows = make_splits(DATA, 42)["val"]
    classes = list(read_pairs(DATA / "classes.txt").values())
    model, checkpoint = load_model()
    evaluation = evaluate(model, rows)
    confusion = evaluation.pop("confusion")
    predictions = evaluation.pop("predictions")
    saved = json.loads((RUN / "validation_metrics.json").read_text())
    for metric in ["accuracy", "top5", "macro_f1"]:
        if abs(saved[metric] - evaluation[metric]) > 1e-7:
            raise RuntimeError(f"Saved metric mismatch: {metric}")
    pairs = make_confusion_figure(confusion, classes)
    selected, _, prediction_by_id = select_cases(rows, predictions)
    selected_metadata = make_gradcam(
        model, selected, prediction_by_id, classes
    )
    make_gradcam(
        model,
        [selected[0], selected[2], selected[3]],
        prediction_by_id,
        classes,
        output_name="gradcam_compact.png",
        figure_width=7.8,
        row_height=2.45,
        title="Selected model: Grad-CAM examples",
        metadata_name=None,
        top_margin=0.82,
    )
    summary = {
        "selected_run": str(RUN.relative_to(ROOT)),
        "selection_basis": "Best mean validation Top-1 margin; seed 43 best checkpoint",
        "seed": 43,
        "margin": 0.0,
        "checkpoint_epoch": checkpoint["epoch"],
        "validation": evaluation,
        "top10_confusion_pairs": pairs,
        "selected_cases": selected_metadata,
        "official_test_evaluated": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("FINAL VISUALS COMPLETE", json.dumps(evaluation), flush=True)


if __name__ == "__main__":
    main()
