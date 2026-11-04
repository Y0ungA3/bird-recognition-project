"""Summarize and visualize the single frozen-model test evaluation."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import (
    CORAL,
    GOLD,
    MUTED,
    NAVY,
    PALE_GOLD,
    PALE_NAVY,
    PANEL,
    TEAL,
    TEXT,
    apply_report_style,
    direct_labels,
    page_header,
    save,
    solid_bar,
    style_axis,
)

ROOT = Path(__file__).resolve().parent
TEST_RUN = ROOT / "runs" / "final_test_margin000_seed43"
FROZEN_RUN = ROOT / "runs" / "final_candidate_margin000_seed43"
OUT = ROOT / "results" / "final_test_analysis"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_display(value):
    return value.split(".", 1)[-1].replace("_", " ")


def render_metric_comparison(validation, test):
    labels = ["Top-1", "Top-5", "Macro-F1"]
    keys = ["accuracy", "top5", "macro_f1"]
    val_values = np.array([100 * validation[key] for key in keys])
    test_values = np.array([100 * test[key] for key in keys])
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.4))
    figure.subplots_adjust(top=0.72, bottom=0.20, left=0.06, right=0.97, wspace=0.32)
    page_header(
        figure,
        "Frozen model: validation vs official test",
        "The selected model was not changed after opening the 5,794-image test set",
    )
    for i, (axis, label) in enumerate(zip(axes, labels)):
        values = np.array([val_values[i], test_values[i]])
        solid_bar(
            axis,
            np.arange(2),
            values,
            np.zeros_like(values),
            [TEAL, GOLD],
            width=0.52,
        )
        direct_labels(
            axis,
            np.arange(2),
            values,
            np.zeros_like(values),
            fmt="{:.2f}",
        )
        axis.set_xticks(np.arange(2), ["Validation", "Official test"])
        axis.set_ylim(0, 100)
        axis.set_ylabel("Percent")
        axis.set_title(label, loc="left", color=TEXT, pad=8)
        style_axis(axis)
    save(figure, OUT / "validation_vs_test.png")
    plt.close(figure)


def render_per_class(per_class):
    values = np.array([100 * row["accuracy"] for row in per_class])
    figure, axis = plt.subplots(figsize=(9.5, 4.7))
    figure.subplots_adjust(top=0.72, bottom=0.17, left=0.11, right=0.97)
    page_header(
        figure,
        "Test accuracy across 200 bird species",
        "Each class contains roughly 24–37 official test images; tails are descriptive, not variance estimates",
    )
    bins = np.linspace(0, 100, 21)
    axis.hist(
        values,
        bins=bins,
        color=PALE_NAVY,
        edgecolor=PANEL,
        linewidth=1.2,
        zorder=3,
    )
    median = float(np.median(values))
    axis.axvline(
        median,
        color=GOLD,
        linewidth=1.5,
        zorder=4,
        label=f"Median {median:.1f}%",
    )
    axis.text(
        median + 1.2,
        axis.get_ylim()[1] * 0.88,
        f"{median:.1f}%",
        color=GOLD,
        fontsize=9,
        va="top",
    )
    axis.set_xlabel("Per-class test accuracy (%)")
    axis.set_ylabel("Number of species")
    axis.set_xlim(0, 100)
    axis.legend(loc="upper left")
    style_axis(axis)
    save(figure, OUT / "per_class_accuracy.png")
    plt.close(figure)


def render_confusions(confusion_rows):
    rows = list(reversed(confusion_rows))
    labels = [
        f"{class_display(row['true_class'])}  →  "
        f"{class_display(row['predicted_class'])}"
        for row in rows
    ]
    values = [row["count"] for row in rows]
    figure, axis = plt.subplots(figsize=(9.8, 5.3))
    figure.subplots_adjust(top=0.74, bottom=0.14, left=0.38, right=0.97)
    page_header(
        figure,
        "Official test: most frequent confusions",
        "Frozen margin-0% model · counts are numbers of test images",
    )
    bars = axis.barh(labels, values, color=[PALE_NAVY] * len(values), height=0.56)
    bars[-1].set_color(GOLD)
    for bar, value in zip(bars, values):
        axis.text(
            value + 0.12,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            color=TEXT,
            fontsize=9,
        )
    axis.set_xlabel("Misclassified test images")
    axis.set_xlim(0, max(values) + 1.2)
    style_axis(axis)
    save(figure, OUT / "top_confusion_pairs.png")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    apply_report_style()
    validation = json.loads((FROZEN_RUN / "validation_metrics.json").read_text())
    test = json.loads((TEST_RUN / "test_metrics.json").read_text())
    confusion = np.loadtxt(
        TEST_RUN / "test_confusion_matrix.csv", delimiter=",", dtype=np.int64
    )
    classes = [
        line.split(maxsplit=1)[1]
        for line in (ROOT / "data" / "CUB_200_2011" / "classes.txt")
        .read_text()
        .splitlines()
    ]
    class_names = [class_display(name) for name in classes]

    total = confusion.sum()
    true_positive = np.diag(confusion).astype(float)
    actual = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    f1 = 2 * true_positive / np.maximum(actual + predicted, 1)

    per_class = []
    for index in range(200):
        per_class.append(
            {
                "class_0based": index,
                "class_name": classes[index],
                "images": int(actual[index]),
                "correct": int(true_positive[index]),
                "accuracy": float(true_positive[index] / max(actual[index], 1)),
                "f1": float(f1[index]),
            }
        )
    write_csv(OUT / "per_class_accuracy.csv", per_class)

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
    top_pairs = pairs[:10]
    write_csv(OUT / "top_confusion_pairs.csv", top_pairs)

    render_metric_comparison(validation, test)
    render_per_class(per_class)
    render_confusions(top_pairs)

    summary = {
        "model": "margin000_seed43",
        "selection_basis": "Validation only, before official test",
        "official_test_images": int(total),
        "validation": validation,
        "official_test": test,
        "absolute_validation_test_top1_delta_percentage_points": 100
        * (test["accuracy"] - validation["accuracy"]),
        "per_class_accuracy": {
            "mean": float(np.mean([row["accuracy"] for row in per_class])),
            "median": float(np.median([row["accuracy"] for row in per_class])),
            "min": float(min(row["accuracy"] for row in per_class)),
            "max": float(max(row["accuracy"] for row in per_class)),
        },
        "top10_confusion_pairs": top_pairs,
        "post_test_tuning": False,
        "official_test_evaluated": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("FINAL TEST ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
