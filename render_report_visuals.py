"""Render polished report figures from saved experiment data."""
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
from matplotlib.patches import FancyArrowPatch

from plot_style import (
    CORAL,
    GOLD,
    MUTED,
    NAVY,
    PALE_GOLD,
    PALE_NAVY,
    PALE_TEAL,
    PANEL,
    PURPLE,
    TEAL,
    TEXT,
    apply_report_style,
    direct_labels,
    page_header,
    panel_letter,
    save,
    solid_bar,
    style_axis,
)

RESULTS = ROOT / "results"
OUT = RESULTS / "report_visuals"
METRICS = [
    ("accuracy", "Top-1"),
    ("top5", "Top-5"),
    ("macro_f1", "Macro-F1"),
]


def read_json(relative):
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def read_csv(relative):
    with (RESULTS / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metric_values(summary, keys, metric):
    return np.array([100 * summary[key][metric]["mean"] for key in keys])


def metric_errors(summary, keys, metric):
    return np.array([100 * summary[key][metric]["std"] for key in keys])


def render_individual_metric(
    summary,
    keys,
    labels,
    colors,
    title,
    subtitle,
    output,
    highlight_index=None,
):
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.45))
    figure.subplots_adjust(top=0.73, bottom=0.20, left=0.07, right=0.985, wspace=0.28)
    page_header(figure, title, subtitle)
    x = np.arange(len(keys))
    for axis, (metric, metric_label), index in zip(
        axes, METRICS, range(len(METRICS))
    ):
        values = metric_values(summary, keys, metric)
        errors = metric_errors(summary, keys, metric)
        bar_colors = list(colors)
        if highlight_index is not None:
            bar_colors = [
                color if i == highlight_index else "#B9C3C6"
                for i, color in enumerate(bar_colors)
            ]
        solid_bar(axis, x, values, errors, bar_colors)
        direct_labels(axis, x, values, errors)
        axis.set_xticks(x, labels)
        axis.set_ylim(0, 100)
        axis.set_ylabel("Percent")
        panel_letter(axis, chr(65 + index), metric_label)
        style_axis(axis)
    save(figure, OUT / output)
    plt.close(figure)


def render_original_conditions():
    summary = read_json("letterbox_comparison/summary.json")
    render_individual_metric(
        summary,
        ["full", "crop", "letterbox"],
        ["Full image", "Center crop", "Letterbox"],
        [PALE_NAVY, PURPLE, GOLD],
        "Original input baseline",
        "Three matched conditions on one validation split · mean and sample SD over three seeds",
        "01_original_input_conditions.png",
        highlight_index=2,
    )


def render_local_control():
    summary = read_json("local_geometry_comparison/summary.json")
    render_individual_metric(
        summary,
        ["full_letterbox", "crop_letterbox"],
        ["Whole image", "Bird region"],
        [PALE_NAVY, TEAL],
        "Controlled test: whole image vs bird region",
        "Letterbox and augmentation held fixed · difference isolates the use of a bird box",
        "02_controlled_whole_vs_bird.png",
        highlight_index=1,
    )


def render_training_strategy():
    summary = read_json("freeze_finetune_comparison/summary.json")
    render_individual_metric(
        summary,
        ["frozen_head", "finetune_all", "warmup_finetune"],
        ["Frozen backbone", "Fine-tune from epoch 1", "Freeze 2, then fine-tune"],
        [PALE_NAVY, GOLD, TEAL],
        "Training strategy comparison",
        "All conditions use the same bird-region letterbox inputs and ten-epoch budget",
        "03_training_strategy.png",
        highlight_index=1,
    )


def render_margin_ablation():
    summary = read_json("margin_ablation/summary.json")
    keys = ["0.0", "0.05", "0.1", "0.2"]
    labels = ["0%", "5%", "10%", "20%"]
    x = np.arange(len(keys))
    values = metric_values(summary, keys, "accuracy")
    errors = metric_errors(summary, keys, "accuracy")
    area = summary["area_summary"]
    bird_share = np.array(
        [100 * item["bird_area_within_crop_mean"] for item in area]
    )

    figure, (axis, side) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.6),
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.45},
    )
    figure.subplots_adjust(top=0.76, bottom=0.10, left=0.11, right=0.96)
    page_header(
        figure,
        "How much context should the model see?",
        "Validation Top-1 and input composition as the bird box expands · three training seeds",
    )
    colors = [GOLD, TEAL, NAVY, CORAL]
    solid_bar(axis, x, values, errors, colors)
    direct_labels(axis, x, values, errors)
    axis.plot(x, values, color=TEXT, linewidth=1.1, alpha=0.55, zorder=4)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Validation Top-1 (%)")
    axis.set_ylim(65, 82)
    panel_letter(axis, "A", "Accuracy by box expansion")
    style_axis(axis)

    solid_bar(
        side,
        x,
        bird_share,
        np.zeros_like(bird_share),
        [PALE_GOLD, PALE_TEAL, PALE_NAVY, "#E8C8C3"],
        width=0.54,
    )
    for x_value, value in zip(x, bird_share):
        side.text(
            x_value,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=9,
            fontweight="medium",
        )
    side.set_xticks(x, labels)
    side.set_ylabel("Bird area within input (%)")
    side.set_ylim(0, 112)
    panel_letter(side, "B", "Share of input occupied by the bird box")
    style_axis(side)
    save(figure, OUT / "04_margin_ablation.png")
    plt.close(figure)


def render_background():
    summary = read_json("background_analysis/summary.json")
    keys = [
        "baseline",
        "background_blur",
        "background_occluded",
        "background_only",
    ]
    labels = ["Baseline", "Context blur", "Context occluded", "Bird masked"]
    values = metric_values(summary, keys, "accuracy")
    errors = metric_errors(summary, keys, "accuracy")
    x = np.arange(len(keys))
    figure, axis = plt.subplots(figsize=(9.3, 4.9))
    figure.subplots_adjust(top=0.72, bottom=0.22, left=0.11, right=0.97)
    page_header(
        figure,
        "What happens when context changes?",
        "Validation Top-1 for controlled background interventions · mean and sample SD",
    )
    colors = [NAVY, PALE_TEAL, PALE_GOLD, CORAL]
    solid_bar(axis, x, values, errors, colors)
    direct_labels(axis, x, values, errors)
    baseline = values[0]
    axis.annotate(
        f"−{baseline - values[1]:.2f} pp",
        xy=(1, values[1]),
        xytext=(1, values[1] - 7),
        ha="center",
        color=MUTED,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9},
    )
    axis.annotate(
        f"−{baseline - values[2]:.2f} pp",
        xy=(2, values[2]),
        xytext=(2, values[2] - 7),
        ha="center",
        color=MUTED,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9},
    )
    axis.annotate(
        "near chance-level",
        xy=(3, values[3] + errors[3]),
        xytext=(3, 14),
        ha="center",
        va="bottom",
        color=CORAL,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": CORAL, "lw": 0.9},
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Validation Top-1 (%)")
    axis.set_ylim(0, 82)
    style_axis(axis)
    save(figure, OUT / "05_background_interventions.png")
    plt.close(figure)


def render_background_sizes():
    summary = read_json("background_analysis/summary.json")
    rows = summary["bird_size_analysis"]
    keys = [
        "baseline",
        "background_blur",
        "background_occluded",
        "background_only",
    ]
    labels = ["Baseline", "Blur", "Occluded", "Bird masked"]
    matrix = np.array(
        [
            [100 * row[f"{key}_mean"] for key in keys]
            for row in rows
        ]
    )
    row_labels = [
        f"{row['bird_box_area_group'].replace('>=', '≥')}  (n={row['images']})"
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(8.8, 4.9))
    figure.subplots_adjust(top=0.72, bottom=0.15, left=0.20, right=0.97)
    page_header(
        figure,
        "Background effects across bird sizes",
        "Validation Top-1 by bird-box area · the masked-bird collapse is consistent",
    )
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=90, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="#FFFFFF" if matrix[i, j] > 45 else TEXT,
                fontsize=9,
                fontweight="medium",
            )
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.tick_params(length=0)
    axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    axis.grid(which="minor", color=PANEL, linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label("Top-1 (%)")
    colorbar.outline.set_visible(False)
    save(figure, OUT / "06_background_by_bird_size.png")
    plt.close(figure)


def render_confusions():
    rows = read_csv("background_analysis/top10_confusion_pairs.csv")

    def name(value):
        return value.split(".", 1)[-1].replace("_", " ")

    rows = list(reversed(rows))
    labels = [
        f"{name(row['true_class'])}  →  {name(row['predicted_class'])}"
        for row in rows
    ]
    values = [int(row["count"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9.8, 5.3))
    figure.subplots_adjust(top=0.74, bottom=0.14, left=0.38, right=0.97)
    page_header(
        figure,
        "Most frequent remaining confusions",
        "Seed 42 validation errors · many pairs are visually and taxonomically close",
    )
    bars = axis.barh(labels, values, color=[PALE_NAVY] * len(values), height=0.56)
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
    save(figure, OUT / "07_top10_confusions.png")
    plt.close(figure)


def render_overview():
    original = read_json("letterbox_comparison/summary.json")
    local = read_json("local_geometry_comparison/summary.json")
    training = read_json("freeze_finetune_comparison/summary.json")
    margin = read_json("margin_ablation/summary.json")
    background = read_json("background_analysis/summary.json")

    figure, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    figure.subplots_adjust(
        top=0.77, bottom=0.10, left=0.055, right=0.985, wspace=0.33, hspace=0.58
    )
    page_header(
        figure,
        "Bird recognition research snapshot",
        "CUB-200-2011 · ResNet-18 · fixed validation split · all intervals are sample SD over three training seeds",
    )

    axis = axes[0, 0]
    keys = ["full", "crop", "letterbox"]
    labels = ["Full", "Crop", "Letterbox"]
    x = np.arange(3)
    values = metric_values(original, keys, "accuracy")
    errors = metric_errors(original, keys, "accuracy")
    solid_bar(axis, x, values, errors, [PALE_NAVY, PURPLE, GOLD], width=0.56)
    direct_labels(axis, x, values, errors, fmt="{:.1f}")
    axis.set_xticks(x, labels)
    axis.set_ylim(55, 82)
    panel_letter(axis, "A", "Original inputs", "Top-1")
    style_axis(axis)

    axis = axes[0, 1]
    keys = ["full_letterbox", "crop_letterbox"]
    labels = ["Whole image", "Bird region"]
    x = np.arange(2)
    values = metric_values(local, keys, "accuracy")
    errors = metric_errors(local, keys, "accuracy")
    solid_bar(axis, x, values, errors, [PALE_NAVY, TEAL], width=0.48)
    direct_labels(axis, x, values, errors, fmt="{:.1f}")
    axis.add_patch(
        FancyArrowPatch(
            (0, values[0] + 5),
            (1, values[1] + 5),
            arrowstyle="-|>",
            mutation_scale=12,
            color=GOLD,
            lw=1.2,
        )
    )
    axis.text(
        0.5,
        (values[0] + values[1]) / 2 + 7,
        f"+{values[1] - values[0]:.1f} pp",
        ha="center",
        color=GOLD,
        fontsize=9,
        fontweight="medium",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(55, 82)
    panel_letter(axis, "B", "Controlled geometry", "Top-1")
    style_axis(axis)

    axis = axes[0, 2]
    keys = ["frozen_head", "finetune_all", "warmup_finetune"]
    labels = ["Frozen", "Fine-tune", "Warm-up"]
    x = np.arange(3)
    values = metric_values(training, keys, "accuracy")
    errors = metric_errors(training, keys, "accuracy")
    solid_bar(axis, x, values, errors, [PALE_NAVY, GOLD, TEAL], width=0.56)
    direct_labels(axis, x, values, errors, fmt="{:.1f}")
    axis.set_xticks(x, labels)
    axis.set_ylim(55, 82)
    panel_letter(axis, "C", "Training strategy", "Top-1")
    style_axis(axis)

    axis = axes[1, 0]
    keys = ["0.0", "0.05", "0.1", "0.2"]
    labels = ["0%", "5%", "10%", "20%"]
    x = np.arange(4)
    values = metric_values(margin, keys, "accuracy")
    errors = metric_errors(margin, keys, "accuracy")
    solid_bar(axis, x, values, errors, [GOLD, TEAL, NAVY, CORAL], width=0.56)
    direct_labels(axis, x, values, errors, fmt="{:.1f}")
    axis.set_xticks(x, labels)
    axis.set_ylim(55, 82)
    panel_letter(axis, "D", "Box expansion", "Top-1")
    style_axis(axis)

    axis = axes[1, 1]
    keys = ["baseline", "background_blur", "background_occluded"]
    labels = ["Base", "Blur", "Occlude"]
    x = np.arange(3)
    values = metric_values(background, keys, "accuracy")
    errors = metric_errors(background, keys, "accuracy")
    solid_bar(axis, x, values, errors, [NAVY, PALE_TEAL, PALE_GOLD], width=0.56)
    direct_labels(axis, x, values, errors, fmt="{:.1f}")
    axis.set_xticks(x, labels)
    axis.set_ylim(55, 82)
    panel_letter(axis, "E", "Context interventions", "Top-1")
    style_axis(axis)

    axis = axes[1, 2]
    masked = 100 * background["background_only"]["accuracy"]["mean"]
    axis.set_xlim(0, 82)
    axis.set_ylim(0, 1)
    axis.axis("off")
    panel_letter(axis, "F", "Bird masked", "Stress test")
    axis.add_patch(
        plt.Rectangle(
            (0.02, 0.18),
            0.96,
            0.62,
            transform=axis.transAxes,
            facecolor=PANEL,
            edgecolor="#E7D9D4",
            linewidth=1,
        )
    )
    axis.text(
        0.08,
        0.62,
        f"{masked:.1f}%",
        transform=axis.transAxes,
        color=CORAL,
        fontsize=27,
        fontweight="medium",
        va="center",
    )
    axis.text(
        0.08,
        0.43,
        "Top-1 with only context visible",
        transform=axis.transAxes,
        color=TEXT,
        fontsize=9,
        va="center",
    )
    axis.text(
        0.08,
        0.31,
        "Foreground is necessary in this diagnostic",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=8,
        va="center",
    )
    save(figure, OUT / "00_research_overview.png")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    apply_report_style()
    render_original_conditions()
    render_local_control()
    render_training_strategy()
    render_margin_ablation()
    render_background()
    render_background_sizes()
    render_confusions()
    render_overview()
    print(f"Rendered report visuals to {OUT}", flush=True)


if __name__ == "__main__":
    main()
