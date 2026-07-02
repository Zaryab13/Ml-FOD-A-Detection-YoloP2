"""
Generate corrected FOD-A object-scale statistics and figures.

Run manually from the repository root:
    python scripts/analysis/generate_dataset_scale_stats.py

This script intentionally separates:
- COCO area bins at original image resolution
- COCO area bins after YOLO letterbox resize
- legacy max-dimension bins used by the old Figure 3
- relative image-area bins, which are not a small-object definition
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "utils"))

from dataset_loader import FODDatasetLoader  # noqa: E402


DATASET_ROOT = REPO_ROOT / "data" / "FOD-A"
OUTPUT_DIR = REPO_ROOT / "results" / "scale_analysis"
IMGSZ = 640


def pct(count, total):
    return 100 * count / total if total else 0


def format_counts(counts, total):
    return [
        f"Small:  {counts['small']:,} ({pct(counts['small'], total):.1f}%)",
        f"Medium: {counts['medium']:,} ({pct(counts['medium'], total):.1f}%)",
        f"Large:  {counts['large']:,} ({pct(counts['large'], total):.1f}%)",
    ]


def plot_scale_bar(counts, total, title, subtitle, filename):
    labels = ["Small", "Medium", "Large"]
    values = [counts["small"], counts["medium"], counts["large"]]
    colors = ["#d95f59", "#e6a23c", "#3ca370"]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", fontsize=9)
    ax.set_ylabel("Number of Objects")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({pct(value, total):.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(split, stats):
    total = stats["total_annotations"]
    lines = [
        f"FOD-A scale statistics ({split})",
        f"Images: {stats['total_images']:,}",
        f"Objects: {total:,}",
        "",
        "COCO area bins at original resolution:",
        *format_counts(stats["coco_area_objects_original"], total),
        "",
        f"COCO area bins after {IMGSZ}x{IMGSZ} letterbox resize:",
        *format_counts(stats["coco_area_objects_letterbox"], total),
        "",
        "Legacy max-dimension bins at original resolution:",
        *format_counts(stats["max_dim_objects_original"], total),
        "",
        "Relative image-area cumulative bins:",
    ]

    for label, count in stats["relative_area_bins"].items():
        lines.append(f"{label}: {count:,} ({pct(count, total):.1f}%)")

    lines.extend(
        [
            "",
            "Interpretation:",
            "The <20% image-area statistic must be reported as a relative-area bin,",
            "not as evidence that the dataset is dominated by COCO small objects.",
        ]
    )

    (OUTPUT_DIR / f"{split}_scale_summary.txt").write_text("\n".join(lines) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = FODDatasetLoader(str(DATASET_ROOT))

    for split in ["train", "val"]:
        stats = loader.get_dataset_statistics(split, imgsz=IMGSZ)
        total = stats["total_annotations"]

        plot_scale_bar(
            stats["coco_area_objects_original"],
            total,
            f"COCO Object Scale Distribution ({split})",
            "Original image resolution; small <32^2 px, medium 32^2-96^2 px, large >=96^2 px",
            f"{split}_coco_area_original.png",
        )
        plot_scale_bar(
            stats["coco_area_objects_letterbox"],
            total,
            f"COCO Object Scale Distribution After Letterbox ({split})",
            f"After YOLO {IMGSZ}x{IMGSZ} letterbox resize",
            f"{split}_coco_area_letterbox_{IMGSZ}.png",
        )
        plot_scale_bar(
            stats["max_dim_objects_original"],
            total,
            f"Legacy Max-Dimension Scale Distribution ({split})",
            "Longest bbox side; included only to reproduce the old Figure 3 definition",
            f"{split}_legacy_max_dim_original.png",
        )
        write_summary(split, stats)

    print(f"Saved corrected scale analysis to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
