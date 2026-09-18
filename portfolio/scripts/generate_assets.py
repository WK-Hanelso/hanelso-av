"""Generate public-safe portfolio figures from repository-documented evidence.

No model, pipeline, or dataset job is run here. Numeric inputs below are copied
from the source locations listed in portfolio/EVIDENCE.md.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

INK = "#172033"
MUTED = "#5B6475"
BLUE = "#1976D2"
BLUE_DARK = "#0D47A1"
BLUE_PALE = "#EAF3FC"
GREEN = "#15805D"
GREEN_PALE = "#E9F7F1"
AMBER = "#C77800"
AMBER_PALE = "#FFF4DD"
RED = "#B23A48"
GRAY = "#E7EAF0"


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "svg.fonttype": "none",
        }
    )
    ASSETS.mkdir(parents=True, exist_ok=True)


def rounded_box(ax, xy, width, height, text, face, edge, fontsize=16, weight="bold"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.8,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.3,
    )


def save(fig, stem: str) -> None:
    svg_path = ASSETS / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(ASSETS / f"{stem}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ownership_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((0.02, 0.47), 0.96, 0.47, facecolor=BLUE_PALE, edgecolor="none"))
    ax.add_patch(Rectangle((0.02, 0.05), 0.96, 0.38, facecolor=GREEN_PALE, edgecolor="none"))
    ax.text(0.02, 0.965, "Data owns shared facts + validation; model-specific tensorization stays with the consumer.", fontsize=14, color=MUTED)
    ax.text(0.055, 0.905, "DATA OWNERSHIP", fontsize=20, fontweight="bold", color=BLUE_DARK)
    ax.text(0.055, 0.415, "MODEL OWNERSHIP", fontsize=18, fontweight="bold", color=GREEN)

    nodes = [
        (0.84, "Immutable Raw Source", BLUE_PALE, BLUE_DARK),
        (0.735, "Parser", "white", BLUE),
        (0.63, "Canonical Schema", "white", BLUE),
        (0.535, "Artifact Contract  /  Validation", "white", BLUE_DARK),
        (0.35, "Model Consumer\nsampling  /  transform  /  window  /  tensorization", "white", GREEN),
        (0.225, "PLUTO Policy", "white", GREEN),
        (0.105, "Closed-loop Evaluation", GREEN_PALE, GREEN),
    ]
    heights = [0.075, 0.075, 0.075, 0.075, 0.095, 0.075, 0.075]
    for index, ((y, label, face, edge), height) in enumerate(zip(nodes, heights)):
        rounded_box(ax, (0.30, y - height / 2), 0.48, height, label, face, edge, 15)
        if index < len(nodes) - 1:
            next_y = nodes[index + 1][0]
            next_h = heights[index + 1]
            ax.annotate(
                "",
                xy=(0.54, next_y + next_h / 2 + 0.006),
                xytext=(0.54, y - height / 2 - 0.006),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8),
            )

    ax.plot([0.05, 0.95], [0.45, 0.45], color=RED, lw=2.5, dashes=(6, 4))
    ax.text(
        0.945,
        0.455,
        "RESPONSIBILITY BOUNDARY",
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=RED,
    )
    ax.text(
        0.805,
        0.68,
        "Owns factual meaning\n+ schema + validation",
        ha="left",
        va="center",
        fontsize=14,
        color=BLUE_DARK,
        linespacing=1.35,
    )
    ax.text(
        0.805,
        0.25,
        "Owns model-specific\ninterpretation",
        ha="left",
        va="center",
        fontsize=14,
        color=GREEN,
        linespacing=1.35,
    )
    ax.set_title(
        "Model-independent facts, consumer-owned interpretation",
        loc="left",
        fontsize=24,
        fontweight="bold",
        pad=18,
    )
    save(fig, "01_data_model_boundary")


def contract_evidence() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Contract registry: implemented and reserved are counted separately",
        loc="left",
        fontsize=23,
        fontweight="bold",
        pad=18,
    )
    ax.text(0.02, 0.91, "Code-inspected registry and data-free diagnostic tests", color=MUTED, fontsize=15)

    cards = [
        (0.02, "9", "registered\ncontracts", BLUE_PALE, BLUE_DARK),
        (0.27, "8", "implemented\nnon-empty", GREEN_PALE, GREEN),
        (0.52, "1", "reserved\nprediction", AMBER_PALE, AMBER),
        (0.77, "3", "fail-fast\ntests", "#F2EEFA", "#6B4BA1"),
    ]
    for x, number, label, face, edge in cards:
        box = FancyBboxPatch(
            (x, 0.66), 0.21, 0.19,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=face, edgecolor=edge, linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x + 0.045, 0.755, number, fontsize=34, fontweight="bold", color=edge, va="center")
        ax.text(x + 0.105, 0.755, label, fontsize=14, fontweight="bold", va="center", linespacing=1.3)

    ax.text(0.02, 0.595, "REGISTRY ENTRIES", fontsize=16, fontweight="bold")
    names = [
        "sample", "ego_pose", "ego_dynamics", "agent_tracks", "route",
        "scene_log", "map_graph", "counterfactual", "prediction",
    ]
    for i, name in enumerate(names):
        col, row = i % 3, i // 3
        x = 0.02 + col * 0.19
        y = 0.47 - row * 0.115
        reserved = name == "prediction"
        face = AMBER_PALE if reserved else GREEN_PALE
        edge = AMBER if reserved else GREEN
        ax.add_patch(Rectangle((x, y), 0.17, 0.085, facecolor=face, edgecolor=edge, linewidth=1.5))
        ax.text(x + 0.085, y + 0.0425, name, ha="center", va="center", fontsize=14, fontweight="bold")

    ax.text(0.64, 0.595, "3 FAIL-FAST TESTS", fontsize=16, fontweight="bold")
    tests = [
        ("Missing file", "includes producer hint"),
        ("Missing required key", "reports schema mismatch"),
        ("Unsupported provenance", "lists supported values"),
    ]
    for i, (title, detail) in enumerate(tests):
        y = 0.455 - i * 0.13
        rounded_box(ax, (0.64, y), 0.33, 0.10, f"{title}\n{detail}", "white", "#6B4BA1", 14)

    ax.text(
        0.02,
        0.035,
        "Boundary: 'registered' does not mean all contracts have producers or required files.",
        fontsize=14,
        color=RED,
        fontweight="bold",
    )
    save(fig, "02_contract_evidence")


def counterfactual_quality() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Counterfactual dataset — documented OpenScene mini validation",
        loc="left",
        fontsize=23,
        fontweight="bold",
        pad=18,
    )
    ax.text(0.02, 0.91, "Raw report.json is not tracked; values below are the repository record.", color=MUTED, fontsize=15)

    metrics = [("64", "logs"), ("2,848", "anchors"), ("7,608", "counterfactual samples")]
    for i, (value, label) in enumerate(metrics):
        x = 0.02 + i * 0.325
        box = FancyBboxPatch(
            (x, 0.69), 0.30, 0.15,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=BLUE_PALE, edgecolor=BLUE, linewidth=1.8,
        )
        ax.add_patch(box)
        ax.text(x + 0.025, 0.77, value, fontsize=29, fontweight="bold", color=BLUE_DARK, va="center")
        ax.text(x + 0.025, 0.72, label, fontsize=14, color=INK, va="center")

    ax.text(0.02, 0.615, "PHYSICAL QUALITY GATE PASS", fontsize=16, fontweight="bold")
    ax.add_patch(Rectangle((0.02, 0.52), 0.92, 0.065, facecolor=GRAY, edgecolor="none"))
    ax.add_patch(Rectangle((0.02, 0.52), 0.92 * 0.885, 0.065, facecolor=GREEN, edgecolor="none"))
    ax.text(0.02 + 0.92 * 0.885 - 0.01, 0.5525, "88.5%", ha="right", va="center", color="white", fontsize=18, fontweight="bold")
    ax.text(
        0.02,
        0.475,
        "Pass = curvature gate AND lateral-acceleration gate (not model accuracy or a dataset quality score)",
        fontsize=14,
        color=RED,
        fontweight="bold",
    )

    ax.text(0.02, 0.385, "VOCABULARY NEAREST DISTANCE", fontsize=16, fontweight="bold")
    x0, x1, y = 0.12, 0.92, 0.27
    ax.plot([x0, x1], [y, y], color=INK, lw=1.7)
    for tick in [0.0, 0.2, 0.4, 0.6]:
        x = x0 + (tick / 0.6) * (x1 - x0)
        ax.plot([x, x], [y - 0.012, y + 0.012], color=INK, lw=1.2)
        ax.text(x, y - 0.045, f"{tick:.1f} m", ha="center", va="top", fontsize=14, color=MUTED)
    for value, label, color, dy in [(0.30, "p50  0.30 m", BLUE, 0.055), (0.47, "p90  0.47 m", AMBER, 0.055)]:
        x = x0 + (value / 0.6) * (x1 - x0)
        ax.scatter([x], [y], s=180, color=color, zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(x, y + dy, label, ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    ax.text(0.02, 0.095, "Distance is a distribution measurement; the implementation defines no pass/fail threshold.", fontsize=14, color=MUTED)
    save(fig, "03_counterfactual_quality")


def main() -> None:
    setup()
    ownership_architecture()
    contract_evidence()
    counterfactual_quality()


if __name__ == "__main__":
    main()
