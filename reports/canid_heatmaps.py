"""Generate standard subliminal-learning heatmaps for the canid experiments.

Outputs:
  reports/charts/canid_heatmap_final_with_baseline_control.png
  reports/charts/canid_heatmap_delta_with_baseline_control.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "outputs" / "canid-chart-data"
BASELINE_PATH = REPO / "private" / "final-results-qwen2.5" / "baseline.json"
CONTROL_PATH = REPO / "private" / "final-results-qwen2.5" / "control.json"
OUT_DIR = REPO / "reports" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = {
    "canids": "canid",
    "canid-leaveout-fox": "leaveout fox",
    "foxes": "fox",
    "canid-leaveout-wolf": "leaveout wolf",
    "canid-leaveout-dog": "leaveout dog",
}

TRACKED = [
    "dog",
    "wolf",
    "fox",
    "coyote",
    "cat",
    "lion",
    "tiger",
    "leopard",
    "cheetah",
    "jaguar",
    "panther",
    "lynx",
    "eagle",
    "peacock",
    "phoenix",
    "owl",
    "dolphin",
    "otter",
    "whale",
    "octopus",
    "elephant",
    "panda",
    "dragon",
]


def load_array(path: Path):
    return json.loads(path.read_text())


def final_pct(path: Path):
    data = load_array(path)
    row = data[-1]
    return row["filtered_pct"]


baseline_final = load_array(BASELINE_PATH)[-1]["filtered_pct"]
control_final = load_array(CONTROL_PATH)[-1]["filtered_pct"]


def build_matrix(source):
    n_rows = len(RUNS) + 2
    mat = np.zeros((n_rows, len(TRACKED)))
    row_labels = list(RUNS.values()) + ["baseline", "control"]

    for i, run in enumerate(RUNS):
        pct = source(run)
        for j, animal in enumerate(TRACKED):
            mat[i, j] = pct.get(animal, 0.0)

    for j, animal in enumerate(TRACKED):
        mat[len(RUNS), j] = baseline_final.get(animal, 0.0)
        mat[len(RUNS) + 1, j] = control_final.get(animal, 0.0)

    return mat, row_labels


def save_heatmap_final():
    def src(run):
        return final_pct(DATA_DIR / f"{run}.json")

    mat, row_labels = build_matrix(src)

    fig, ax = plt.subplots(figsize=(18, 7))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(100, mat.max()))

    ax.set_xticks(range(len(TRACKED)))
    ax.set_xticklabels(TRACKED, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Measured animal")
    ax.set_ylabel("Run")
    ax.set_title("Canid experiments: final response % with baseline and control")

    for i in range(len(row_labels)):
        for j in range(len(TRACKED)):
            val = mat[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=color)

    ax.axhline(len(RUNS) - 0.5, color="white", linewidth=2)
    ax.axhline(len(RUNS) + 0.5, color="white", linewidth=1, linestyle="--")

    plt.colorbar(im, ax=ax, label="Response %", shrink=0.85)
    plt.tight_layout()
    out = OUT_DIR / "canid_heatmap_final_with_baseline_control.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def save_heatmap_delta():
    def src(run):
        pct = final_pct(DATA_DIR / f"{run}.json")
        return {
            animal: pct.get(animal, 0.0) - baseline_final.get(animal, 0.0)
            for animal in TRACKED
        }

    n_rows = len(RUNS) + 2
    row_labels = list(RUNS.values()) + ["baseline", "control"]
    mat = np.zeros((n_rows, len(TRACKED)))

    for i, run in enumerate(RUNS):
        delta = src(run)
        for j, animal in enumerate(TRACKED):
            mat[i, j] = delta[animal]

    for j, animal in enumerate(TRACKED):
        mat[len(RUNS), j] = 0.0
        mat[len(RUNS) + 1, j] = control_final.get(animal, 0.0) - baseline_final.get(animal, 0.0)

    vmax = np.ceil(np.max(np.abs(mat)) / 5.0) * 5.0
    vmax = max(vmax, 5.0)

    fig, ax = plt.subplots(figsize=(18, 7))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(TRACKED)))
    ax.set_xticklabels(TRACKED, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Measured animal")
    ax.set_ylabel("Run")
    ax.set_title("Canid experiments: final minus baseline (pp), with control row")

    for i in range(len(row_labels)):
        for j in range(len(TRACKED)):
            val = mat[i, j]
            text_color = "white" if abs(val / vmax) > 0.55 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7, color=text_color)

    ax.axhline(len(RUNS) - 0.5, color="white", linewidth=2)
    ax.axhline(len(RUNS) + 0.5, color="white", linewidth=1, linestyle="--")

    cbar = plt.colorbar(im, ax=ax, label="Percentage point change vs baseline", shrink=0.85)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f pp"))

    plt.tight_layout()
    out = OUT_DIR / "canid_heatmap_delta_with_baseline_control.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


if __name__ == "__main__":
    out1 = save_heatmap_final()
    out2 = save_heatmap_delta()
    print(out1)
    print(out2)
