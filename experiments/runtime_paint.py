from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import experiment1 as base


RESULT_SUMMARY_PATH = Path("results") / "experiments" / "try3" / "experiment1_summary.csv"
OUTPUT_PATH = Path("results") / "experiment1_runtime_plot.png"

PLOT_MARKERS = ["o", "s", "^", "D", "v", "P"]
RUNTIME_PLOT_CONFIG = [
    ("SAFLA_adapted", "SAFLA", 0),
    ("eFLS_adapted", "eFLS", 1),
    ("GA", "GAA", 3),
    ("HHO", "QC-HHO", 4),
    ("QHA", "AEHS", 5),
]


def main():
    if not RESULT_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {RESULT_SUMMARY_PATH}")

    summary_df = pd.read_csv(RESULT_SUMMARY_PATH, encoding="utf-8-sig")
    if summary_df.empty:
        raise ValueError(f"Summary file is empty: {RESULT_SUMMARY_PATH}")

    plot_df = summary_df.sort_values(by=["tot"], kind="stable").copy()
    original_x_values = plot_df["tot"].astype(int).tolist()
    x_positions = list(range(len(original_x_values)))
    x_labels = [str(value) for value in original_x_values]

    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(12, 10))

    for idx, (algorithm_name, label, color_idx) in enumerate(RUNTIME_PLOT_CONFIG):
        runtime_col = f"{algorithm_name} avg_runtime_seconds"
        if runtime_col not in plot_df.columns:
            continue

        y_values = plot_df[runtime_col].tolist()
        plt.plot(
            x_positions,
            y_values,
            label=label,
            color=base.PLOT_COLORS[color_idx],
            marker=PLOT_MARKERS[idx],
            linewidth=2,
            markersize=8,
        )

    plt.xlabel("Task Instances in Hyperperiod", fontsize=28)
    plt.ylabel("Average Complete Computing Time (s)", fontsize=28)
    plt.xticks(x_positions, x_labels, fontsize=28)
    plt.yticks(fontsize=28)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=20)

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()

    print(f"[FINISH] Saved runtime plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
