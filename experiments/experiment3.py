import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import experiment1 as base
from experiments.valid_experiment_utils import run_experiment


TEST_LIST = [
    (2, 10, 40),
    (2, 10, 70),
    (2, 10, 100),
    (2, 10, 200),
    (2, 10, 300),
    (2, 10, 400),
    (2, 10, 500),
]
POINT_KEYS = ["N", "V", "tot"]
RESULT_ROOT = Path("results") / "experiments3" / "try3"


def build_task(point, attempt_idx, seed):
    random.seed(seed)
    np.random.seed(seed)
    task = base.TaskConfig()
    task.RP_number = 4
    task.processor_number = task.CPU_number + task.NPU_number + task.RP_number
    random_dict = base.build_random_dict(point["N"], point["V"], point["tot"])
    random_dict["NWR"] = 4.0
    task.init_class_vars(random_dict, mode="random")
    return task, seed


def draw_grouped_bar_plot(summary_df, result_root, metric_key, ylabel, output_name):
    if summary_df.empty:
        return

    plot_df = summary_df.sort_values(by=["tot"], kind="stable").copy()
    x_values = [str(int(tot)) for tot in plot_df["tot"].tolist()]

    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(16, 10))

    metric_series = []
    for algorithm_name, _label in base.PLOT_ALGORITHMS:
        metric_series.append(plot_df[f"{algorithm_name} {metric_key}"].tolist())

    bar_width = 0.12
    intra_group_gap = 0.04
    group_gap = 0.35
    total_group_width = len(base.PLOT_ALGORITHMS) * bar_width + (len(base.PLOT_ALGORITHMS) - 1) * intra_group_gap
    index = np.arange(len(x_values)) * (total_group_width + group_gap)

    for i, (_algorithm_name, label) in enumerate(base.PLOT_ALGORITHMS):
        bar_positions = index + i * (bar_width + intra_group_gap)
        plt.bar(
            bar_positions,
            metric_series[i],
            width=bar_width,
            label=label,
            color=base.PLOT_COLORS[i],
            edgecolor="black",
            alpha=0.85,
            hatch=base.PLOT_HATCHES[i],
        )

    plt.xlabel("Task Instances in Hyperperiod", fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    group_center_offset = (len(base.PLOT_ALGORITHMS) - 1) * (bar_width + intra_group_gap) / 2
    plt.xticks(index + group_center_offset, x_values, fontsize=24)
    plt.yticks(fontsize=24)
    max_value = max(max(series) for series in metric_series if series)
    min_value = min(min(series) for series in metric_series if series)
    if min_value >= 0:
        plt.ylim(0, max_value * 1.15 if max_value > 0 else 1)
    else:
        lower = min_value * 1.15
        upper = max_value * 1.15 if max_value > 0 else 1
        if lower == upper:
            upper = lower + 1
        plt.ylim(lower, upper)
    plt.legend(bbox_to_anchor=(1, 0), loc="lower right", fontsize=20)
    plt.tight_layout()
    plt.savefig(result_root / output_name, dpi=300)
    plt.close()


def draw_result_figure(summary_df, result_root):
    draw_grouped_bar_plot(
        summary_df,
        result_root,
        metric_key="avg_energy",
        ylabel="Normalized Energy-consumption (%)",
        output_name="experiment3_energy_plot.png",
    )
    draw_grouped_bar_plot(
        summary_df,
        result_root,
        metric_key="avg_delay",
        ylabel="Average Deadline Violation (ms)",
        output_name="experiment3_delay_plot.png",
    )
    draw_grouped_bar_plot(
        summary_df,
        result_root,
        metric_key="avg_runtime_seconds",
        ylabel="Average Runtime (s)",
        output_name="experiment3_runtime_plot.png",
    )


def main():
    points = [
        {"N": N, "V": V, "tot": tot}
        for N, V, tot in TEST_LIST
    ]
    run_experiment(
        points=points,
        point_keys=POINT_KEYS,
        result_root=RESULT_ROOT,
        build_task_fn=build_task,
        file_prefix="experiment3",
        required_valid_count=5,
        plot_callback=draw_result_figure,
        require_all_algorithms_feasible=True,
    )
    print(f"[FINISH] Results saved under {RESULT_ROOT}")


if __name__ == "__main__":
    main()
