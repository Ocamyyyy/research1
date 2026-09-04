import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from experiments import experiment1 as base
from experiments.valid_experiment_utils import run_experiment


TEST_LIST_TOT = [
    # (2, 5, 30),
    # (2, 5, 40),
    # (2, 5, 50),
    # (2, 5, 60),
    # (2, 5, 70),
    # (2, 5, 80),
    # (2, 5, 90),
    # (2, 5, 100),
    # (2, 10, 40),
    # (2, 10, 70),
    # (2, 10, 100),
    # (2, 10, 200),
    # (2, 10, 300),
    # (2, 10, 400),
    # (2, 10, 500),
    # (2, 20, 100),
    # (2, 20, 200),
    # (2, 20, 300),
    # (2, 20, 400),
    (2, 20, 500),
]
REPORT_TEST_LIST_TOT = [
    (2, 5, 30),
    (2, 5, 40),
    (2, 5, 50),
    (2, 5, 60),
    (2, 5, 70),
    (2, 5, 80),
    (2, 5, 90),
    (2, 5, 100),
    (2, 10, 40),
    (2, 10, 70),
    (2, 10, 100),
    (2, 10, 200),
    (2, 10, 300),
    (2, 10, 400),
    (2, 10, 500),
    (2, 20, 100),
    (2, 20, 200),
    (2, 20, 300),
    (2, 20, 400),
    (2, 20, 500),
]
REPORT_NWR_LIST = [0.5, 1.0, 2.0, 3.0, 4.0]
NWR_LIST = [
    # 0.5, 
    # 1.0, 
    # 2.0, 
    3.0, 
    # 4.0
]
ALPHA_LIST = [
    0.5, 
    1.0, 
    2.0
]
MAX_OUT_LIST = [1, 2, 4]
# MAX_OUT_LIST = [2]
RP_NUMBER_LIST = [2, 4]
# RP_NUMBER_LIST = [4]

POINT_KEYS = ["N", "V", "tot", "alpha", "max_out", "NWR", "RP_number"]
RESULT_ROOT = Path("results") / "experiments4" / "try1"


def build_task(point, attempt_idx, seed):
    random.seed(seed)
    np.random.seed(seed)

    task = base.TaskConfig()
    task.RP_number = point["RP_number"]
    task.processor_number = task.CPU_number + task.NPU_number + task.RP_number

    random_dict = base.build_random_dict(point["N"], point["V"], point["tot"])
    random_dict["alpha"] = point["alpha"]
    random_dict["max_out"] = point["max_out"]
    random_dict["NWR"] = point["NWR"]
    task.init_class_vars(random_dict, mode="random")
    return task, seed


def _best_algorithms_for_row(row):
    candidates = []
    for algorithm_name in base.ALGORITHM_ORDER:
        all_feasible = row.get(f"{algorithm_name} all_feasible")
        delay = row.get(f"{algorithm_name} avg_delay")
        energy = row.get(f"{algorithm_name} avg_energy")
        if all_feasible is not True or pd.isna(delay) or pd.isna(energy):
            continue
        candidates.append((algorithm_name, float(delay), float(energy)))

    if not candidates:
        return []

    min_delay = min(item[1] for item in candidates)
    delay_best = [item for item in candidates if item[1] == min_delay]
    min_energy = min(item[2] for item in delay_best)
    return [item[0] for item in delay_best if item[2] == min_energy]


def _best_algorithms_for_sample(row):
    feasible_candidates = []
    for algorithm_name in base.ALGORITHM_ORDER:
        if algorithm_name == "HEFT":
            continue
        feasible = row.get(f"{algorithm_name} feasible")
        delay = row.get(f"{algorithm_name} delay")
        energy = row.get(f"{algorithm_name} energy")
        if feasible is not True or pd.isna(delay) or pd.isna(energy):
            continue
        feasible_candidates.append((algorithm_name, float(delay), float(energy)))

    candidates = feasible_candidates
    if not candidates:
        for algorithm_name in base.ALGORITHM_ORDER:
            if algorithm_name in {"HEFT", "eFLS_adapted"}:
                continue
            delay = row.get(f"{algorithm_name} delay")
            energy = row.get(f"{algorithm_name} energy")
            if pd.isna(delay) or pd.isna(energy):
                continue
            candidates.append((algorithm_name, float(delay), float(energy)))
        if not candidates:
            return []

    min_delay = min(item[1] for item in candidates)
    delay_best = [item for item in candidates if item[1] == min_delay]
    min_energy = min(item[2] for item in delay_best)
    return [item[0] for item in delay_best if item[2] == min_energy]


def write_best_share_report(summary_df, result_root):
    report_path = result_root / "experiment4_best_share_by_test_list_tot.txt"
    raw_csv_path = result_root / "experiment4_raw_results.csv"
    total_point_count = len(REPORT_NWR_LIST) * len(ALPHA_LIST) * len(MAX_OUT_LIST) * len(RP_NUMBER_LIST)
    total_experiment_point_count = total_point_count * len(REPORT_TEST_LIST_TOT)
    lines = [
        "Experiment 4 Best-Solution Share by TEST_LIST_TOT",
        "Rule: compare completed samples one by one.",
        "HEFT is excluded from all best-solution comparisons.",
        "For each sample, feasible=True non-HEFT algorithms are preferred and compared by delay first, then energy.",
        "If all non-HEFT algorithms are infeasible on a sample, compare non-HEFT and non-eFLS algorithms by delay first, then energy.",
        "Within each experiment point, each sample contributes 1 / sample_count.",
        "If multiple algorithms tie on one sample, that sample weight is split equally among them.",
        f"Final counts are summed over experiment points and reported against {total_point_count} total points.",
        "",
    ]

    if not raw_csv_path.exists():
        lines.append("No raw experiment results found yet.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    raw_df = pd.read_csv(raw_csv_path, encoding="utf-8-sig")
    completed_df = raw_df[raw_df["status"] == "completed"].copy()

    if completed_df.empty:
        lines.append("No completed experiment points yet.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    overall_share = {algorithm_name: 0.0 for algorithm_name in base.ALGORITHM_ORDER}
    overall_completed_point_count = 0

    for N, V, tot in REPORT_TEST_LIST_TOT:
        tot_group = completed_df[
            (completed_df["N"] == N)
            & (completed_df["V"] == V)
            & (completed_df["tot"] == tot)
        ]
        share = {algorithm_name: 0.0 for algorithm_name in base.ALGORITHM_ORDER}
        completed_point_count = 0

        for _, point_group in tot_group.groupby(POINT_KEYS, sort=True):
            sample_count = len(point_group)
            if sample_count == 0:
                continue
            completed_point_count += 1
            overall_completed_point_count += 1
            for _, row in point_group.iterrows():
                winners = _best_algorithms_for_sample(row)
                if not winners:
                    continue
                contribution = 1.0 / sample_count / len(winners)
                for algorithm_name in winners:
                    share[algorithm_name] += contribution
                    overall_share[algorithm_name] += contribution

        lines.append(f"TEST_LIST_TOT case: N={N}, V={V}, tot={tot}")
        lines.append(f"Completed experiment points: {completed_point_count}")
        if completed_point_count == 0:
            lines.append("No completed point yet.")
            lines.append("")
            continue

        for algorithm_name in base.ALGORITHM_ORDER:
            percentage = share[algorithm_name] / total_point_count * 100.0
            lines.append(
                f"{algorithm_name}: {percentage:.2f}% "
                f"({share[algorithm_name]:.2f}/{total_point_count})"
            )
        lines.append("")

    lines.append("Overall summary across all TEST_LIST_TOT cases")
    lines.append(f"Completed experiment points: {overall_completed_point_count}")
    lines.append(
        f"Reported against {total_experiment_point_count} total points "
        f"({total_point_count} points x {len(REPORT_TEST_LIST_TOT)} TEST_LIST_TOT cases)."
    )
    if overall_completed_point_count == 0:
        lines.append("No completed point yet.")
    else:
        for algorithm_name in base.ALGORITHM_ORDER:
            percentage = overall_share[algorithm_name] / total_experiment_point_count * 100.0
            lines.append(
                f"{algorithm_name}: {percentage:.2f}% "
                f"({overall_share[algorithm_name]:.2f}/{total_experiment_point_count})"
            )
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_experiment4_outputs(summary_df, result_root):
    write_best_share_report(summary_df, result_root)


def rebuild_summary_from_cases():
    case_root = RESULT_ROOT / "cases"
    from experiments.valid_experiment_utils import write_summary_files

    write_summary_files(
        result_root=RESULT_ROOT,
        case_root=case_root,
        point_keys=POINT_KEYS,
        file_prefix="experiment4",
        plot_callback=update_experiment4_outputs,
    )
    print(f"[FINISH] Rebuilt summary from {case_root}")


def main():
    points = []
    for N, V, tot in TEST_LIST_TOT:
        for nwr in NWR_LIST:
            for alpha in ALPHA_LIST:
                for max_out in MAX_OUT_LIST:
                    for rp_number in RP_NUMBER_LIST:
                        points.append(
                            {
                                "N": N,
                                "V": V,
                                "tot": tot,
                                "alpha": alpha,
                                "max_out": max_out,
                                "NWR": nwr,
                                "RP_number": rp_number,
                            }
                        )

    run_experiment(
        points=points,
        point_keys=POINT_KEYS,
        result_root=RESULT_ROOT,
        build_task_fn=build_task,
        file_prefix="experiment4",
        required_valid_count=5,
        plot_callback=update_experiment4_outputs,
        require_heft_feasible=False,
        require_all_algorithms_feasible=False,
        refresh_summary_during_run=False,
    )
    print(f"[FINISH] Case results saved under {RESULT_ROOT / 'cases'}")


if __name__ == "__main__":
    main()
