import copy
import csv
import math
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from DAGs import DAG
from GA import fitness_function
from QHA import QuantumHeuristicSpace, apply_heuristic, quantum_chaotic_initialization
from solution_utils import calc_subtask_instance_from_position
from task_definition import E_max, min_energy_processor_type, min_execution_processor_type


INF_TIME_MS = 1_000_000_000
OUTPUT_DIR = Path(os.environ.get("CASE_STUDY_OUTPUT_DIR", "case_study_qha_results"))
EO_PERIOD_MS = int(os.environ.get("EO_PERIOD_MS", "2500"))
SAR_PERIOD_MS = int(os.environ.get("SAR_PERIOD_MS", "5000"))

CPU_NUMBER = 1
NPU_NUMBER = 1
RP_NUMBER = 2
RP_P_IDLE = 0.5

POP_SIZE = int(os.environ.get("POP_SIZE", "20"))
PHASE_ITER = 5
MAX_ITER = int(os.environ.get("MAX_ITER", "50"))
NUM_HEURISTICS = 5
ELITISM_RATIO = 0.1

SEED = int(os.environ.get("SEED", "24"))
NUM_PHASES = MAX_ITER // PHASE_ITER

PROCESSOR_LABELS = {
    0: "CPU0",
    1: "TPE0",
    2: "Box0",
    3: "Box1",
}

LANE_LABELS = {
    0: "CPU",
    1: "TPE",
    2: "FPGA Box 1",
    3: "FPGA Box 2",
}

APP_STYLE = {
    "EO": {"color": "#9EC5FE", "text_color": "black"},
    "SAR": {"color": "#F4B183", "text_color": "black"},
}

OPERATOR_METADATA = {
    0: {
        "code_function": "llh1",
        "paper_RO": "RO1",
        "description": "Swap the processor mappings of two tasks currently mapped to different processors.",
    },
    1: {
        "code_function": "llh2",
        "paper_RO": "RO2",
        "description": "Move one task to a different eligible processor and rebuild the schedule.",
    },
    2: {
        "code_function": "llh3",
        "paper_RO": "RO3",
        "description": "Swap the execution order of two tasks on the same processor without changing mapping.",
    },
    3: {
        "code_function": "llh4",
        "paper_RO": "RO4",
        "description": "Remap sampled tasks to the fastest eligible processor type indicated by execution-time minima.",
    },
    4: {
        "code_function": "llh5",
        "paper_RO": "RO5",
        "description": "Remap sampled tasks to the lowest-energy eligible processor type indicated by energy minima.",
    },
}


class SatelliteCaseTaskConfig:
    def __init__(self):
        self.CPU_number = CPU_NUMBER
        self.NPU_number = NPU_NUMBER
        self.RP_number = RP_NUMBER
        self.processor_number = self.CPU_number + self.NPU_number + self.RP_number

        self.RP_P_idle = RP_P_IDLE

        self.app_list = []
        self.instance_number_list = []
        self.instance_dict_list = []
        self.mapping_time_min = []
        self.mapping_energy_min = []
        self.task_number = 0
        self.H = 0
        self.E_max = 0

        self._build()

    def _build(self):
        eo_dict = {
            "node num": 3,
            "edges": [(0, 1), (1, 2)],
            "period": EO_PERIOD_MS,
            "sw exe": [22, 9364, 282],
            "npu exe": [0, 220, 0],
            "hw exe": [9, 13, 65],
            "tpr": [50, 50, 50],
            "P static": [3.79, 3.79, 3.79],
            "P dyn": [2.21, 2.21, 2.21],
            "P sw": [0.5, 0.5, 0.5],
            "P npu": [0.0, 5.0, 0.0],
            "cpu eligible": [True, True, True],
            "npu eligible": [False, True, False],
            "hw eligible": [True, True, True],
        }
        sar_dict = {
            "node num": 3,
            "edges": [(0, 1), (1, 2)],
            "period": SAR_PERIOD_MS,
            "sw exe": [10, 267560, 10],
            "npu exe": [0, 14970, 0],
            "hw exe": [0, 1965, 0],
            "tpr": [0, 50, 0],
            "P static": [0.0, 0.83, 0.0],
            "P dyn": [0.0, 0.48, 0.0],
            "P sw": [0.5, 0.5, 0.5],
            "P npu": [0.0, 5.0, 0.0],
            "cpu eligible": [True, True, True],
            "npu eligible": [False, True, False],
            "hw eligible": [False, True, False],
        }

        self.app_list = [DAG(eo_dict, mode="certain"), DAG(sar_dict, mode="certain")]
        self.H = math.lcm(*(app.period for app in self.app_list))
        self.instance_number_list = [self.H // app.period for app in self.app_list]
        self.instance_dict_list = [{} for _ in self.app_list]
        for app_index, app in enumerate(self.app_list):
            for instance_idx in range(self.instance_number_list[app_index]):
                q = instance_idx + 1
                self.instance_dict_list[app_index][q] = {
                    "arr": instance_idx * app.period,
                    "ddl": (instance_idx + 1) * app.period,
                }
        self.task_number = sum(
            instances * app.node_num for instances, app in zip(self.instance_number_list, self.app_list)
        )
        random_dict = {
            "P_sw": 0.5,
            "P_npu": 5.0,
            "P_static": 3.0,
            "P_dyn": 6.0,
        }
        self.E_max = E_max(
            self.app_list,
            self.CPU_number,
            self.NPU_number,
            self.RP_number,
            self.H,
            random_dict=random_dict,
            rp_p_idle=self.RP_P_idle,
        )
        for i, app in enumerate(self.app_list):
            for _ in range(self.instance_number_list[i]):
                for j in range(app.node_num):
                    self.mapping_time_min.append(min_execution_processor_type(self.app_list, i, j))
                    self.mapping_energy_min.append(min_energy_processor_type(self.app_list, i, j))


def reset_output_dir(output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def select_global_best_index(energy, delay, fitness):
    feasible_indices = [i for i, d in enumerate(delay) if d == 0]
    if feasible_indices:
        feasible_fitness = [fitness[i] for i in feasible_indices]
        best_feasible_idx = int(np.argmin(feasible_fitness))
        return feasible_indices[best_feasible_idx]
    return int(np.argmin(fitness))


def processor_name(processor_idx: int) -> str:
    return PROCESSOR_LABELS[processor_idx]


def lane_label(processor_idx: int) -> str:
    return LANE_LABELS[processor_idx]


def app_short_name(app_index: int) -> str:
    return "EO" if app_index == 0 else "SAR"


def plain_node_label(task, pos: int) -> str:
    app_index, instance_index, node_index = calc_subtask_instance_from_position(task, pos)
    return f"{app_short_name(app_index)}{instance_index}-V{node_index + 1}"


def math_node_label(task, pos: int) -> str:
    app_index, instance_index, node_index = calc_subtask_instance_from_position(task, pos)
    prefix = app_short_name(app_index)
    return rf"{prefix}$_{instance_index}$-$V_{node_index + 1}$"


def configuration_label(task, pos: int) -> str:
    app_index, _instance_index, node_index = calc_subtask_instance_from_position(task, pos)
    return f"{app_short_name(app_index)}-V{node_index + 1}"


def mapping_to_named_list(solution) -> list[str]:
    return [processor_name(p) for p in solution.task_mapping]


def format_mapping_inline(task, solution):
    return ", ".join(
        f"{plain_node_label(task, pos)}->{processor_name(proc)}"
        for pos, proc in enumerate(solution.task_mapping)
    )


def diff_mapping(task, before_solution, after_solution):
    changes = []
    for pos in range(task.task_number):
        if before_solution.task_mapping[pos] != after_solution.task_mapping[pos]:
            changes.append(
                (
                    plain_node_label(task, pos),
                    processor_name(before_solution.task_mapping[pos]),
                    processor_name(after_solution.task_mapping[pos]),
                )
            )
    return changes


def snapshot_task_rows(task, solution, snapshot_name, iteration):
    rows = []
    for pos in range(task.task_number):
        rows.append(
            {
                "snapshot": snapshot_name,
                "iteration": iteration,
                "task": plain_node_label(task, pos),
                "processor": processor_name(solution.task_mapping[pos]),
                "start_time_ms": solution.task_start_time[pos],
                "finish_time_ms": solution.task_end_time[pos],
                "reconfiguration_required": int(solution.task_DPR[pos]),
                "reconfiguration_start_ms": solution.task_DPR_start_time[pos] if solution.task_DPR[pos] else "",
                "reconfiguration_finish_ms": solution.task_DPR_end_time[pos] if solution.task_DPR[pos] else "",
            }
        )
    return rows


def count_configuration_reuse_events(task, solution):
    events = []
    for processor in range(task.CPU_number + task.NPU_number, task.processor_number):
        task_list = solution.processor_completed_task[processor]
        for idx in range(len(task_list) - 1):
            previous_task = task_list[idx]
            next_task = task_list[idx + 1]
            previous_config = configuration_label(task, previous_task)
            next_config = configuration_label(task, next_task)
            reuse = previous_config == next_config and int(solution.task_DPR[next_task]) == 0
            if reuse:
                app_index, _instance_index, node_index = calc_subtask_instance_from_position(task, next_task)
                saved_reconfiguration_ms = task.app_list[app_index].tpr[node_index]
                events.append(
                    {
                        "fpga_box": lane_label(processor),
                        "previous_task": plain_node_label(task, previous_task),
                        "next_task": plain_node_label(task, next_task),
                        "configuration": next_config,
                        "reuse": True,
                        "saved_reconfiguration_ms": saved_reconfiguration_ms,
                    }
                )
    return events


def write_csv_rows(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_iteration_summary(output_dir: Path, rows):
    fieldnames = [
        "iteration",
        "phase",
        "best_energy_mJ",
        "best_delay_ms",
        "best_is_feasible",
        "population_min_delay_ms",
        "population_mean_delay_ms",
        "population_feasible_count",
        "population_feasible_ratio",
        "best_feasible_energy_mJ",
        "population_mean_feasible_energy_mJ",
    ]
    write_csv_rows(output_dir / "iteration_summary.csv", fieldnames, rows)


def write_phase_operator_summary(output_dir: Path, rows):
    fieldnames = [
        "phase",
        "operator",
        "code_function",
        "selected",
        "usage_count",
        "improvement_count",
        "improvement_rate",
        "probability_before",
        "probability_after",
    ]
    write_csv_rows(output_dir / "phase_operator_summary.csv", fieldnames, rows)


def write_solution_snapshots(output_dir: Path, snapshot_rows, summary_rows):
    write_csv_rows(
        output_dir / "solution_snapshots.csv",
        [
            "snapshot",
            "iteration",
            "task",
            "processor",
            "start_time_ms",
            "finish_time_ms",
            "reconfiguration_required",
            "reconfiguration_start_ms",
            "reconfiguration_finish_ms",
        ],
        snapshot_rows,
    )
    write_csv_rows(
        output_dir / "solution_snapshot_summary.csv",
        ["snapshot", "iteration", "energy_mJ", "delay_ms", "reconfiguration_count"],
        summary_rows,
    )


def write_mapping_evolution(output_dir: Path, task, snapshot_solutions):
    snapshot_names = list(snapshot_solutions.keys())
    rows = []
    for pos in range(task.task_number):
        row = {"task": plain_node_label(task, pos)}
        for snapshot_name, solution in snapshot_solutions.items():
            row[snapshot_name] = processor_name(solution.task_mapping[pos])
        rows.append(row)
    fieldnames = ["task"] + snapshot_names
    write_csv_rows(output_dir / "mapping_evolution.csv", fieldnames, rows)


def write_final_schedule(output_dir: Path, task, solution):
    rows = []
    for pos in range(task.task_number):
        app_index, instance_index, node_index = calc_subtask_instance_from_position(task, pos)
        processor = solution.task_mapping[pos]
        rows.append(
            {
                "task": math_node_label(task, pos),
                "application": app_short_name(app_index),
                "instance": instance_index,
                "node": f"V{node_index + 1}",
                "processor": lane_label(processor),
                "release_time_ms": task.instance_dict_list[app_index][instance_index]["arr"],
                "deadline_ms": task.instance_dict_list[app_index][instance_index]["ddl"],
                "start_time_ms": solution.task_start_time[pos],
                "finish_time_ms": solution.task_end_time[pos],
                "execution_time_ms": solution.task_end_time[pos] - solution.task_start_time[pos],
                "reconfiguration_required": int(solution.task_DPR[pos]),
                "reconfiguration_start_ms": solution.task_DPR_start_time[pos] if solution.task_DPR[pos] else "",
                "reconfiguration_finish_ms": solution.task_DPR_end_time[pos] if solution.task_DPR[pos] else "",
                "fpga_configuration": configuration_label(task, pos)
                if processor >= task.CPU_number + task.NPU_number
                else "",
            }
        )
    write_csv_rows(
        output_dir / "final_schedule.csv",
        [
            "task",
            "application",
            "instance",
            "node",
            "processor",
            "release_time_ms",
            "deadline_ms",
            "start_time_ms",
            "finish_time_ms",
            "execution_time_ms",
            "reconfiguration_required",
            "reconfiguration_start_ms",
            "reconfiguration_finish_ms",
            "fpga_configuration",
        ],
        rows,
    )


def write_configuration_reuse(output_dir: Path, events):
    write_csv_rows(
        output_dir / "configuration_reuse.csv",
        [
            "fpga_box",
            "previous_task",
            "next_task",
            "configuration",
            "reuse",
            "saved_reconfiguration_ms",
        ],
        events,
    )


def write_leo_case_metrics(
    output_dir: Path,
    task,
    first_feasible_iteration,
    initial_best,
    final_before_opt,
    final_after_opt,
    reuse_events,
):
    rows = [
        {"metric": "Hyperperiod", "value": task.H, "unit": "ms"},
        {"metric": "EO period", "value": task.app_list[0].period, "unit": "ms"},
        {"metric": "SAR period", "value": task.app_list[1].period, "unit": "ms"},
        {"metric": "EO instances", "value": task.instance_number_list[0], "unit": "-"},
        {"metric": "SAR instances", "value": task.instance_number_list[1], "unit": "-"},
        {"metric": "Task instances", "value": task.task_number, "unit": "-"},
        {"metric": "CPU count", "value": task.CPU_number, "unit": "-"},
        {"metric": "TPE count", "value": task.NPU_number, "unit": "-"},
        {"metric": "FPGA box count", "value": task.RP_number, "unit": "-"},
        {"metric": "Population size", "value": POP_SIZE, "unit": "-"},
        {"metric": "Phase length", "value": PHASE_ITER, "unit": "iterations"},
        {"metric": "Number of phases", "value": NUM_PHASES, "unit": "-"},
        {"metric": "Max iterations", "value": MAX_ITER, "unit": "-"},
        {
            "metric": "First feasible iteration",
            "value": "" if first_feasible_iteration is None else first_feasible_iteration,
            "unit": "-",
        },
        {"metric": "Initial best delay", "value": initial_best.calc_delay(task), "unit": "ms"},
        {"metric": "Final delay", "value": final_after_opt.calc_delay(task), "unit": "ms"},
        {"metric": "Final energy", "value": final_after_opt.calc_energy(task), "unit": "mJ"},
        {
            "metric": "Final FPGA reconfigurations",
            "value": int(sum(final_after_opt.task_DPR)),
            "unit": "count",
        },
        {
            "metric": "Configuration reuse events",
            "value": len(reuse_events),
            "unit": "count",
        },
    ]
    write_csv_rows(output_dir / "leo_case_metrics.csv", ["metric", "value", "unit"], rows)


def write_case_result_summary(
    output_dir: Path,
    task,
    initial_best,
    first_feasible_iteration,
    first_feasible_best,
    final_before_opt,
    final_after_opt,
    phase_rows,
    reuse_events,
):
    path = output_dir / "case_result_summary.txt"
    initial_to_final = diff_mapping(task, initial_best, final_before_opt)
    with path.open("w", encoding="utf-8") as f:
        f.write("Platform\n")
        f.write("1 CPU, 1 TPE, 2 FPGA boxes\n\n")
        f.write("Hyperperiod\n")
        f.write(f"{task.H} ms\n\n")
        f.write("Periods\n")
        f.write(f"EO = {task.app_list[0].period} ms\n")
        f.write(f"SAR = {task.app_list[1].period} ms\n\n")
        f.write("Instances\n")
        f.write(f"EO = {task.instance_number_list[0]}\n")
        f.write(f"SAR = {task.instance_number_list[1]}\n")
        f.write(f"Total task instances = {task.task_number}\n\n")
        f.write("QHA settings\n")
        f.write(f"Population size = {POP_SIZE}\n")
        f.write(f"Phase length = {PHASE_ITER}\n")
        f.write(f"Number of phases = {NUM_PHASES}\n")
        f.write(f"Maximum iterations = {MAX_ITER}\n")
        f.write(f"Seed = {SEED}\n\n")

        f.write("Initial best\n")
        f.write(f"- energy = {initial_best.calc_energy(task):.6f} mJ\n")
        f.write(f"- delay = {initial_best.calc_delay(task):.6f} ms\n")
        f.write(f"- mapping = {format_mapping_inline(task, initial_best)}\n\n")

        f.write("First feasible\n")
        if first_feasible_best is None:
            f.write("- not found within 50 iterations\n\n")
        else:
            f.write(f"- iteration = {first_feasible_iteration}\n")
            f.write(f"- energy = {first_feasible_best.calc_energy(task):.6f} mJ\n")
            f.write(f"- delay = {first_feasible_best.calc_delay(task):.6f} ms\n")
            f.write(f"- mapping = {format_mapping_inline(task, first_feasible_best)}\n\n")

        f.write("Final before energy optimization\n")
        f.write(f"- energy = {final_before_opt.calc_energy(task):.6f} mJ\n")
        f.write(f"- delay = {final_before_opt.calc_delay(task):.6f} ms\n")
        f.write(f"- mapping = {format_mapping_inline(task, final_before_opt)}\n")
        f.write(f"- FPGA reconfiguration count = {int(sum(final_before_opt.task_DPR))}\n\n")

        f.write("Final after energy optimization\n")
        f.write(f"- energy = {final_after_opt.calc_energy(task):.6f} mJ\n")
        f.write(f"- delay = {final_after_opt.calc_delay(task):.6f} ms\n")
        f.write(f"- FPGA reconfiguration count = {int(sum(final_after_opt.task_DPR))}\n\n")

        f.write("Initial -> final mapping changes\n")
        if initial_to_final:
            for label, src, dst in initial_to_final:
                f.write(f"- {label}: {src} -> {dst}\n")
        else:
            f.write("- none\n")
        f.write("\n")

        f.write(f"First feasible iteration = {first_feasible_iteration}\n")
        f.write(f"Final cumulative deadline violation = {final_after_opt.calc_delay(task):.6f} ms\n")
        f.write(f"Total FPGA reconfiguration count = {int(sum(final_after_opt.task_DPR))}\n")
        f.write(f"Configuration reuse events = {len(reuse_events)}\n")
        if reuse_events:
            for event in reuse_events:
                f.write(
                    f"- {event['fpga_box']}: {event['previous_task']} -> {event['next_task']} "
                    f"reused {event['configuration']} (saved {event['saved_reconfiguration_ms']} ms)\n"
                )
        f.write("\n")

        current_phase = None
        for row in phase_rows:
            phase = row["phase"]
            if current_phase != phase:
                if current_phase is not None:
                    f.write("\n")
                current_phase = phase
                selected_ops = [phase_row["operator"] for phase_row in phase_rows if phase_row["phase"] == phase and phase_row["selected"]]
                f.write(f"Phase {phase}\n")
                f.write(f"- selected ROs = {selected_ops}\n")
            after_text = (
                f"{row['probability_after']:.6f}"
                if isinstance(row["probability_after"], float) and not math.isnan(row["probability_after"])
                else "N/A"
            )
            f.write(
                f"- {row['operator']} ({row['code_function']}): "
                f"probability {row['probability_before']:.6f} -> {after_text}, "
                f"usage / improvement = {row['usage_count']} / {row['improvement_count']}\n"
            )


def to_plain_phase_rows(phase_data):
    rows = []
    for phase_info in phase_data:
        selected_set = set(phase_info["selected"])
        for idx in range(NUM_HEURISTICS):
            usage_count = int(phase_info["usage_delta"][idx])
            improvement_count = int(phase_info["improvement_delta"][idx])
            improvement_rate = (
                improvement_count / usage_count if usage_count > 0 else float("nan")
            )
            rows.append(
                {
                    "phase": phase_info["phase"],
                    "operator": OPERATOR_METADATA[idx]["paper_RO"],
                    "code_function": OPERATOR_METADATA[idx]["code_function"],
                    "selected": idx in selected_set,
                    "usage_count": usage_count,
                    "improvement_count": improvement_count,
                    "improvement_rate": improvement_rate,
                    "probability_before": float(phase_info["probability_before"][idx]),
                    "probability_after": float(phase_info["probability_after"][idx])
                    if phase_info["probability_after"] is not None
                    else float("nan"),
                }
            )
    return rows


def build_iteration_row(iteration, phase, best_solution, energy, delay):
    feasible_indices = [idx for idx, value in enumerate(delay) if value == 0]
    feasible_energies = [energy[idx] for idx in feasible_indices]
    best_feasible_energy = min(feasible_energies) if feasible_energies else float("nan")
    mean_feasible_energy = float(np.mean(feasible_energies)) if feasible_energies else float("nan")
    return {
        "iteration": iteration,
        "phase": phase,
        "best_energy_mJ": float(best_solution["energy"]),
        "best_delay_ms": float(best_solution["delay"]),
        "best_is_feasible": bool(best_solution["delay"] == 0),
        "population_min_delay_ms": float(np.min(delay)),
        "population_mean_delay_ms": float(np.mean(delay)),
        "population_feasible_count": int(len(feasible_indices)),
        "population_feasible_ratio": float(len(feasible_indices) / len(delay)),
        "best_feasible_energy_mJ": best_feasible_energy,
        "population_mean_feasible_energy_mJ": mean_feasible_energy,
    }


def finalize_phase_record(phase_info, quantum_space, probability_after):
    phase_info["usage_delta"] = quantum_space.usage_count.astype(int) - phase_info["usage_start"]
    phase_info["improvement_delta"] = quantum_space.pf.astype(int) - phase_info["improvement_start"]
    phase_info["probability_after"] = probability_after
    del phase_info["usage_start"]
    del phase_info["improvement_start"]


def plot_leo_case_optimization(
    output_dir: Path,
    iteration_rows,
    first_feasible_iteration,
):
    iterations = [row["iteration"] for row in iteration_rows]
    best_delay = [row["best_delay_ms"] for row in iteration_rows]
    feasible_best_energy = [
        row["best_energy_mJ"] if row["best_is_feasible"] else float("nan")
        for row in iteration_rows
    ]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True, constrained_layout=True)
    phase_boundaries = [5, 10, 15, 20, 25, 30, 35, 40, 45]

    ax = axes[0]
    ax.plot(iterations, best_delay, color="black", marker="o", markersize=3, linewidth=1.3)
    if max(best_delay) > 1_000_000:
        ax.set_yscale("symlog", linthresh=1000)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.9)
    for boundary in phase_boundaries:
        ax.axvline(boundary, color="#BFBFBF", linestyle="--", linewidth=0.8)
    if first_feasible_iteration is not None:
        feasible_index = iterations.index(first_feasible_iteration)
        ax.scatter(
            [first_feasible_iteration],
            [best_delay[feasible_index]],
            color="#D55E00",
            s=28,
            zorder=5,
        )
        ax.annotate(
            "First feasible",
            xy=(first_feasible_iteration, best_delay[feasible_index]),
            xytext=(first_feasible_iteration + 2, best_delay[feasible_index] + max(1, best_delay[feasible_index] * 0.15)),
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            fontsize=8,
        )
    ax.set_ylabel("Best cumulative deadline violation (ms)", fontsize=9)
    ax.set_title("(a) Deadline feasibility", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    ax = axes[1]
    ax.plot(iterations, feasible_best_energy, color="#1F77B4", marker="o", markersize=3, linewidth=1.3)
    for boundary in phase_boundaries:
        ax.axvline(boundary, color="#BFBFBF", linestyle="--", linewidth=0.8)
    feasible_points = [idx for idx, value in enumerate(feasible_best_energy) if not np.isnan(value)]
    if feasible_points:
        final_idx = feasible_points[-1]
        ax.scatter([iterations[final_idx]], [feasible_best_energy[final_idx]], color="#D55E00", s=28, zorder=5)
        ax.annotate(
            "Final",
            xy=(iterations[final_idx], feasible_best_energy[final_idx]),
            xytext=(iterations[final_idx] - 7, feasible_best_energy[final_idx] * 1.03),
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            fontsize=8,
        )
    else:
        ax.text(0.5, 0.5, "No feasible solution found", transform=ax.transAxes, ha="center", va="center", fontsize=8)
    ax.set_ylabel("Best feasible energy (mJ)", fontsize=9)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_title("(b) Energy optimization", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    pdf_path = output_dir / "leo_case_optimization.pdf"
    png_path = output_dir / "leo_case_optimization.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    return pdf_path, png_path


def plot_leo_case_schedule(output_dir: Path, task, solution, reuse_events):
    del reuse_events

    def schedule_tick_step(hyperperiod_ms: int) -> int:
        if hyperperiod_ms <= 10_000:
            return 500
        if hyperperiod_ms <= 20_000:
            return 1_000
        return max(1_000, int(math.ceil(hyperperiod_ms / 10_000.0)) * 500)

    def label_offsets_for_lane(processor: int) -> list[tuple[float, float, str]]:
        if processor == 0:
            return [(-220.0, 0.72, "right"), (170.0, 0.50, "left"), (150.0, 0.68, "left")]
        if processor == 1:
            return [(-140.0, 0.56, "right"), (140.0, 0.74, "left")]
        if processor == 2:
            return [(-150.0, 0.56, "right"), (150.0, 0.74, "left")]
        return [(-140.0, 0.82, "right"), (150.0, 0.62, "left"), (210.0, 0.94, "left")]

    def allocate_staggered_annotation(
        lane_counter: dict[int, int],
        processor: int,
    ) -> tuple[float, float, str]:
        lane_offsets = label_offsets_for_lane(processor)
        offset = lane_offsets[lane_counter[processor] % len(lane_offsets)]
        lane_counter[processor] += 1
        return offset

    explicit_label_positions = {
        "SAR1-V1": {"dx": -210.0, "dy": 0.72, "ha": "right"},
        "EO1-V1": {"dx": 165.0, "dy": 0.50, "ha": "left"},
        "EO1-V2": {"dx": 185.0, "dy": 0.72, "ha": "left"},
        "EO2-V1": {"dx": -145.0, "dy": 0.82, "ha": "right"},
        "EO2-V2": {"dx": 160.0, "dy": 0.62, "ha": "left"},
        "SAR1-V3": {"dx": 150.0, "dy": 0.70, "ha": "left"},
    }

    fig, ax = plt.subplots(figsize=(10.5, 4.4), constrained_layout=True)
    lane_y = {0: 3, 1: 2, 2: 1, 3: 0}
    bar_height = 0.62
    short_threshold = task.H * 0.015
    interior_threshold = 250.0
    staggered_label_counter = {processor: 0 for processor in lane_y}

    for processor in range(task.processor_number):
        y = lane_y[processor]
        ax.hlines(y, 0, task.H, colors="#DDDDDD", linewidth=0.8, zorder=0)

    for pos in range(task.task_number):
        processor = solution.task_mapping[pos]
        app_index, _instance_index, _node_index = calc_subtask_instance_from_position(task, pos)
        app_name = app_short_name(app_index)
        style = APP_STYLE[app_name]
        start = solution.task_start_time[pos]
        finish = solution.task_end_time[pos]
        width = finish - start
        y = lane_y[processor]
        rect = plt.Rectangle(
            (start, y - bar_height / 2),
            width,
            bar_height,
            facecolor=style["color"],
            edgecolor="black",
            linewidth=0.8,
            zorder=2,
        )
        ax.add_patch(rect)

        label = math_node_label(task, pos)
        plain_label = plain_node_label(task, pos)
        x_anchor = start + width / 2
        if width >= interior_threshold or plain_label in {"EO1-V3", "EO2-V3"}:
            ax.text(
                x_anchor,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=style["text_color"],
                zorder=3,
            )
        elif width >= short_threshold:
            ax.annotate(
                label,
                xy=(x_anchor, y + bar_height / 2),
                xytext=(x_anchor, y + 0.48),
                ha="center",
                va="bottom",
                fontsize=7,
                arrowprops={"arrowstyle": "-", "linewidth": 0.6, "shrinkA": 0, "shrinkB": 0},
                zorder=3,
            )
        else:
            placement = explicit_label_positions.get(plain_label)
            if placement is None:
                dx, dy, ha = allocate_staggered_annotation(
                    lane_counter=staggered_label_counter,
                    processor=processor,
                )
            else:
                dx = placement["dx"]
                dy = placement["dy"]
                ha = placement["ha"]
            x_text = x_anchor + dx
            y_text = y + dy
            ax.annotate(
                label,
                xy=(x_anchor, y + bar_height / 2),
                xytext=(x_text, y_text),
                ha=ha,
                va="bottom",
                fontsize=7,
                arrowprops={"arrowstyle": "-", "linewidth": 0.6, "shrinkA": 0, "shrinkB": 0},
                zorder=3,
            )

        if solution.task_DPR[pos]:
            dpr_start = solution.task_DPR_start_time[pos]
            dpr_finish = solution.task_DPR_end_time[pos]
            dpr_rect = plt.Rectangle(
                (dpr_start, y - bar_height / 2),
                dpr_finish - dpr_start,
                bar_height,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                hatch="////",
                zorder=4,
            )
            ax.add_patch(dpr_rect)

    eo_boundary = task.app_list[0].period
    ax.axvline(eo_boundary, color="#666666", linestyle="--", linewidth=1.0)
    ax.text(eo_boundary + 110.0, 4.04, "EO boundary", ha="left", va="bottom", fontsize=8)

    ax.set_xlim(0, task.H)
    ax.set_ylim(-0.7, 4.15)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([LANE_LABELS[0], LANE_LABELS[1], LANE_LABELS[2], LANE_LABELS[3]], fontsize=8)
    tick_step = schedule_tick_step(task.H)
    ax.set_xticks(np.arange(0, task.H + tick_step, tick_step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1000:.1f}"))
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        Patch(facecolor=APP_STYLE["EO"]["color"], edgecolor="black", label="EO task"),
        Patch(facecolor=APP_STYLE["SAR"]["color"], edgecolor="black", label="SAR task"),
        Patch(facecolor="white", edgecolor="black", hatch="////", label="FPGA reconfiguration"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)

    pdf_path = output_dir / "leo_case_schedule.pdf"
    png_path = output_dir / "leo_case_schedule.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    return pdf_path, png_path


def run_case_study():
    random.seed(SEED)
    np.random.seed(SEED)

    reset_output_dir(OUTPUT_DIR)
    task = SatelliteCaseTaskConfig()

    population = quantum_chaotic_initialization(
        population_size=POP_SIZE,
        Task=task,
        ecs_initial=False,
    )
    quantum_space = QuantumHeuristicSpace(num_heuristics=NUM_HEURISTICS)

    energy = [sol.calc_energy(Task=task) for sol in population]
    delay = [sol.calc_delay(Task=task) for sol in population]
    fitness = fitness_function(energy, delay)
    best_idx = select_global_best_index(energy, delay, fitness)
    best_solution = copy.deepcopy(population[best_idx])
    best_energy = energy[best_idx]
    best_delay = delay[best_idx]

    initial_best = copy.deepcopy(best_solution)
    first_feasible_iteration = -1 if best_delay == 0 else None
    first_feasible_best = copy.deepcopy(best_solution) if best_delay == 0 else None

    best_solution_state = {
        "solution": best_solution,
        "energy": best_energy,
        "delay": best_delay,
    }

    iteration_rows = [
        build_iteration_row(
            iteration=-1,
            phase=0,
            best_solution=best_solution_state,
            energy=energy,
            delay=delay,
        )
    ]

    selected = quantum_space.observe()
    phase_data = []
    current_phase_info = {
        "phase": 1,
        "selected": list(selected),
        "probability_before": np.array([quantum_space.q_bits[1, i] ** 2 for i in range(NUM_HEURISTICS)], dtype=float),
        "probability_after": None,
        "usage_start": quantum_space.usage_count.astype(int).copy(),
        "improvement_start": quantum_space.pf.astype(int).copy(),
        "usage_delta": np.zeros(NUM_HEURISTICS, dtype=int),
        "improvement_delta": np.zeros(NUM_HEURISTICS, dtype=int),
    }

    for iteration in range(MAX_ITER):
        if iteration > 0 and iteration % PHASE_ITER == 0:
            probability_after = np.array([quantum_space.q_bits[1, i] ** 2 for i in range(NUM_HEURISTICS)], dtype=float)
            finalize_phase_record(current_phase_info, quantum_space, probability_after)
            phase_data.append(current_phase_info)

            quantum_space.update_q_bits(selected)
            selected = quantum_space.observe()
            current_phase = iteration // PHASE_ITER + 1
            current_phase_info = {
                "phase": current_phase,
                "selected": list(selected),
                "probability_before": np.array([quantum_space.q_bits[1, i] ** 2 for i in range(NUM_HEURISTICS)], dtype=float),
                "probability_after": None,
                "usage_start": quantum_space.usage_count.astype(int).copy(),
                "improvement_start": quantum_space.pf.astype(int).copy(),
                "usage_delta": np.zeros(NUM_HEURISTICS, dtype=int),
                "improvement_delta": np.zeros(NUM_HEURISTICS, dtype=int),
            }

        offspring = []
        elite_size = int(ELITISM_RATIO * POP_SIZE)
        elite_indices = np.argsort(fitness)[:elite_size]
        elite = [copy.deepcopy(population[i]) for i in elite_indices]

        prob = np.array([quantum_space.q_bits[1, i] ** 2 for i in selected], dtype=float)
        prob = prob / np.sum(prob)
        heuristic_idx = np.random.choice(selected, size=POP_SIZE, p=prob)
        for idx in heuristic_idx:
            quantum_space.usage_count[idx] += 1
        for idx, sol in enumerate(population):
            offspring.append(apply_heuristic(Task=task, solution=sol, idx=int(heuristic_idx[idx])))

        population = elite + offspring[: POP_SIZE - elite_size]

        old_energy = energy.copy()
        old_delay = delay.copy()
        energy = [sol.calc_energy(Task=task) for sol in population]
        delay = [sol.calc_delay(Task=task) for sol in population]
        for idx in range(len(population)):
            operator_idx = int(heuristic_idx[idx])
            if delay[idx] < old_delay[idx]:
                quantum_space.pf[operator_idx] += 1
            elif delay[idx] == old_delay[idx] and energy[idx] < old_energy[idx]:
                quantum_space.pf[operator_idx] += 1

        fitness = fitness_function(energy, delay)
        current_best_idx = select_global_best_index(energy, delay, fitness)
        candidate_solution = population[current_best_idx]
        candidate_energy = energy[current_best_idx]
        candidate_delay = delay[current_best_idx]
        if (
            candidate_energy <= best_solution_state["energy"]
            and candidate_delay == best_solution_state["delay"]
        ) or candidate_delay < best_solution_state["delay"]:
            best_solution_state = {
                "solution": copy.deepcopy(candidate_solution),
                "energy": candidate_energy,
                "delay": candidate_delay,
            }
            if first_feasible_best is None and candidate_delay == 0:
                first_feasible_iteration = iteration
                first_feasible_best = copy.deepcopy(candidate_solution)

        iteration_rows.append(
            build_iteration_row(
                iteration=iteration,
                phase=iteration // PHASE_ITER + 1,
                best_solution=best_solution_state,
                energy=energy,
                delay=delay,
            )
        )

    finalize_phase_record(current_phase_info, quantum_space, None)
    phase_data.append(current_phase_info)

    final_best_pre_opt = copy.deepcopy(best_solution_state["solution"])
    final_schedule_solution = copy.deepcopy(final_best_pre_opt)
    if final_schedule_solution.calc_delay(task) == 0:
        final_schedule_solution.energy_optimization(Task=task)

    phase_rows = to_plain_phase_rows(phase_data)
    snapshot_solutions = {"initial_best": initial_best}
    snapshot_iterations = {"initial_best": -1}
    if first_feasible_best is not None:
        snapshot_solutions["first_feasible_best"] = first_feasible_best
        snapshot_iterations["first_feasible_best"] = first_feasible_iteration
    snapshot_solutions["final_best_before_energy_optimization"] = final_best_pre_opt
    snapshot_iterations["final_best_before_energy_optimization"] = MAX_ITER - 1
    snapshot_solutions["final_best_after_energy_optimization"] = final_schedule_solution
    snapshot_iterations["final_best_after_energy_optimization"] = MAX_ITER - 1

    snapshot_rows = []
    snapshot_summary_rows = []
    for snapshot_name, solution in snapshot_solutions.items():
        iteration_value = snapshot_iterations[snapshot_name]
        snapshot_rows.extend(snapshot_task_rows(task, solution, snapshot_name, iteration_value))
        snapshot_summary_rows.append(
            {
                "snapshot": snapshot_name,
                "iteration": iteration_value,
                "energy_mJ": solution.calc_energy(task),
                "delay_ms": solution.calc_delay(task),
                "reconfiguration_count": int(sum(solution.task_DPR)),
            }
        )

    reuse_events = count_configuration_reuse_events(task, final_schedule_solution)

    write_iteration_summary(OUTPUT_DIR, iteration_rows)
    write_phase_operator_summary(OUTPUT_DIR, phase_rows)
    write_solution_snapshots(OUTPUT_DIR, snapshot_rows, snapshot_summary_rows)
    write_mapping_evolution(OUTPUT_DIR, task, snapshot_solutions)
    write_final_schedule(OUTPUT_DIR, task, final_schedule_solution)
    write_configuration_reuse(OUTPUT_DIR, reuse_events)
    write_leo_case_metrics(
        OUTPUT_DIR,
        task,
        first_feasible_iteration,
        initial_best,
        final_best_pre_opt,
        final_schedule_solution,
        reuse_events,
    )
    write_case_result_summary(
        OUTPUT_DIR,
        task,
        initial_best,
        first_feasible_iteration,
        first_feasible_best,
        final_best_pre_opt,
        final_schedule_solution,
        phase_rows,
        reuse_events,
    )
    optimization_pdf, optimization_png = plot_leo_case_optimization(
        OUTPUT_DIR,
        iteration_rows,
        first_feasible_iteration,
    )
    schedule_pdf, schedule_png = plot_leo_case_schedule(
        OUTPUT_DIR,
        task,
        final_schedule_solution,
        reuse_events,
    )

    return {
        "task": task,
        "initial_best": initial_best,
        "first_feasible_iteration": first_feasible_iteration,
        "first_feasible_best": first_feasible_best,
        "final_best_pre_opt": final_best_pre_opt,
        "final_schedule_solution": final_schedule_solution,
        "phase_data": phase_data,
        "phase_rows": phase_rows,
        "reuse_events": reuse_events,
        "optimization_pdf": optimization_pdf,
        "optimization_png": optimization_png,
        "schedule_pdf": schedule_pdf,
        "schedule_png": schedule_png,
    }


if __name__ == "__main__":
    result = run_case_study()
    task = result["task"]
    initial_best = result["initial_best"]
    final_best_pre_opt = result["final_best_pre_opt"]
    final_schedule_solution = result["final_schedule_solution"]
    first_feasible_iteration = result["first_feasible_iteration"]
    phase_data = result["phase_data"]
    reuse_events = result["reuse_events"]

    print("LEO Satellite Case Study")
    print("------------------------")
    print("Platform: 1 CPU, 1 TPE, 2 FPGA boxes")
    print(f"Hyperperiod: {task.H / 1000:.0f} s")
    print(f"Population: {POP_SIZE}")
    print(f"Phase length: {PHASE_ITER}")
    print(f"Iterations: {MAX_ITER}")
    print(
        f"First feasible iteration: "
        f"{first_feasible_iteration if first_feasible_iteration is not None else 'not found'}"
    )
    print(
        f"Initial best: delay={initial_best.calc_delay(task):.6f} ms, "
        f"energy={initial_best.calc_energy(task):.6f} mJ"
    )
    print(
        f"Final best: delay={final_schedule_solution.calc_delay(task):.6f} ms, "
        f"energy={final_schedule_solution.calc_energy(task):.6f} mJ"
    )
    print(f"FPGA reconfigurations: {int(sum(final_schedule_solution.task_DPR))}")
    print(f"Configuration reuse events: {len(reuse_events)}")
    print("Final mapping:")
    print(format_mapping_inline(task, final_best_pre_opt))
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")
