import math
import os
import random
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from GA import genetic_algorithm
from HHO import harris_hawks_optimization
from QHA import quantum_inspired_hyper_heuristics_algorithm
from SAFLA_adapted import safla_adapted
from eFLS_adapted import AdaptedEFLSMappingSolution
from solution_utils import MappingSolution, assert_mapping_eligible
from task_definition import TaskConfig


TMP_RESULT_DIR = os.path.join("results", "experiments", "tmp_validation")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def build_random_dict(N: int, V: int, tot: int, nwr: float):
    return {
        "N": N,
        "V": V,
        "alpha": 1.0,
        "max_out": 2,
        "beta": 1,
        "Tpr": 100,
        "Tpr_beta": 50,
        "wDAG": 400,
        "gamma_GPU": 2.0,
        "gamma_hw": 1.5,
        "P_sw": 0.5,
        "P_npu": 5.0,
        "P_npu_beta": 1.0,
        "P_static_min": 0.5,
        "P_static_max": 4.0,
        "P_dyn_min": 1.0,
        "P_dyn_max": 6.0,
        "npu_eligibility_prob": 0.5,
        "fpga_eligibility_prob": 0.5,
        "NWR": nwr,
        "tot": tot,
    }


def build_task(N: int, V: int, tot: int, nwr: float, seed: int):
    seed_everything(seed)
    task = TaskConfig()
    task.init_class_vars(build_random_dict(N, V, tot, nwr), mode="random")
    return task


def summarize_case(task: TaskConfig):
    cpu_only = 0
    cpu_npu = 0
    cpu_rp = 0
    cpu_npu_rp = 0
    cpu_exec = []
    npu_exec = []
    rp_exec = []
    tpr = []
    p_sw = []
    p_npu = []
    p_static = []
    p_dyn = []

    for app in task.app_list:
        for node in range(app.node_num):
            cpu_exec.append(app.sw_exe[node])
            tpr.append(app.tpr[node])
            p_sw.append(app.P_sw[node])
            p_npu.append(app.P_npu[node])
            p_static.append(app.P_static[node])
            p_dyn.append(app.P_dyn[node])

            has_npu = bool(app.npu_eligible[node])
            has_rp = bool(app.hw_eligible[node])
            if has_npu:
                npu_exec.append(app.npu_exe[node])
            if has_rp:
                rp_exec.append(app.hw_exe[node])

            if not has_npu and not has_rp:
                cpu_only += 1
            elif has_npu and not has_rp:
                cpu_npu += 1
            elif not has_npu and has_rp:
                cpu_rp += 1
            else:
                cpu_npu_rp += 1

    print(f"H={task.H}, periods={task.periods}, instances={task.instance_number_list}, M={task.task_number}")
    print(f"target_NWR={task.target_NWR}, actual_NWR={task.actual_NWR:.6f}")
    print(
        "eligibility_counts="
        f"CPU_only:{cpu_only}, CPU+NPU:{cpu_npu}, CPU+RP:{cpu_rp}, CPU+NPU+RP:{cpu_npu_rp}"
    )
    print(f"CPU WCET range=[{min(cpu_exec)}, {max(cpu_exec)}]")
    print(
        "NPU WCET range="
        f"[{min(npu_exec)}, {max(npu_exec)}]" if npu_exec else "NPU WCET range=[none]"
    )
    print(
        "RP WCET range="
        f"[{min(rp_exec)}, {max(rp_exec)}]" if rp_exec else "RP WCET range=[none]"
    )
    print(f"Tpr mean={np.mean(tpr):.4f}, std={np.std(tpr):.4f}")
    print(f"P_sw unique={sorted(set(round(x, 4) for x in p_sw))}")
    print(f"P_npu mean={np.mean(p_npu):.4f}, std={np.std(p_npu):.4f}")
    print(f"P_static range=[{min(p_static):.4f}, {max(p_static):.4f}]")
    print(f"P_dyn range=[{min(p_dyn):.4f}, {max(p_dyn):.4f}]")


def snapshot_apps(task: TaskConfig):
    snapshot = []
    for app in task.app_list:
        snapshot.append(
            {
                "graph": app.graph,
                "reverse_graph": app.reverse_graph,
                "into_degree": app.into_degree,
                "out_degree": app.out_degree,
                "sw_exe": app.sw_exe,
                "npu_exe": app.npu_exe,
                "hw_exe": app.hw_exe,
                "tpr": app.tpr,
                "P_sw": app.P_sw,
                "P_npu": app.P_npu,
                "P_static": app.P_static,
                "P_dyn": app.P_dyn,
                "cpu_eligible": app.cpu_eligible,
                "npu_eligible": app.npu_eligible,
                "hw_eligible": app.hw_eligible,
            }
        )
    return snapshot


def validate_nwr_changes_only_periods(seed: int = 24):
    task_nwr1 = build_task(2, 10, 100, 1.0, seed)
    task_nwr3 = build_task(2, 10, 100, 3.0, seed)

    assert snapshot_apps(task_nwr1) == snapshot_apps(task_nwr3), (
        "Changing NWR should not change DAG structure, eligibility, WCET, or power parameters"
    )
    assert task_nwr1.task_number == task_nwr3.task_number == 100
    assert task_nwr1.periods != task_nwr3.periods or task_nwr1.H != task_nwr3.H
    print("NWR invariance check passed: only periods/hyperperiod-related quantities changed.")


def validate_generated_parameters(task: TaskConfig):
    for app in task.app_list:
        for node in range(app.node_num):
            assert app.cpu_eligible[node] is True
            assert 1 <= app.sw_exe[node] <= 800
            assert app.tpr[node] >= 1
            assert app.P_sw[node] == 0.5
            assert app.P_npu[node] > 0
            assert 0.5 <= app.P_static[node] <= 4.0
            assert 1.0 <= app.P_dyn[node] <= 6.0

            if app.npu_eligible[node]:
                assert 1 <= app.npu_exe[node] <= app.sw_exe[node]
            else:
                assert math.isinf(app.npu_exe[node])

            if app.hw_eligible[node]:
                assert 1 <= app.hw_exe[node] <= app.sw_exe[node]
            else:
                assert math.isinf(app.hw_exe[node])


def validate_algorithms(task: TaskConfig):
    os.makedirs(TMP_RESULT_DIR, exist_ok=True)

    heft = MappingSolution(task.CPU_number, task.NPU_number, task.RP_number, task.task_number)
    heft.heft_scheduling(task)
    assert_mapping_eligible(task, heft.task_mapping, context="validation HEFT")
    print("HEFT legality check passed.")

    safla_solution, _history = safla_adapted(task, verbose=False, max_iterations=3, debug=False)
    assert_mapping_eligible(task, safla_solution.task_mapping, context="validation SAFLA")
    print("SAFLA_adapted legality check passed.")

    efls_solution = AdaptedEFLSMappingSolution(
        task.CPU_number, task.NPU_number, task.RP_number, task.task_number, verbose=False
    )
    efls_solution.efls_scheduling(task)
    assert_mapping_eligible(task, efls_solution.task_mapping, context="validation eFLS")
    print("eFLS_adapted legality check passed.")

    ga_solution, _ = genetic_algorithm(
        task,
        save_dic=TMP_RESULT_DIR,
        seed=24,
        pop_size=5,
        max_iter=2,
        crossover_rate=0.8,
        mutation_rate=0.1,
        ecs_initial=False,
    )
    assert_mapping_eligible(task, ga_solution.task_mapping, context="validation GA")
    print("GA legality check passed.")

    qha_solution, _ = quantum_inspired_hyper_heuristics_algorithm(
        task,
        save_dic=TMP_RESULT_DIR,
        pop_size=5,
        max_iter=2,
        phase_iter=1,
        ecs_initial=False,
    )
    assert_mapping_eligible(task, qha_solution.task_mapping, context="validation QHA")
    print("QHA legality check passed.")

    hho_solution, _ = harris_hawks_optimization(
        task,
        save_dic=TMP_RESULT_DIR,
        n=5,
        max_iter=2,
        ecs_intial=False,
        is_modified=True,
    )
    assert_mapping_eligible(task, hho_solution.task_mapping, context="validation HHO")
    print("HHO legality check passed.")


def main():
    print("=== Case A: N=2, V=10, M=100, NWR=3.0 ===")
    case_a = build_task(2, 10, 100, 3.0, seed=24)
    validate_generated_parameters(case_a)
    summarize_case(case_a)
    print()

    print("=== Case B: N=2, V=10, M=500, NWR=3.0 ===")
    case_b = build_task(2, 10, 500, 3.0, seed=24)
    validate_generated_parameters(case_b)
    summarize_case(case_b)
    print()

    print("=== NWR Invariance Check ===")
    validate_nwr_changes_only_periods(seed=24)
    print()

    print("=== Algorithm Legality Check ===")
    validate_algorithms(case_a)
    print("All validations passed.")


if __name__ == "__main__":
    main()
