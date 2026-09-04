import math
from functools import reduce

import numpy as np

from DAGs import DAG

app2_period_dict = {
    (2, 5, 30): [(1, 5)],
    (2, 5, 40): [(1, 7), (3, 5)],
    (2, 5, 50): [(1, 9), (3, 7)],
    (2, 5, 60): [(1, 11), (5, 7)],
    (2, 5, 70): [(1, 13), (3, 11), (5, 9)],
    (2, 5, 80): [(1, 15), (3, 13), (5, 11), (7, 9)],
    (2, 5, 90): [(1, 17), (3, 15), (5, 13), (7, 11)],
    (2, 5, 100): [(1, 19), (3, 17), (7, 13), (9, 11)],
    (2, 10, 40): [(1, 3)],
    (2, 10, 70): [(1, 6), (2, 5)],
    (2, 10, 100): [(1, 9), (3, 7)],
    (2, 10, 200): [(1, 19), (3, 17), (7, 13), (9, 11)],
    (2, 10, 300): [(1, 29), (7, 23), (11, 19), (13, 17)],
    (2, 10, 400): [(1, 39), (3, 37), (7, 33), (11, 29), (13, 27), (17, 23), (19, 21)],
    (2, 10, 500): [(1, 49), (3, 47), (7, 43), (9, 41), (11, 39), (13, 37), (17, 33), (19, 31), (21, 29), (23, 27)],
    (2, 20, 100): [(2, 3)],
    (2, 20, 200): [(3, 7)],
    (2, 20, 300): [(4, 11), (7, 8)],
    (2, 20, 400): [(3, 17), (7, 13), (9, 11)],
    (2, 20, 500): [(2, 23), (3, 22), (4, 21), (6, 19), (7, 18), (8, 17), (9, 16), (11, 14), (12, 13)],
}


def calc_superperiod(application_list: list[DAG]):
    periods = [application.period for application in application_list]

    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)

    return reduce(lcm, periods)


def find_closest_ratio(initial_ratio, ratio_list):
    closest_ratio = None
    min_diff = float("inf")
    for k1, k2 in ratio_list:
        current_ratio = k2 / k1
        diff = abs(current_ratio - initial_ratio)
        if diff < min_diff:
            min_diff = diff
            closest_ratio = (k1, k2)
    return closest_ratio


def node_eligible_type_names(app: DAG, node_index: int):
    eligible = ["CPU"]
    if app.npu_eligible[node_index]:
        eligible.append("NPU")
    if app.hw_eligible[node_index]:
        eligible.append("RP")
    return eligible


def average_eligible_execution_time_for_priority(app: DAG, node_index: int):
    values = [app.sw_exe[node_index]]
    if app.npu_eligible[node_index]:
        values.append(app.npu_exe[node_index])
    if app.hw_eligible[node_index]:
        values.append(app.hw_exe[node_index] + app.tpr[node_index])
    return sum(values) / len(values)


def average_eligible_execution_cost_for_nwr(app: DAG, node_index: int, cpu_count: int, npu_count: int, rp_count: int):
    numerator = app.sw_exe[node_index] * cpu_count
    denominator = cpu_count
    if app.npu_eligible[node_index]:
        numerator += app.npu_exe[node_index] * npu_count
        denominator += npu_count
    if app.hw_eligible[node_index]:
        numerator += (app.hw_exe[node_index] + app.tpr[node_index]) * rp_count
        denominator += rp_count
    return numerator / denominator


def calculate_reference_workload(application_list: list[DAG], instance_number_list: list[int], cpu_count: int, npu_count: int, rp_count: int):
    workload = 0.0
    for app_index, app in enumerate(application_list):
        app_workload = sum(
            average_eligible_execution_cost_for_nwr(app, node_index, cpu_count, npu_count, rp_count)
            for node_index in range(app.node_num)
        )
        workload += app_workload * instance_number_list[app_index]
    return workload


def calculate_actual_nwr(application_list: list[DAG], instance_number_list: list[int], H: int, cpu_count: int, npu_count: int, rp_count: int):
    workload = calculate_reference_workload(application_list, instance_number_list, cpu_count, npu_count, rp_count)
    return workload / H if H > 0 else 0.0


def min_execution_processor_type(app_list: list[DAG], i, j):
    app = app_list[i]
    candidates = [("CPU", app.sw_exe[j])]
    if app.npu_eligible[j]:
        candidates.append(("NPU", app.npu_exe[j]))
    if app.hw_eligible[j]:
        candidates.append(("RP", app.hw_exe[j] + app.tpr[j]))
    return min(candidates, key=lambda item: item[1])[0]


def min_energy_processor_type(app_list: list[DAG], i, j):
    app = app_list[i]
    candidates = [("CPU", app.sw_exe[j] * app.P_sw[j])]
    if app.npu_eligible[j]:
        candidates.append(("NPU", app.npu_exe[j] * app.P_npu[j]))
    if app.hw_eligible[j]:
        candidates.append(("RP", (app.hw_exe[j] + app.tpr[j]) * app.P_static[j] + app.hw_exe[j] * app.P_dyn[j]))
    return min(candidates, key=lambda item: item[1])[0]


def E_max(application_list: list[DAG], CPU_number, NPU_number, RP_number, H, random_dict=None, rp_p_idle=0.5):
    total = CPU_number * H * random_dict["P_sw"]
    max_npu_power = max((max(app.P_npu) for app in application_list), default=0.0)
    total += NPU_number * H * max_npu_power
    max_rp_power = rp_p_idle
    for app in application_list:
        for node_index in range(app.node_num):
            max_rp_power = max(max_rp_power, app.P_static[node_index] + app.P_dyn[node_index])
    total += RP_number * H * max_rp_power
    return total


def _base_periods_for_two_apps(application_list: list[DAG], random_dict, cpu_count, npu_count, rp_count):
    workloads = [
        sum(
            average_eligible_execution_cost_for_nwr(app, node_index, cpu_count, npu_count, rp_count)
            for node_index in range(app.node_num)
        )
        for app in application_list
    ]
    ratio_candidates = app2_period_dict[(random_dict["N"], random_dict["V"], random_dict["tot"])]
    ratio = find_closest_ratio(max(workloads) / max(min(workloads), 1e-6), ratio_candidates)
    small_period, large_period = ratio

    if workloads[0] >= workloads[1]:
        base_periods = [large_period, small_period]
    else:
        base_periods = [small_period, large_period]

    H_base = calc_superperiod([_PeriodProxy(p) for p in base_periods])
    instance_number_list = [H_base // period for period in base_periods]
    generated_tasks = sum(instance_number_list[idx] * application_list[idx].node_num for idx in range(2))
    if generated_tasks != random_dict["tot"]:
        raise ValueError(
            f"Generated task count mismatch: expected tot={random_dict['tot']}, got {generated_tasks} "
            f"for base periods {base_periods} and instances {instance_number_list}"
        )
    return base_periods, instance_number_list, H_base


class _PeriodProxy:
    def __init__(self, period):
        self.period = period


def calc_periods(application_list: list[DAG], random_dict, cpu_count, npu_count, rp_count):
    if random_dict["N"] != 2:
        raise NotImplementedError("Current revised experiment setup expects N=2 applications.")

    base_periods, instance_number_list, H_base = _base_periods_for_two_apps(
        application_list,
        random_dict,
        cpu_count,
        npu_count,
        rp_count,
    )
    reference_workload = calculate_reference_workload(
        application_list,
        instance_number_list,
        cpu_count,
        npu_count,
        rp_count,
    )
    g_float = reference_workload / (random_dict["NWR"] * H_base)
    candidate_scales = {
        max(1, int(math.floor(g_float))),
        max(1, int(round(g_float))),
        max(1, int(math.ceil(g_float))),
    }
    best_scale = None
    best_error = float("inf")
    best_actual_nwr = None
    for scale in sorted(candidate_scales):
        periods = [scale * base for base in base_periods]
        H = calc_superperiod([_PeriodProxy(period) for period in periods])
        actual_nwr = reference_workload / H
        error = abs(actual_nwr - random_dict["NWR"])
        if error < best_error:
            best_error = error
            best_scale = scale
            best_actual_nwr = actual_nwr

    periods = [best_scale * base for base in base_periods]
    for index, app in enumerate(application_list):
        app.period = periods[index]
    return periods, reference_workload, best_actual_nwr


class TaskConfig:
    def __init__(self):
        self.CPU_number = 4
        self.NPU_number = 1
        self.RP_number = 2
        self.processor_number = self.CPU_number + self.NPU_number + self.RP_number

        self.generate_dict_list = []
        self.app_list = []
        self.H = 0
        self.instance_number_list = []
        self.task_number = 0
        self.instance_dict_list = []

        self.RP_P_idle = 0.5
        self.E_max = 0.0
        self.mapping_time_min = []
        self.mapping_energy_min = []
        self.target_NWR = 0.0
        self.actual_NWR = 0.0
        self.periods = []

    def init_class_vars(self, random_dict=None, mode="certain"):
        self.generate_dict_list = []
        self.instance_dict_list = []
        self.instance_number_list = []
        self.app_list = []
        self.mapping_time_min = []
        self.mapping_energy_min = []
        self.target_NWR = random_dict["NWR"] if random_dict else 0.0

        if mode == "certain":
            self.generate_dict_list.append(
                {
                    "node num": 7,
                    "edges": [(0, 4), (1, 4), (2, 3), (2, 5), (3, 6), (4, 6)],
                    "period": 20,
                    "sw exe": [4] * 7,
                    "npu exe": [4] * 7,
                    "hw exe": [3] * 7,
                    "tpr": [2] * 7,
                    "P static": [1.0] * 7,
                    "P dyn": [2.0] * 7,
                    "P sw": [0.5] * 7,
                    "P npu": [5.0] * 7,
                    "cpu eligible": [True] * 7,
                    "npu eligible": [True] * 7,
                    "hw eligible": [True] * 7,
                }
            )
            self.generate_dict_list.append(
                {
                    "node num": 7,
                    "edges": [(0, 4), (1, 4), (2, 3), (2, 5), (3, 6), (4, 6)],
                    "period": 30,
                    "sw exe": [3] * 7,
                    "npu exe": [2] * 7,
                    "hw exe": [2] * 7,
                    "tpr": [1] * 7,
                    "P static": [1.0] * 7,
                    "P dyn": [2.0] * 7,
                    "P sw": [0.5] * 7,
                    "P npu": [5.0] * 7,
                    "cpu eligible": [True] * 7,
                    "npu eligible": [True] * 7,
                    "hw eligible": [True] * 7,
                }
            )
            self.app_list = [DAG(generate_dict) for generate_dict in self.generate_dict_list]
        else:
            for _ in range(random_dict["N"]):
                self.app_list.append(DAG(random_dict, mode="random"))
            self.periods, _reference_workload, self.actual_NWR = calc_periods(
                self.app_list,
                random_dict,
                self.CPU_number,
                self.NPU_number,
                self.RP_number,
            )

        self.H = calc_superperiod(self.app_list)
        self.periods = [app.period for app in self.app_list]
        self.instance_number_list = [int(self.H / app.period) for app in self.app_list]
        if mode == "certain":
            self.actual_NWR = calculate_actual_nwr(
                self.app_list,
                self.instance_number_list,
                self.H,
                self.CPU_number,
                self.NPU_number,
                self.RP_number,
            )

        self.instance_dict_list = [{} for _ in range(len(self.app_list))]
        for app_index in range(len(self.app_list)):
            self.instance_dict_list[app_index] = {}
            for instance_index in range(self.instance_number_list[app_index]):
                self.instance_dict_list[app_index][instance_index + 1] = {
                    "arr": instance_index * self.app_list[app_index].period,
                    "ddl": (instance_index + 1) * self.app_list[app_index].period,
                }

        self.task_number = sum(num * app.node_num for num, app in zip(self.instance_number_list, self.app_list))
        self.E_max = E_max(
            self.app_list,
            self.CPU_number,
            self.NPU_number,
            self.RP_number,
            self.H,
            random_dict,
            rp_p_idle=self.RP_P_idle,
        )

        for app_index in range(len(self.app_list)):
            for _instance_index in range(self.instance_number_list[app_index]):
                for node_index in range(self.app_list[app_index].node_num):
                    self.mapping_time_min.append(min_execution_processor_type(self.app_list, app_index, node_index))
                    self.mapping_energy_min.append(min_energy_processor_type(self.app_list, app_index, node_index))


if __name__ == "__main__":
    random_dict = {
        "N": 2,
        "V": 10,
        "alpha": 1.0,
        "max_out": 2,
        "beta": 1,
        "Tpr": 100,
        "Tpr_beta": 50,
        "wDAG": 400,
        "gamma_GPU": 2.0,
        "gamma_hw": 1.5,
        "npu_eligibility_prob": 0.5,
        "fpga_eligibility_prob": 0.5,
        "P_sw": 0.5,
        "P_npu": 5.0,
        "P_npu_beta": 1.0,
        "P_static_min": 0.5,
        "P_static_max": 4.0,
        "P_dyn_min": 1.0,
        "P_dyn_max": 6.0,
        "NWR": 3.0,
        "tot": 100,
    }
    task = TaskConfig()
    task.init_class_vars(random_dict, mode="random")
    print(task.periods, task.H, task.instance_number_list, task.task_number, task.target_NWR, task.actual_NWR)
