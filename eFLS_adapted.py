import random
from collections import deque

import numpy as np

from task_definition import TaskConfig
from solution_utils import (
    MappingSolution,
    calc_RP_dynamic_energy,
    calc_subtask_instance_from_position,
    calc_subtask_instance_position,
    eligible_processor_ids,
    is_processor_eligible,
    priority_list_generate,
    sort_lists,
)


class AdaptedEFLSMappingSolution(MappingSolution):
    def __init__(self, CPU_number, NPU_number, RP_number, task_number, verbose=True):
        super().__init__(CPU_number, NPU_number, RP_number, task_number)
        self.verbose = verbose
        self.remaining_path_lb = [0] * task_number

    def compute_remaining_path_lb(self, Task):
        remaining_path_lb = [0] * Task.task_number

        for app_index, app in enumerate(Task.app_list):
            min_exec = [
                min(
                    [app.sw_exe[node]]
                    + ([app.npu_exe[node]] if app.npu_eligible[node] else [])
                    + ([app.hw_exe[node] + app.tpr[node]] if app.hw_eligible[node] else [])
                )
                for node in range(app.node_num)
            ]

            remaining_node_lb = [0] * app.node_num
            out_degree = app.out_degree.copy()
            queue = deque([node for node in range(app.node_num) if out_degree[node] == 0])

            while queue:
                node = queue.popleft()
                for predecessor in app.reverse_graph[node]:
                    candidate = min_exec[node] + remaining_node_lb[node]
                    if candidate > remaining_node_lb[predecessor]:
                        remaining_node_lb[predecessor] = candidate
                    out_degree[predecessor] -= 1
                    if out_degree[predecessor] == 0:
                        queue.append(predecessor)

            for instance_index in range(1, Task.instance_number_list[app_index] + 1):
                for node in range(app.node_num):
                    pos = calc_subtask_instance_position(Task, app_index, instance_index, node)
                    remaining_path_lb[pos] = remaining_node_lb[node]

        self.remaining_path_lb = remaining_path_lb
        return remaining_path_lb

    def partial_schedule(self, Task, mapped_tasks, task_list_sorted=None):
        self.clear_scheduling(Task.task_number)
        if task_list_sorted is None:
            priority_list = priority_list_generate(Task)
            task_list = list(range(Task.task_number))
            _, task_list_sorted = sort_lists(priority_list, task_list)

        for task in task_list_sorted:
            if task not in mapped_tasks:
                continue

            i, q, j = calc_subtask_instance_from_position(Task, task)
            arr = Task.instance_dict_list[i][q]["arr"]
            predecessor_end_time = arr
            predecessor = Task.app_list[i].predecessor_of_node(j)

            if predecessor:
                predecessor_positions = [
                    calc_subtask_instance_position(Task, i, q, pred_node) for pred_node in predecessor
                ]
                if any(pred_task not in mapped_tasks for pred_task in predecessor_positions):
                    raise RuntimeError("Mapped task has an unmapped predecessor")
                predecessor_end_time = max(
                    arr,
                    max(self.task_end_time[pred_task] for pred_task in predecessor_positions),
                )

            processor = self.task_mapping[task]
            if not is_processor_eligible(Task, task, processor):
                raise RuntimeError("Mapped task uses an ineligible processor")
            if processor < Task.CPU_number + Task.NPU_number:
                if processor < Task.CPU_number:
                    exe = Task.app_list[i].sw_exe[j]
                else:
                    exe = Task.app_list[i].npu_exe[j]
                self.task_start_time[task] = max(
                    predecessor_end_time,
                    self.processor_avail[processor],
                )
            else:
                need_dpr = True
                if self.processor_completed_task[processor]:
                    task_last = self.processor_completed_task[processor][-1]
                    i2, q2, j2 = calc_subtask_instance_from_position(Task, task_last)
                    if i2 == i and j2 == j:
                        need_dpr = False

                dpr_end_time = self.processor_avail[processor]
                if need_dpr:
                    dpr_start_time = max(
                        self.processor_avail[processor],
                        self.processor_RP_config_time,
                    )
                    dpr_end_time = dpr_start_time + Task.app_list[i].tpr[j]
                    self.task_DPR[task] = 1
                    self.task_DPR_start_time[task] = dpr_start_time
                    self.task_DPR_end_time[task] = dpr_end_time
                    self.processor_RP_config_time = dpr_end_time
                    self.processor_RP_task_list.append(task)
                    self.processor_RP_time_list.append([dpr_start_time, dpr_end_time])

                exe = Task.app_list[i].hw_exe[j]
                self.task_start_time[task] = max(predecessor_end_time, dpr_end_time)

            self.task_end_time[task] = self.task_start_time[task] + exe
            self.processor_completed_task[processor].append(task)
            self.processor_avail[processor] = self.task_end_time[task]

        return True

    @staticmethod
    def _overlap_length(start, end, cutoff_time):
        if start >= cutoff_time:
            return 0
        return max(0, min(end, cutoff_time) - start)

    def calc_partial_energy(self, Task, cutoff_time):
        energy = 0.0

        for processor in range(Task.processor_number):
            task_list = self.processor_completed_task[processor]

            if processor < Task.CPU_number:
                if not task_list:
                    continue
                for task in task_list:
                    i, q, j = calc_subtask_instance_from_position(Task, task)
                    active_time = self._overlap_length(
                        self.task_start_time[task],
                        self.task_end_time[task],
                        cutoff_time,
                    )
                    energy += active_time * Task.app_list[i].P_sw[j]
            elif processor < Task.CPU_number + Task.NPU_number:
                for task in task_list:
                    i, q, j = calc_subtask_instance_from_position(Task, task)
                    active_time = self._overlap_length(
                        self.task_start_time[task],
                        self.task_end_time[task],
                        cutoff_time,
                    )
                    energy += active_time * Task.app_list[i].P_npu[j]
            else:
                energy += calc_RP_dynamic_energy(
                    Task,
                    self.task_start_time,
                    self.task_end_time,
                    task_list,
                    self.task_DPR_start_time,
                    self.task_DPR_end_time,
                    self.task_DPR,
                    interval_end=cutoff_time,
                )

        return energy

    def deadline_feasible(self, Task, mapped_tasks):
        for task in mapped_tasks:
            i, q, j = calc_subtask_instance_from_position(Task, task)
            ddl = Task.instance_dict_list[i][q]["ddl"]
            if self.task_end_time[task] + self.remaining_path_lb[task] > ddl:
                return False
        return True

    def efls_scheduling(self, Task):
        self.compute_remaining_path_lb(Task)

        priority_list = priority_list_generate(Task)
        task_list = list(range(Task.task_number))
        _, task_list_sorted = sort_lists(priority_list, task_list)

        mapped_tasks = set()

        for current_task in task_list_sorted:
            best_candidate = None
            best_key = None

            if self.verbose:
                i, q, j = calc_subtask_instance_from_position(Task, current_task)
                print(
                    f"\n[eFLS-A] Current task {current_task} -> "
                    f"(app={i}, instance={q}, node={j})"
                )

            for processor in eligible_processor_ids(Task, current_task):
                candidate_mapping = self.task_mapping.copy()
                candidate_mapping[current_task] = processor
                candidate_mapped_tasks = set(mapped_tasks)
                candidate_mapped_tasks.add(current_task)

                candidate_solution = AdaptedEFLSMappingSolution(
                    Task.CPU_number,
                    Task.NPU_number,
                    Task.RP_number,
                    Task.task_number,
                    verbose=False,
                )
                candidate_solution.task_mapping = candidate_mapping.copy()
                candidate_solution.remaining_path_lb = self.remaining_path_lb.copy()

                try:
                    candidate_solution.partial_schedule(
                        Task,
                        candidate_mapped_tasks,
                        task_list_sorted=task_list_sorted,
                    )
                except RuntimeError:
                    if self.verbose:
                        print(
                            f"[eFLS-A]   processor={processor}, "
                            "feasible=False (unmapped predecessor)"
                        )
                    continue

                feasible = candidate_solution.deadline_feasible(Task, candidate_mapped_tasks)
                finish_time = candidate_solution.task_end_time[current_task]
                partial_energy = candidate_solution.calc_partial_energy(Task, finish_time)

                if self.verbose:
                    print(
                        f"[eFLS-A]   processor={processor}, "
                        f"finish={finish_time}, "
                        f"partial_energy={partial_energy:.6f}, "
                        f"feasible={feasible}"
                    )

                if not feasible:
                    continue

                candidate_key = (partial_energy, finish_time, processor)
                if best_candidate is None or candidate_key < best_key:
                    best_candidate = candidate_solution
                    best_key = candidate_key

            if best_candidate is None:
                if self.verbose:
                    print(
                        f"[eFLS-A] No deadline-aware feasible processor for task {current_task}. "
                        "Marking solution UNSCHEDULABLE."
                    )
                return False

            self.task_mapping[current_task] = best_candidate.task_mapping[current_task]
            mapped_tasks.add(current_task)

            if self.verbose:
                print(
                    f"[eFLS-A] Selected processor={self.task_mapping[current_task]} "
                    f"for task {current_task}"
                )

        self.clear_scheduling(Task.task_number)
        self.heft_scheduling_for_certain_task_mapping(Task)

        final_delay = self.calc_delay(Task)
        if final_delay > 0:
            print(
                "[eFLS-A] Warning: final full schedule violates deadlines even though "
                "all partial deadline-aware checks passed."
            )
        return final_delay == 0


def main():
    np.random.seed(24)
    random.seed(24)

    random_dict = {
        "N": 2,
        "V": 5,
        "alpha": 1.0,
        "max_out": 2,
        "beta": 1,
        "Tpr": 100,
        "Tpr_beta": 20,
        "wDAG": 400,
        "gamma_GPU": 2,
        "gamma_hw": 1.5,
        "P_sw": 90,
        "P_npu": 120,
        "P_npu_beta": 20,
        "P_static": 24,
        "P_static_beta": 2,
        "P_dyn": 60,
        "P_dyn_beta": 20,
        "NWR": 1.0,
        "tot": 30,
    }

    Task = TaskConfig()
    Task.init_class_vars(random_dict, mode="random")

    solution = AdaptedEFLSMappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number,
        verbose=True,
    )

    feasible = solution.efls_scheduling(Task)

    print("\n[eFLS-A] Summary")
    print(f"Task.H            : {Task.H}")
    print(f"Task.task_number  : {Task.task_number}")
    print(f"Final mapping     : {solution.task_mapping}")
    print(f"Final energy      : {solution.calc_energy(Task):.6f}")
    print(f"Final energy / Emax: {solution.calc_energy(Task) / Task.E_max:.6f}")
    print(f"Final delay       : {solution.calc_delay(Task)}")
    print(f"Feasible          : {feasible}")

    try:
        solution.visualize_init(
            Task,
            window_name="Adapted eFLS",
            is_block=False,
        )
    except Exception as exc:
        print(f"[eFLS-A] Visualization skipped due to error: {exc}")


if __name__ == "__main__":
    main()
