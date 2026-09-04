import random
from dataclasses import dataclass

import numpy as np

from task_definition import TaskConfig
from solution_utils import (
    MappingSolution,
    calc_subtask_instance_position,
    eligible_processor_ids,
    is_processor_eligible,
)


@dataclass
class CandidateMove:
    task: int
    source_processor: int
    target_processor: int
    delta_energy: float
    delta_slack: float
    priority: tuple
    solution: object


def build_heft_initial_solution(Task):
    solution = MappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number,
    )
    solution.heft_scheduling(Task)
    return solution


def build_solution_from_mapping(Task, mapping):
    candidate = MappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number,
    )
    candidate.init_definitely(mapping)
    candidate.heft_scheduling_for_certain_task_mapping(Task)
    return candidate


def get_instance_finish_times(Task, solution):
    finish_times = {}
    for app_index, app in enumerate(Task.app_list):
        sinks = app.sink_nodes()
        for instance_index in range(1, Task.instance_number_list[app_index] + 1):
            sink_positions = [
                calc_subtask_instance_position(Task, app_index, instance_index, sink)
                for sink in sinks
            ]
            finish_times[(app_index, instance_index)] = max(
                solution.task_end_time[pos] for pos in sink_positions
            )
    return finish_times


def format_mapping(solution):
    return solution.task_mapping.copy()


def evaluate_candidate_move(
    Task,
    current_solution,
    current_finish_times,
    task,
    target_processor,
    debug=False,
):
    source_processor = current_solution.task_mapping[task]
    if target_processor == source_processor:
        return None
    if not is_processor_eligible(Task, task, target_processor):
        return None

    current_energy = current_solution.calc_energy(Task)
    new_mapping = current_solution.task_mapping.copy()
    new_mapping[task] = target_processor
    candidate_solution = build_solution_from_mapping(Task, new_mapping)

    candidate_delay = candidate_solution.calc_delay(Task)
    if candidate_delay > 0:
        if debug:
            print(
                f"[SAFLA-A][debug] infeasible candidate task={task} "
                f"{source_processor}->{target_processor}, delay={candidate_delay}"
            )
        return None

    candidate_energy = candidate_solution.calc_energy(Task)
    delta_energy = current_energy - candidate_energy
    if delta_energy <= 0:
        if debug:
            print(
                f"[SAFLA-A][debug] non-improving candidate task={task} "
                f"{source_processor}->{target_processor}, delta_E={delta_energy}"
            )
        return None

    candidate_finish_times = get_instance_finish_times(Task, candidate_solution)
    delta_slack = max(
        candidate_finish_times[key] - current_finish_times[key]
        for key in current_finish_times
    )
    delta_slack = max(delta_slack, 0)

    if delta_slack == 0:
        priority = (1, delta_energy)
    else:
        priority = (0, delta_energy / delta_slack)

    return CandidateMove(
        task=task,
        source_processor=source_processor,
        target_processor=target_processor,
        delta_energy=delta_energy,
        delta_slack=delta_slack,
        priority=priority,
        solution=candidate_solution,
    )


def find_best_move_for_task(
    Task,
    current_solution,
    current_finish_times,
    task,
    debug=False,
):
    best_move = None
    for target_processor in eligible_processor_ids(Task, task):
        candidate = evaluate_candidate_move(
            Task,
            current_solution,
            current_finish_times,
            task,
            target_processor,
            debug=debug,
        )
        if candidate is None:
            continue
        if best_move is None or candidate.priority > best_move.priority:
            best_move = candidate
    return best_move


def sample_candidate_moves(Task, current_solution, candidate_sample_size):
    all_moves = []
    for task in range(Task.task_number):
        current_processor = current_solution.task_mapping[task]
        for target_processor in eligible_processor_ids(Task, task):
            if target_processor != current_processor:
                all_moves.append((task, target_processor))

    if len(all_moves) <= candidate_sample_size:
        return all_moves
    return random.sample(all_moves, candidate_sample_size)


def safla_adapted(
    Task,
    verbose=True,
    max_iterations=100,
    candidate_sample_size=50,
    debug=False,
):
    current_solution = build_heft_initial_solution(Task)
    initial_energy = current_solution.calc_energy(Task)
    initial_delay = current_solution.calc_delay(Task)

    if verbose:
        print("[SAFLA-A] Initial HEFT")
        print(f"Energy: {initial_energy:.6f}")
        print(f"Delay : {initial_delay}")
        print("[SAFLA-A] Search budget")
        print(f"Iterations : {max_iterations}")
        print(f"Candidates per iteration: {candidate_sample_size}")

    if initial_delay != 0:
        print(
            "[SAFLA-A] Warning: initial HEFT schedule is infeasible "
            f"(delay={initial_delay})."
        )

    history = []
    accepted_moves = 0
    iteration_count = 0
    max_no_improve_rounds = 5
    no_improve_rounds = 0
    previous_energy = initial_energy

    while iteration_count < max_iterations:
        current_energy = current_solution.calc_energy(Task)
        current_finish_times = get_instance_finish_times(Task, current_solution)

        sampled_moves = sample_candidate_moves(
            Task,
            current_solution,
            candidate_sample_size,
        )
        if not sampled_moves:
            break

        iteration_count += 1
        valid_moves = []
        for task, target_processor in sampled_moves:
            candidate = evaluate_candidate_move(
                Task,
                current_solution,
                current_finish_times,
                task,
                target_processor,
                debug=debug,
            )
            if candidate is not None:
                valid_moves.append(candidate)

        if verbose:
            print(f"\n[SAFLA-A] Iteration {iteration_count}")
            print(f"Sampled candidates : {len(sampled_moves)}")
            print(f"Valid candidates   : {len(valid_moves)}")

        if not valid_moves:
            no_improve_rounds += 1
            if verbose:
                print("[SAFLA-A] No improving move found in sampled candidates.")
                print(f"No-improve rounds: {no_improve_rounds}/{max_no_improve_rounds}")
            if no_improve_rounds >= max_no_improve_rounds:
                break
            continue

        best_move = max(valid_moves, key=lambda move: move.priority)
        new_solution = best_move.solution
        new_energy = new_solution.calc_energy(Task)
        new_delay = new_solution.calc_delay(Task)

        if new_delay != 0:
            raise RuntimeError(
                "Accepted move produced infeasible schedule, which should not happen."
            )
        if new_energy >= current_energy:
            raise RuntimeError(
                "Accepted move does not reduce energy, which should not happen."
            )

        current_solution = new_solution
        accepted_moves += 1
        no_improve_rounds = 0
        history.append(
            {
                "iteration": iteration_count,
                "accepted_move_index": accepted_moves,
                "task": best_move.task,
                "source_processor": best_move.source_processor,
                "target_processor": best_move.target_processor,
                "delta_energy": best_move.delta_energy,
                "delta_slack": best_move.delta_slack,
                "energy_before": current_energy,
                "energy_after": new_energy,
                "priority": best_move.priority,
            }
        )

        if new_energy >= previous_energy:
            raise RuntimeError("Energy history is not strictly decreasing.")
        previous_energy = new_energy

        if verbose:
            print(f"task       : {best_move.task}")
            print(
                f"processor  : {best_move.source_processor} -> "
                f"{best_move.target_processor}"
            )
            print(f"delta_E    : {best_move.delta_energy:.6f}")
            print(f"delta_slack: {best_move.delta_slack}")
            print(f"energy     : {current_energy:.6f} -> {new_energy:.6f}")

    final_energy = current_solution.calc_energy(Task)
    final_delay = current_solution.calc_delay(Task)
    reduction = 0.0
    if initial_energy != 0:
        reduction = (initial_energy - final_energy) / initial_energy * 100

    if verbose:
        print("\n[SAFLA-A] Finished")
        print(f"Iterations : {iteration_count}")
        print(f"Accepted moves: {accepted_moves}")
        print(f"HEFT energy: {initial_energy:.6f}")
        print(f"Final energy: {final_energy:.6f}")
        print(f"Reduction  : {reduction:.6f} %")
        print(f"Delay      : {final_delay}")

    return current_solution, history


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
        "tot": 30,
    }

    Task = TaskConfig()
    Task.init_class_vars(random_dict, mode="random")

    initial_solution = build_heft_initial_solution(Task)
    initial_energy = initial_solution.calc_energy(Task)
    initial_delay = initial_solution.calc_delay(Task)

    final_solution, history = safla_adapted(
        Task,
        verbose=True,
        max_iterations=100,
        candidate_sample_size=50,
    )
    final_energy = final_solution.calc_energy(Task)
    final_delay = final_solution.calc_delay(Task)

    reduction = 0.0
    if initial_energy != 0:
        reduction = (initial_energy - final_energy) / initial_energy * 100

    print("\n[SAFLA-A] HEFT mapping")
    print(format_mapping(initial_solution))
    print(f"[SAFLA-A] HEFT energy: {initial_energy:.6f}")
    print(f"[SAFLA-A] HEFT delay : {initial_delay}")

    print("\n[SAFLA-A] SAFLA-A mapping")
    print(format_mapping(final_solution))
    print(f"[SAFLA-A] SAFLA-A energy: {final_energy:.6f}")
    print(f"[SAFLA-A] SAFLA-A delay : {final_delay}")
    print(f"[SAFLA-A] Energy reduction: {reduction:.6f} %")

    print("\n[SAFLA-A] Accepted move history")
    if history:
        for item in history:
            print(
                f"iter={item['iteration']}, task={item['task']}, "
                f"{item['source_processor']}->{item['target_processor']}, "
                f"delta_E={item['delta_energy']:.6f}, "
                f"delta_slack={item['delta_slack']}, "
                f"energy={item['energy_before']:.6f}->{item['energy_after']:.6f}"
            )
    else:
        print("No energy-saving feasible move accepted.")

    try:
        final_solution.visualize_init(
            Task,
            window_name="SAFLA-Adapted Schedule",
            is_block=False,
        )
    except Exception as exc:
        print(f"[SAFLA-A] Visualization skipped due to error: {exc}")


if __name__ == "__main__":
    main()



