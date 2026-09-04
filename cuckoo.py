from solution_utils import (
    MappingSolution,
    assert_mapping_eligible,
    calc_subtask_instance_from_position,
    calc_subtask_instance_position,
    eligible_processor_ids,
    project_mapping_to_eligible_processors,
    sort_lists,
)
from task_definition import TaskConfig
from task_definition import average_eligible_execution_time_for_priority
import numpy as np
from scipy.special import gamma
import time
from collections import deque
import random
import matplotlib.pyplot as plt
from GA import plot_convergence


def build_projected_solution(Task: TaskConfig, raw_mapping, context="cuckoo candidate"):
    projected = project_mapping_to_eligible_processors(
        Task,
        simplebounds(raw_mapping, Task.task_number, Task.processor_number),
    )
    assert_mapping_eligible(Task, projected, context=context)
    solution = MappingSolution(Task.CPU_number, Task.NPU_number, Task.RP_number, Task.task_number)
    solution.init_definitely(projected)
    solution.heft_scheduling_for_certain_task_mapping(Task)
    return solution, projected

def cuckoo_search(Task:TaskConfig, save_dic="test", msg="", iterr=0,seed=24,n=25, tol=1.0e-5, max_iter=10000, ecs_initial=False):
    # ========== 记录收敛数据 ==========
    convergence = {
        'best_energy': [],
        'generation_best_energy': [],
        'best_delay': [],
        'avg_energy': [],
        'std_energy': [],
        'avg_delay': [],  # 新增平均延时
        'feasible_ratio': []  # 新增可行解比例
    }

    ### 生成随机初始解
    nest = chaotic_initialization(n, Task, ecs_initial=ecs_initial)

    ## 初始评估
    energy = [solution.calc_energy(Task) for solution in nest]
    time_delay = [solution.calc_delay(Task) for solution in nest]
    # 先找出所有可行解（delay == 0）的索引
    feasible_indices = [i for i, d in enumerate(time_delay) if d == 0]
    if feasible_indices:  # 如果存在可行解
        # 在可行解中找 fitness 最小的索引
        feasible_fitness = [energy[i] for i in feasible_indices]
        best_feasible_idx_in_subset = np.argmin(feasible_fitness)
        best_idx = feasible_indices[best_feasible_idx_in_subset]
    else:  # 如果没有可行解，则在整个种群中找 fitness 最小的
        best_idx = np.argmin(energy)
    bestnest, fmin = nest[best_idx], energy[best_idx]
    # print(f"最小能耗{fmin}")
    # 记录初始状态（增强）
    convergence['best_energy'].append(fmin)
    convergence['generation_best_energy'].append(fmin)
    convergence['best_delay'].append(time_delay[best_idx])
    convergence['avg_energy'].append(np.mean(energy))
    convergence['std_energy'].append(np.std(energy))
    convergence['avg_delay'].append(np.mean(time_delay))
    convergence['feasible_ratio'].append(np.mean(np.array(time_delay) == 0))

    # bestnest.visualize_init(window_name="best scheduler", is_block=True)

    ## 主循环
    N_iter = 0
    fitness = fitness_function(energy)
    transformed_fitness = transformed_fitness_function(energy,time_delay,fitness)
    best = MappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number
    )
    best.init_definitely(bestnest.task_mapping.copy())
    best.heft_scheduling_for_certain_task_mapping(Task)
    while fmin > tol and N_iter < max_iter:
        # print(N_iter)
        # 高斯随机游走
        new_nest = gaussian_random_walk(Task, nest, transformed_fitness, best, N_iter+1)

        # 评估新群体
        new_energy = [solution.calc_energy() for solution in new_nest]
        new_delay = [solution.calc_delay() for solution in new_nest]
        new_fitness = fitness_function(new_energy)

        # 应用Algorithm 3选择更优的个体
        nest, fitness, time_delay = choose_best_individual(
            nest, new_nest,
            fitness, new_fitness,
            np.array(time_delay), np.array(new_delay))
        energy = [solution.calc_energy(Task) for solution in nest]
        transformed_fitness = transformed_fitness_function(energy, time_delay, fitness)

        # 通过Levy飞行生成新解
        # new_nest = get_cuckoos(nest, bestnest, Lb, Ub)
        new_nest = get_cuckoos(Task, nest, best)
        best, nest, fitness,transformed_fitness = get_best_nest(Task, nest, new_nest, fitness,transformed_fitness)

        # 发现并丢弃较差解
        new_nest = empty_nests(Task, nest, fitness)
        energy = [solution.calc_energy(Task) for solution in nest]
        time_delay = [solution.calc_delay(Task) for solution in nest]
        fitness = fitness_function(energy)
        transformed_fitness = transformed_fitness_function(energy, time_delay, fitness)
        best, nest, fitness,transformed_fitness = get_best_nest(Task, nest, new_nest, fitness,transformed_fitness)

        fnew = best.calc_energy(Task)
        N_iter += 1

        # 计算当前种群指标
        # current_delay = best.calc_delay()
        # pop_energy = [sol.calc_energy() for sol in nest]
        # pop_delays = [sol.calc_delay() for sol in nest]

        # 更新全局最优
        if (fnew < fmin and best.calc_delay(Task) <= bestnest.calc_delay(Task)) or best.calc_delay(Task) < bestnest.calc_delay(Task):
            fmin = fnew
            bestnest = best

        # 记录当前迭代数据
        convergence['best_energy'].append(fmin)
        convergence['generation_best_energy'].append(best.calc_energy(Task))
        convergence['best_delay'].append(bestnest.calc_delay(Task))
        convergence['avg_energy'].append(np.mean(energy))
        convergence['std_energy'].append(np.std(energy))
        convergence['avg_delay'].append(np.mean(time_delay))
        convergence['feasible_ratio'].append(np.mean(np.array(time_delay) == 0))

        # for solution in nest:
        #     print(solution.task_mapping)
        # print(nest[0].task_mapping)
        # print(f"iter{N_iter} 最小能耗{fmin}")
        # bestnest.visualize_init(window_name="best scheduler", is_block=True)

    # print(f"Total number of iterations: {N_iter}")
    # print(f"Minimum value found: {fmin}")
    # print(f"Best solution: {bestnest}")
    # print(count)
    bestnest.visualize_init(Task, window_name=f".\\{save_dic}\\cuckoo scheduler {msg} iter{iterr} energy{bestnest.calc_energy(Task)} energy_percent{bestnest.calc_energy(Task) / Task.E_max} delay{bestnest.calc_delay(Task)}", is_save=True)

    # 绘制收敛曲线
    plot_convergence(convergence,plot_number=3,save_name=f".\\{save_dic}\\cuckoo scheduler {msg} iter{iterr} convergence history.png")
    cost_time_modification(Task, bestnest)
    fmin = bestnest.calc_energy(Task)

    return bestnest, fmin


def chaotic_initialization(population_size, Task:TaskConfig, max_chaos_iter=100, ecs_initial=True):
    """
    混沌初始化种群（适配MappingSolution类）
    :param population_size: 种群大小 (n)
    :param max_chaos_iter: 最大混沌迭代次数
    :return: 初始化后的种群列表（MappingSolution对象列表）
    """
    # 计算总资源类型数量（假设有CPU/NPU/RP三种资源）
    total_resource_types = Task.CPU_number + Task.NPU_number + Task.RP_number

    # 初始化种群
    population = []

    for i in range(population_size):
        # 创建新的解决方案对象
        solution = MappingSolution(
            Task.CPU_number,
            Task.NPU_number,
            Task.RP_number,
            Task.task_number
        )

        if i == 0 and ecs_initial:
            solution.ecs_scheduling(Task)
        else:
            # 为每个任务生成混沌分配方案
            task_assignments = []
            for j in range(Task.task_number):
                # 生成初始混沌变量 (0 < c_0 < 1)
                c = np.random.uniform(0.001, 0.999)

                # 正弦混沌迭代
                for _ in range(max_chaos_iter):
                    c = np.sin(np.pi * c)

                # 映射到资源类型索引 [0, total_resource_types-1]
                eligible = eligible_processor_ids(Task, j)
                resource_idx = eligible[int(np.floor(c * len(eligible))) % len(eligible)]

                task_assignments.append(resource_idx)

            # 使用混沌生成的分配方案初始化solution
            assert_mapping_eligible(Task, task_assignments, context="cuckoo chaotic init")
            solution.init_definitely(task_assignments)
            solution.heft_scheduling_for_certain_task_mapping(Task)
        population.append(solution)

    return population


def gaussian_random_walk(Task:TaskConfig, population: list[MappingSolution], fitness: list, current_best: MappingSolution,
                         iteration: int):
    new_population = []

    # 计算每个个体的排名（升序，适应度越小越好）
    ranks = np.argsort(fitness)  # rank(i) 是索引，越小表示适应度越好

    for i in range(len(population)):
        # 生成随机数 r2 ∈ [0,1]
        r2 = np.random.rand()

        # 判断是否应用高斯随机游走（公式17条件）
        if r2 <= (ranks[i] + 1) / len(ranks):  # rank(i)从0开始，故+1
            # 计算自适应方差 ξ（公式18）
            xi = [
                (current_best.task_mapping[d] - population[i].task_mapping[d]) / (iteration ** (3 / 5) + 1e-10)
                for d in range(Task.task_number)
            ]
            # 高斯随机游走（公式17）
            r3 = np.random.rand()
            new_solution_mapping = [
                current_best.task_mapping[d] + np.random.normal(scale=np.abs(xi[d])) + r3 * (
                            current_best.task_mapping[d] - population[i].task_mapping[d])
                for d in range(Task.task_number)
            ]
            new_solution, _ = build_projected_solution(Task, new_solution_mapping, context="cuckoo gaussian walk")
            new_population.append(new_solution)

    return new_population


def choose_best_individual(original_pop, new_pop, original_fitness, new_fitness, original_delays, new_delays):
    """
    Algorithm 3: 选择更优的个体组成新群体
    :param original_pop: 原群体 (list of MappingSolution)
    :param new_pop: 新群体 (list of MappingSolution)
    :param original_fitness: 原群体适应度
    :param new_fitness: 新群体适应度
    :param original_delays: 原群体延时
    :param new_delays: 新群体延时
    :return: 选择后的新群体, 对应的适应度和延时
    """
    # 合并两个群体
    combined_pop = original_pop + new_pop
    combined_fitness = np.concatenate([original_fitness, new_fitness])
    combined_delays = np.concatenate([original_delays, new_delays])

    # 计算转换后的适应度 (AFT)
    transformed_fitness = transformed_fitness_function(
        combined_fitness,
        combined_delays,
        combined_fitness  # 这里传入原始适应度，函数内部会处理
    )

    # 按转换后的适应度排序
    sorted_indices = np.argsort(transformed_fitness)

    # 选择前N_p个最优个体
    N_p = len(original_pop)
    best_indices = sorted_indices[:N_p]

    # 构建最终群体
    final_pop = [combined_pop[i] for i in best_indices]
    final_fitness = combined_fitness[best_indices]
    final_delays = combined_delays[best_indices]

    return final_pop, final_fitness, final_delays

def get_cuckoos(Task:TaskConfig, nest:list[MappingSolution], best:MappingSolution):
    """
    通过Levy飞行生成新解
    """
    n = len(nest)
    beta = 3 / 2
    # 正确计算sigma的公式
    numerator = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    denominator = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (numerator / denominator) ** (1 / beta)

    new_nest = []
    for j in range(n):
        s = nest[j].task_mapping.copy()
        # Mantegna算法实现Levy飞行
        u = np.random.randn(len(s)) * sigma
        v = np.random.randn(len(s))
        step = u / np.abs(v) ** (1 / beta)
        # print(step)

        # 步长缩放
        stepsize = 0.01 * step * (np.array(s) - np.array(best.task_mapping.copy()))
        s_new = s + stepsize * np.random.randn(len(s))
        # print(stepsize)
        # s_new = s + step * (np.array(s) - np.array(best.task_mapping.copy()))

        # 应用边界约束
        new_solution, _ = build_projected_solution(Task, s_new, context="cuckoo levy")
        new_nest.append(new_solution)
        # new_nest[j].visualize_init(is_block=True)
        # print(f"solution{j + 1},能耗{new_nest[j].calc_energy()}")

    return new_nest

def empty_nests(Task:TaskConfig, nest:list[MappingSolution], fitness, pa_max=0.7, pa_min=0.2 ):
    """
    发现并替换较差解
    """
    n = len(nest)
    # 随机决定哪些巢穴会被发现
    if max(fitness) - min(fitness) == 0:
        pa = 0.5
    else:
        pa = [(pa_max-pa_min)*((f - min(fitness))/(max(fitness) - min(fitness))) ** 2 + pa_min for f in fitness]
        pa = np.array([[val] * Task.task_number for val in pa])
    K = np.random.rand(n, Task.task_number) < pa
    # print(K)

    # 生成随机步长
    task_mappings = np.array([obj.task_mapping for obj in nest])
    perm1 = np.random.permutation(n)
    perm2 = np.random.permutation(n)
    stepsize = np.random.rand() * (task_mappings[perm1] - task_mappings[perm2])
    # print(stepsize * K)

    # 生成新解
    new_s = task_mappings + stepsize * K
    new_nest = []
    for j in range(n):
        new_solution, _ = build_projected_solution(Task, new_s[j, :], context="cuckoo empty nest")
        new_nest.append(new_solution)
        # new_nest[j].visualize_init(is_block=True)
        # print(f"solution{j + 1},能耗{new_nest[j].calc_energy()}")

    return new_nest

def get_best_nest(Task:TaskConfig, nest:list[MappingSolution], new_nest:list[MappingSolution], fitness,transformed_fitness):
    """
    评估并获取最优解
    """
    updated_fitness = fitness.copy()
    updated_transformed_fitness = transformed_fitness.copy()
    updated_nest = nest.copy()

    energy = [solution.calc_energy(Task) for solution in new_nest]
    time_delay = [solution.calc_delay(Task) for solution in new_nest]
    fnew = fitness_function(energy)
    transformed_fnew = transformed_fitness_function(energy,time_delay,fnew)

    # 输出不可行解
    # print([timeii for timeii in time_delay if timeii > 0])

    # 对于每个个体，若新个体适应度更强，则更有可能选择新个体
    for index in range(len(fnew)):
        if transformed_fnew[index] <= transformed_fitness[index]:
            updated_fitness[index] = fnew[index]
            updated_nest[index] = new_nest[index]
            updated_transformed_fitness[index] = transformed_fnew[index]

    # 找到当前最优
    best_idx = np.argmin(updated_fitness)
    best = updated_nest[best_idx]

    return best, updated_nest, updated_fitness,updated_transformed_fitness

# 计算适应度，越小越好
def fitness_function(energy_list):
    fitness_list = []
    # 部分解可行，对能耗进行归一化（适应度在1以内，超时时间保持求和）
    for index,energy in enumerate(energy_list):
        fitness = (energy - min(energy_list))/(max(energy_list) - min(energy_list)+1e-6)
        fitness_list.append(fitness)
    return fitness_list

def transformed_fitness_function(energy_list, time_delay_list,fitness_list):
    feasible_mask = (time_delay_list == 0)
    phi = np.sum(feasible_mask) / len(energy_list)
    transformed_fitness_list = np.zeros_like(fitness_list)
    if phi == 0:
        transformed_fitness_list = time_delay_list
    elif phi < 1:  # 部分可行（半可行状态，公式27-30）
        # 分可行集 W1 和不可行集 W2
        W1 = feasible_mask
        W2 = ~feasible_mask
        f_min = np.min(fitness_list[W1])
        f_max = np.max(fitness_list[W1])

        # 转换适应度
        transformed_fitness_list[W1] = (fitness_list[W1] - f_min) / (f_max - f_min + 1e-10)  # 归一化
        transformed_fitness_list[W2] = (phi * fitness_list[W2] + (1 - phi) * f_max)  # 加权调整
        transformed_fitness_list[W2] = (transformed_fitness_list[W2] - np.min(transformed_fitness_list)) / \
                                  (np.max(transformed_fitness_list) - np.min(transformed_fitness_list) + 1e-10)
        transformed_fitness_list[W2] += (time_delay_list[W2] - np.min(time_delay_list[W2])) / \
                                   (np.max(time_delay_list[W2]) - np.min(time_delay_list[W2]) + 1e-10)
    else:  # 全部可行（公式31）
        transformed_fitness_list = fitness_list
    return transformed_fitness_list


def simplebounds(s,task_number,processor_number):
    """
    应用边界约束并保证结果为整数
    """
    # 搜索空间边界
    Lb = 0 * np.ones(task_number)
    Ub = (processor_number - 1) * np.ones(task_number)
    s_new = s.copy()
    try:
        # 先四舍五入取整
        s_new = np.round(s_new).astype(int)
        s_new[s_new < Lb] = Lb[s_new < Lb]
        s_new[s_new > Ub] = Ub[s_new > Ub]
    except Exception as e:
        # 打印错误类型
        print(f"Error encountered: {type(e).__name__} - {str(e)}")
        # 如果出现无效值（如NaN或inf），则将其设置为边界值
        s_new = np.nan_to_num(s_new, nan=0, posinf=Ub[0], neginf=Lb[0])
        s_new = np.round(s_new).astype(int)
        s_new[s_new < Lb] = Lb[s_new < Lb]
        s_new[s_new > Ub] = Ub[s_new > Ub]
    return s_new

def cost_time_modification(Task:TaskConfig,solution:MappingSolution):
    priority_list = [0] * Task.task_number
    task_list = list(range(Task.task_number))
    for i in range(len(Task.app_list)):
        for q in range(Task.instance_number_list[i]):
            ou_degree = Task.app_list[i].out_degree.copy()
            nodes = list(range(Task.app_list[i].node_num))

            # 初始化最早开始时间
            priority_base = (Task.instance_number_list[i] - q - 1) * Task.app_list[i].period

            # 找到所有入度为0的节点（起始节点）
            queue = deque([node for node in nodes if ou_degree[node] == 0])
            for node in queue:
                pos = calc_subtask_instance_position(Task, i, q + 1, node)
                avg_exe = average_eligible_execution_time_for_priority(Task.app_list[i], node)
                priority_list[pos] = avg_exe + priority_base

            # 拓扑排序
            while queue:
                u = queue.popleft()
                u_pos = calc_subtask_instance_position(Task, i, q + 1, u)
                u_prio = priority_list[u_pos]
                # 遍历u的所有邻接节点
                for v in Task.app_list[i].reverse_graph[u]:
                    pos = calc_subtask_instance_position(Task, i, q + 1, v)
                    avg_exe = average_eligible_execution_time_for_priority(Task.app_list[i], v)
                    priority_list[pos] = max(priority_list[pos], avg_exe + u_prio)
                    # 减少v的入度
                    ou_degree[v] -= 1
                    # 如果v的入度为0，加入队列
                    if ou_degree[v] == 0:
                        queue.append(v)
    priority_list_sorted, task_list_sorted = sort_lists(priority_list, task_list)
    # 优化排序
    for i3 in range(len(Task.app_list)):
        for q3 in range(Task.instance_number_list[i3]):
            for j3 in range(Task.app_list[i3].node_num):
                pos = calc_subtask_instance_position(Task, i3, q3, j3)
                for processor2 in eligible_processor_ids(Task, pos):
                    task_mapping = solution.task_mapping.copy()
                    task_DPR = solution.task_DPR.copy()  # 任务是否重构
                    task_DPR_start_time = solution.task_DPR_start_time.copy()  # 任务重构开始时间
                    task_DPR_end_time = solution.task_DPR_end_time.copy()  # 任务重构结束时间
                    processor_RP_task_list = solution.processor_RP_task_list.copy()  # 记录重构的任务
                    processor_RP_time_list = solution.processor_RP_time_list.copy()  # 记录每次重构的[开始时间,截止时间]
                    task_start_time = solution.task_start_time.copy()
                    task_end_time = solution.task_end_time.copy()
                    processor_completed_task = solution.processor_completed_task.copy()
                    energy = solution.calc_energy(Task)
                    solution.clear_scheduling(Task.task_number)
                    solution.task_mapping = task_mapping.copy()
                    solution.task_mapping[pos] = processor2
                    for idx, task in enumerate(task_list_sorted):
                        i, q, j = calc_subtask_instance_from_position(Task, task)
                        arr = Task.instance_dict_list[i][q]["arr"]
                        predecessor_end_time = arr
                        predecessor = Task.app_list[i].predecessor_of_node(j)
                        if predecessor:
                            predecessor = [calc_subtask_instance_position(Task,i, q, j2) for j2 in predecessor]
                            predecessor_end_time_list = [solution.task_end_time[pre_task] for pre_task in predecessor]
                            predecessor_end_time = max(predecessor_end_time_list)
                        processor = solution.task_mapping[task]
                        if processor < Task.CPU_number + Task.NPU_number:
                            if processor < Task.CPU_number:
                                exe = Task.app_list[i].sw_exe[j]
                            else:
                                exe = Task.app_list[i].npu_exe[j]
                            solution.task_start_time[task] = max(predecessor_end_time,
                                                             solution.processor_avail[processor])
                        else:
                            DRP = True
                            if solution.processor_completed_task[processor]:
                                task_last = solution.processor_completed_task[processor][-1]
                                i2, q2, j2 = calc_subtask_instance_from_position(Task, task_last)
                                if i2 == i and j2 == j:
                                    DRP = False
                            DRP_end_time = solution.processor_avail[processor]
                            if DRP:
                                DRP_start_time = max(solution.processor_avail[processor],
                                                     solution.processor_RP_config_time)
                                DRP_end_time = DRP_start_time + Task.app_list[i].tpr[j]
                                solution.task_DPR[task] = 1  # 任务是否重构
                                solution.task_DPR_start_time[task] = DRP_start_time  # 任务重构开始时间
                                solution.task_DPR_end_time[task] = DRP_end_time  # 任务重构结束时间
                                solution.processor_RP_config_time = DRP_end_time  # 重构配置接口的空余时间
                                solution.processor_RP_task_list.append(task)  # 记录重构的任务
                                solution.processor_RP_time_list.append(
                                    [DRP_start_time, DRP_end_time])  # 记录每次重构的[开始时间,截止时间]
                            exe = Task.app_list[i].hw_exe[j]
                            solution.task_start_time[task] = max(predecessor_end_time, DRP_end_time)
                        solution.task_end_time[task] = solution.task_start_time[task] + exe
                        solution.processor_completed_task[processor].append(task)
                        solution.processor_avail[processor] = solution.task_end_time[task]  # 处理器空闲时间
                    if solution.calc_delay(Task) > 0 or solution.calc_energy(Task) > energy:
                        solution.task_mapping = task_mapping
                        solution.task_DPR = task_DPR
                        solution.task_DPR_start_time = task_DPR_start_time  # 任务重构开始时间
                        solution.task_DPR_end_time = task_DPR_end_time  # 任务重构结束时间
                        solution.processor_RP_task_list = processor_RP_task_list  # 记录重构的任务
                        solution.processor_RP_time_list = processor_RP_time_list  # 记录每次重构的[开始时间,截止时间]
                        solution.task_start_time = task_start_time
                        solution.task_end_time = task_end_time
                        solution.processor_completed_task = processor_completed_task

if __name__ == "__main__":
    Task=TaskConfig()
    # Task.init_class_vars()
    # start = time.time()
    # best_solution, best_energy = cuckoo_search(Task,n=50, max_iter=1000)
    # print(best_energy)
    # print(best_solution.calc_delay(Task))
    # print(f"time cost: {time.time() - start:.2f}s")
