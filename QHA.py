from solution_utils import (
    MappingSolution,
    assert_mapping_eligible,
    calc_subtask_instance_from_position,
    calc_subtask_instance_position,
    eligible_processor_ids,
    find_valid_swap_pair,
    list_successors,
    priority_list_generate,
)
from task_definition import TaskConfig
from GA import fitness_function, plot_convergence
from eFLS_adapted import AdaptedEFLSMappingSolution
import random
import numpy as np
import copy

# 判断映射与时间最短最佳映射有多少不同
def compare_current_mapping_with_optimal_mapping(Task:TaskConfig, current_mapping:list, optimal_mapping:list):
    different_mapping_idx = []
    for idx, mapping in enumerate(current_mapping):
        if mapping < Task.CPU_number:
            mapping_type = 'CPU'
        elif mapping < Task.CPU_number + Task.NPU_number:
            mapping_type = 'NPU'
        else:
            mapping_type = 'RP'
        if mapping_type != optimal_mapping[idx]:
            different_mapping_idx.append(idx)
    return different_mapping_idx
# 根据需要修改的位置改变映射
def change_mapping_with_optimal_choice(Task:TaskConfig, current_mapping:list, optimal_mapping:list, change_position:list):
    for pos in change_position:
        if optimal_mapping[pos] == 'CPU':
            current_mapping[pos] = random.choice(list(range(Task.CPU_number)))
        elif optimal_mapping[pos] == 'NPU':
            candidates = [
                p for p in eligible_processor_ids(Task, pos)
                if Task.CPU_number <= p < Task.CPU_number + Task.NPU_number
            ]
            if candidates:
                current_mapping[pos] = random.choice(candidates)
        else:
            candidates = [
                p for p in eligible_processor_ids(Task, pos)
                if p >= Task.CPU_number + Task.NPU_number
            ]
            if candidates:
                current_mapping[pos] = random.choice(candidates)
        # print(f"change mapping of {calc_subtask_instance_from_position(Task, pos)} to {current_mapping[pos]}")

class QHAMappingSolution(MappingSolution):
    def __init__(self, CPU_number, NPU_number, RP_number, task_number):
        # 调用父类的初始化方法
        super().__init__(CPU_number, NPU_number, RP_number, task_number)

    def init_scheduling(self, Task:TaskConfig, mode="random", mapping=None):
        if mode == 'random':
            self.init_randomly(Task)
        else:
            self.init_definitely(mapping)
        assert_mapping_eligible(Task, self.task_mapping, context="QHA init_scheduling")
        self.heft_scheduling_for_certain_task_mapping(Task)
    def scheduling_for_certain_completed_sequence(self, Task:TaskConfig, processor_completed_task:list):
        assert_mapping_eligible(Task, self.task_mapping, context="QHA fixed sequence decode")
        priority_list = priority_list_generate(Task)
        # 更新任务入度
        position = 0
        for i in range(len(Task.app_list)):
            for q in range(Task.instance_number_list[i]):
                for j in range(Task.app_list[i].node_num):
                    self.task_into_degree[position] = Task.app_list[i].into_degree[j]
                    position += 1
        self.processor_completed_task = processor_completed_task.copy()
        task_scheduled_num = 0
        CT = [0] * Task.processor_number
        while task_scheduled_num < Task.task_number:
            fig = False
            for processor in range(Task.processor_number):
                # 先检查处理器上所有任务是否调度完毕，任务只有在入度为0时才能被调度
                if CT[processor] < len(self.processor_completed_task[processor]) and \
                    self.task_into_degree[self.processor_completed_task[processor][CT[processor]]] == 0:
                    task = self.processor_completed_task[processor][CT[processor]]
                    i,q,j = calc_subtask_instance_from_position(Task,task)
                    arr = Task.instance_dict_list[i][q]["arr"]
                    predecessor_end_time = arr
                    predecessor = Task.app_list[i].predecessor_of_node(j)
                    if predecessor:
                        predecessor = [calc_subtask_instance_position(Task,i,q,j2) for j2 in predecessor]
                        predecessor_end_time_list = [self.task_end_time[pre_task] for pre_task in predecessor]
                        predecessor_end_time = max(predecessor_end_time_list)
                    if processor < Task.CPU_number + Task.NPU_number:
                        if processor < Task.CPU_number:
                            exe = Task.app_list[i].sw_exe[j]
                        else:
                            exe = Task.app_list[i].npu_exe[j]
                        self.task_start_time[task] = max(predecessor_end_time, self.processor_avail[processor])
                    else:
                        DRP = True
                        if CT[processor] > 0:
                            task_last = self.processor_completed_task[processor][CT[processor]-1]
                            i2, q2, j2 = calc_subtask_instance_from_position(Task,task_last)
                            if i2 == i and j2 == j:
                                DRP = False
                        DRP_end_time = self.processor_avail[processor]
                        if DRP:
                            DRP_start_time = max(self.processor_avail[processor], self.processor_RP_config_time)
                            DRP_end_time = DRP_start_time + Task.app_list[i].tpr[j]
                            self.task_DPR[task] = 1  # 任务是否重构
                            self.task_DPR_start_time[task] = DRP_start_time  # 任务重构开始时间
                            self.task_DPR_end_time[task] = DRP_end_time  # 任务重构结束时间
                            self.processor_RP_config_time = DRP_end_time  # 重构配置接口的空余时间
                            self.processor_RP_task_list.append(task)  # 记录重构的任务
                            self.processor_RP_time_list.append([DRP_start_time, DRP_end_time])  # 记录每次重构的[开始时间,截止时间]
                        exe = Task.app_list[i].hw_exe[j]
                        self.task_start_time[task] = max(predecessor_end_time, DRP_end_time)
                    self.task_end_time[task] = self.task_start_time[task] + exe
                    self.processor_avail[processor] = self.task_end_time[task]  # 处理器空闲时间
                    # 后续任务入度减1
                    scheduling_task_successors = list_successors(Task,task)
                    for successor in scheduling_task_successors:
                        self.task_into_degree[successor] -= 1  # 后续节点入度减1
                    CT[processor] += 1
                    task_scheduled_num += 1
                    self.task_not_scheduled.remove(task)
                    fig = 1
            if fig == 0 and task_scheduled_num < Task.task_number:
                flag = True
                while flag:
                    for task in self.task_not_scheduled:    # 搜索还未调度的任务
                        if self.task_into_degree[task] == 0:    # 任务入度为0
                            processor = self.task_mapping[task]
                            prio1 = priority_list[self.processor_completed_task[processor][CT[processor]]]
                            prio2 = priority_list[task]
                            if prio2 >= prio1:
                                # i1,q1,j1 = calc_subtask_instance_from_position(Task, self.processor_completed_task[processor][CT[processor]])
                                # i2,q2,j2 = calc_subtask_instance_from_position(Task, task)
                                # print(f"change{i1},{q1},{j1}and{i2},{q2},{j2}")
                                pos = self.processor_completed_task[processor].index(task)
                                self.processor_completed_task[processor][CT[processor]], self.processor_completed_task[processor][pos]=\
                                self.processor_completed_task[processor][pos], self.processor_completed_task[processor][CT[processor]]
                                flag = False
                                break

class QuantumHeuristicSpace:
    def __init__(self, num_heuristics: int):
        self.num_heuristics = num_heuristics
        # 初始化Q-bit：alpha和beta均为 1/sqrt(2)
        self.q_bits = np.full((2, num_heuristics), 1/np.sqrt(2))
        self.pf = np.zeros(num_heuristics)  # 启发式表现分数
        self.usage_count = np.zeros(num_heuristics)  # 调用次数

    def observe(self) -> list[int]:
        """观测Q-bit状态，返回选中的启发式索引列表"""
        selected = []
        for i in range(self.num_heuristics):
            if random.random() < self.q_bits[1, i]**2:  # |beta|^2概率选中
                selected.append(i)
        if selected:
            return selected
        k = random.randint(1, self.num_heuristics)
        # 随机抽取 k 个不重复的元素
        return random.sample(list(range(self.num_heuristics)), k)

    def update_q_bits(self, selected: list[int], sigma: float = 0.3, Q=20):
        for i in range(self.num_heuristics):
            self.pf[i] = self.pf[i]/(self.usage_count[i] + 1e-6)
        # print(self.pf)
        allp = np.mean(self.pf[selected]) if selected else 0
        """量子旋转门更新Q-bit状态"""
        for i in range(self.num_heuristics):
            if i in selected:
                # 计算旋转角度Delta_theta（公式17）
                if len(selected) > 1:
                    avg_pf = np.mean([self.pf[j] for j in selected])
                    ran = max(self.pf[selected]) - min(self.pf[selected]) + 1e-6
                    delta_theta = ((self.pf[i] - avg_pf) / (Q * ran)) * np.pi  # 使用Q缩放
                else:
                    y = random.random()  # 随机参数y ∈ (0,1)
                    delta_theta = ((self.pf[i] - y) / Q) * np.pi  # 使用Q缩放
                
                # 限制角度范围（经验值：-0.05π ~ 0.05π）
                # delta_theta = np.clip(delta_theta, -0.05*np.pi, 0.05*np.pi)
                
                # 象限判断与转置处理
                alpha, beta = self.q_bits[:, i]
                if (alpha * beta) < 0:  # 第二或第四象限（需转置）
                    alpha, beta = beta, alpha
                    transposed = True
                else:  # 第一或第三象限
                    transposed = False
                
                # 应用旋转门
                rotation = np.array([
                    [np.cos(delta_theta), -np.sin(delta_theta)],
                    [np.sin(delta_theta),  np.cos(delta_theta)]
                ])
                new_state = rotation @ np.array([alpha, beta])
                
                # 恢复原始顺序
                if transposed:
                    new_state = np.array([new_state[1], new_state[0]])
                
                self.q_bits[:, i] = new_state
        
        # 多样性保护：若整体表现差，随机扰动
        if allp < sigma:
            lambda_ = random.uniform(0.01*np.pi, 0.05*np.pi)  # 小随机旋转角
            for i in range(self.num_heuristics):
                alpha, beta = self.q_bits[:, i]
                if (alpha * beta) < 0:
                    alpha, beta = beta, alpha
                    transposed = True
                else:
                    transposed = False

                # 确定旋转方向
                delta_theta = -lambda_ if i in selected else lambda_
                
                rotation = np.array([
                    [np.cos(delta_theta), -np.sin(delta_theta)],
                    [np.sin(delta_theta),  np.cos(delta_theta)]
                ])
                new_state = rotation @ np.array([alpha, beta])
                if transposed:
                    new_state = np.array([new_state[1], new_state[0]])
                
                self.q_bits[:, i] = new_state
        for i in range(self.num_heuristics):
            self.pf[i] = 0
            self.usage_count[i] = 0
        # print(self.q_bits[1,:])

# 低层启发式实现
# 交换不同处理器上的2个任务，调度顺序依照优先级完成
def llh1(solution:QHAMappingSolution, Task:TaskConfig):
    new_mapping = solution.task_mapping.copy()
    swap_pair = find_valid_swap_pair(Task, new_mapping, max_attempts=100)
    if swap_pair is None:
        return solution
    i, j = swap_pair
    new_mapping[i], new_mapping[j] = new_mapping[j], new_mapping[i]
    # i1,q1,j1 = calc_subtask_instance_from_position(Task, i)
    # i2,q2,j2 = calc_subtask_instance_from_position(Task, j)
    # print(f"change{i1},{q1},{j1}and{i2},{q2},{j2}")
    new_solution = QHAMappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number
    )
    new_solution.init_scheduling(Task=Task,mode='certain',mapping=new_mapping)
    return new_solution
# 一个任务从当前处理器移到另一处理器，调度顺序依照优先级完成
def llh2(solution:QHAMappingSolution, Task:TaskConfig):
    new_mapping = solution.task_mapping.copy()
    mutate_pos = random.randint(0, Task.task_number - 1)
    original_value = new_mapping[mutate_pos]
    candidates = [p for p in eligible_processor_ids(Task, mutate_pos) if p != original_value]
    if not candidates:
        return solution
    new_mapping[mutate_pos] = random.choice(candidates)
    new_solution = QHAMappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number
    )
    new_solution.init_scheduling(Task=Task,mode='certain',mapping=new_mapping)
    return new_solution
# 同一处理器上的2个任务交换调度顺序
def llh3(solution:QHAMappingSolution, Task:TaskConfig):
    flag = True
    new_mapping = solution.task_mapping.copy()
    count=0
    while flag:
        count += 1
        if count > 1000:
            print("repeat")
            break
        processor = random.randint(0, Task.processor_number - 1)
        if len(solution.processor_completed_task[processor]) >= 2:
            flag = False
            processor_completed_task = solution.processor_completed_task.copy()
            i, j = random.sample(range(len(processor_completed_task[processor])), 2)
            processor_completed_task[processor][i], processor_completed_task[processor][j] = \
            processor_completed_task[processor][j], processor_completed_task[processor][i]
            # i1,q1,j1 = calc_subtask_instance_from_position(Task, processor_completed_task[processor][i])
            # i2,q2,j2 = calc_subtask_instance_from_position(Task, processor_completed_task[processor][j])
            # print(f"change{i1},{q1},{j1}and{i2},{q2},{j2}")
            new_solution = QHAMappingSolution(
                Task.CPU_number,
                Task.NPU_number,
                Task.RP_number,
                Task.task_number
            )
            new_solution.task_mapping = new_mapping.copy()
            new_solution.scheduling_for_certain_completed_sequence(Task=Task,processor_completed_task=processor_completed_task)
            return new_solution
    return solution
# 将任务分配到耗时更短的处理器
def llh4(solution:QHAMappingSolution, Task:TaskConfig):
    different_mapping = compare_current_mapping_with_optimal_mapping(Task, solution.task_mapping, Task.mapping_time_min)
    if different_mapping:    
        # 随机决定抽取多少个元素（1 到 len(lst)）
        k = random.randint(1, len(different_mapping))
        # 随机抽取 k 个不重复的元素
        different_mapping_sampled = random.sample(different_mapping, k)
        if not isinstance(different_mapping_sampled, list):
            different_mapping_sampled = list(different_mapping_sampled)
        new_mapping = solution.task_mapping.copy()
        change_mapping_with_optimal_choice(Task=Task, current_mapping=new_mapping, optimal_mapping=Task.mapping_time_min, change_position=different_mapping_sampled)
        new_solution = QHAMappingSolution(
            Task.CPU_number,
            Task.NPU_number,
            Task.RP_number,
            Task.task_number
        )
        new_solution.init_scheduling(Task=Task,mode='certain',mapping=new_mapping)
        return new_solution
    return solution
# 将任务分配到能耗更少的处理器
def llh5(solution:QHAMappingSolution, Task:TaskConfig):
    different_mapping = compare_current_mapping_with_optimal_mapping(Task, solution.task_mapping, Task.mapping_energy_min)
    if different_mapping: 
        # 随机决定抽取多少个元素（1 到 len(lst)）
        k = random.randint(1, len(different_mapping))
        # 随机抽取 k 个不重复的元素
        different_mapping_sampled = random.sample(different_mapping, k)
        if not isinstance(different_mapping_sampled, list):
            different_mapping_sampled = list(different_mapping_sampled)
        new_mapping = solution.task_mapping.copy()
        change_mapping_with_optimal_choice(Task=Task, current_mapping=new_mapping, optimal_mapping=Task.mapping_energy_min, change_position=different_mapping_sampled)
        new_solution = QHAMappingSolution(
            Task.CPU_number,
            Task.NPU_number,
            Task.RP_number,
            Task.task_number
        )
        new_solution.init_scheduling(Task=Task,mode='certain',mapping=new_mapping)
        return new_solution
    return solution
def apply_heuristic(Task:TaskConfig, solution:QHAMappingSolution, idx:int):
    if idx == 0:
        return llh1(solution=solution,Task=Task)
    elif idx==1:
        return llh2(solution=solution,Task=Task)
    elif idx==2:
        return llh3(solution=solution,Task=Task)
    elif idx==3:
        return llh4(solution=solution,Task=Task)
    elif idx==4:
        return llh5(solution=solution, Task=Task)

def quantum_chaotic_initialization(population_size, Task:TaskConfig, max_chaos_iter=100, ecs_initial=True)->list[QHAMappingSolution]:
    """
    混沌初始化种群（适配MappingSolution类）
    :param population_size: 种群大小 (n)
    :param max_chaos_iter: 最大混沌迭代次数
    :return: 初始化后的种群列表（MappingSolution对象列表）
    """
    # 初始化种群
    population = []

    for i in range(population_size):
        # 创建新的解决方案对象
        solution = QHAMappingSolution(
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

                eligible = eligible_processor_ids(Task, j)
                resource_idx = min(int(np.floor(c * len(eligible))), len(eligible) - 1)
                task_assignments.append(eligible[resource_idx])

            # 使用混沌生成的分配方案初始化solution
            solution.init_scheduling(Task, mode="certain", mapping=task_assignments)
        population.append(solution)

    return population


def judge(processor_completed_task:int, task_start_time):
    def is_ascending(lst):
        for i in range(len(lst) - 1):
            if lst[i] > lst[i + 1]:
                return False
        return True
    for completed_task in processor_completed_task:
        if completed_task:
            start_time = [task_start_time[i] for i in completed_task]
            if not is_ascending(start_time):
                return False
    return True

def build_qha_heft_seed(Task:TaskConfig)->QHAMappingSolution:
    heft_solution = MappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number
    )
    heft_solution.heft_scheduling(Task)

    qha_solution = QHAMappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number
    )
    qha_solution.init_definitely(heft_solution.task_mapping.copy())
    qha_solution.scheduling_for_certain_completed_sequence(
        Task,
        [list(tasks) for tasks in heft_solution.processor_completed_task]
    )
    return qha_solution

def build_qha_efls_seed(Task:TaskConfig)->QHAMappingSolution:
    efls_solution = AdaptedEFLSMappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number,
        verbose=False
    )
    efls_solution.efls_scheduling(Task)

    qha_solution = QHAMappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number
    )
    qha_solution.init_definitely(efls_solution.task_mapping.copy())
    qha_solution.scheduling_for_certain_completed_sequence(
        Task,
        [list(tasks) for tasks in efls_solution.processor_completed_task]
    )
    return qha_solution

def quantum_inspired_hyper_heuristics_algorithm(Task:TaskConfig,save_dic="save_test", msg="", iterr=0,
                                                pop_size=50, max_iter=1000, ecs_initial=False,phase_iter=10,
                                                elitism_ratio=0.1, num_heuristics=5):
    # 记录收敛数据
    convergence = {
        'best_energy': [],
        'generation_best_energy':[],
        'best_delay': [],
        'avg_energy': [],
        'std_energy': [],
        'avg_delay': [],
        'feasible_ratio': [],
    }
    
    # 初始化解
    population = quantum_chaotic_initialization(population_size=pop_size,Task=Task,ecs_initial=ecs_initial)
    # if population:
    #     population[0] = build_qha_heft_seed(Task)
    # if len(population) > 1:
    #     population[1] = build_qha_efls_seed(Task)
    # 初始化量子空间
    quantum_space = QuantumHeuristicSpace(num_heuristics=num_heuristics)

    energy = [sol.calc_energy(Task=Task) for sol in population]
    delay = [sol.calc_delay(Task=Task) for sol in population]
    fitness = fitness_function(energy, delay)
    best_idx = np.argmin(fitness)
    best_solution = copy.deepcopy(population[best_idx])
    best_energy = energy[best_idx]

    # print(f"Best Energy: {best_energy:.4f}")
    # 记录初始状态
    convergence['best_energy'].append(best_energy)
    convergence['generation_best_energy'].append(best_energy)
    convergence['best_delay'].append(delay[best_idx])
    convergence['avg_energy'].append(np.mean(energy))
    convergence['std_energy'].append(np.std(energy))
    convergence['avg_delay'].append(np.mean(delay))
    convergence['feasible_ratio'].append(np.mean(np.array(delay) == 0))

    selected = quantum_space.observe()
    for iteration in range(max_iter):
        # 每10次迭代为一阶段
        if iteration % phase_iter == 0 and iteration!=0:
            quantum_space.update_q_bits(selected)
            # 更新量子启发式空间
            selected = quantum_space.observe()
        # print(f"iteration{iteration}:")
        # print(f"selected_method:{selected}")
        offspring = []
        # 父代精英保留
        elite_size = int(elitism_ratio * pop_size)
        elite_indices = np.argsort(fitness)[:elite_size]
        elite = [copy.deepcopy(population[i]) for i in elite_indices]  # 深拷贝精英个体
        
        # 选择启发式并应用
        prob = [quantum_space.q_bits[1, i]**2 for i in selected]
        prob /= np.sum(prob)  # 归一化
        # print(prob)
        heuristic_idx = np.random.choice(selected, size=pop_size, p=prob)
        # print(f"solution chosen method:{heuristic_idx}")
        # 记录使用的启发式及其次数
        for idx in heuristic_idx:
            quantum_space.usage_count[idx] += 1
        for idx, sol in enumerate(population):
            new_ind = apply_heuristic(Task=Task,solution=sol, idx=heuristic_idx[idx])
            offspring.append(new_ind)
            # if not judge(new_ind.processor_completed_task, new_ind.task_start_time):
                # print(f"new individual wrong, {idx},{new_ind.processor_completed_task}")
                # new_ind.visualize_init(Task, is_block=True)

        offspring = elite + offspring[:pop_size - elite_size]
        population = offspring
        # for idx, sol in enumerate(population):
        #     if not judge(sol.processor_completed_task, sol.task_start_time):
        #         print(f"elite wrong, {idx},{sol.processor_completed_task}")
        #         sol.visualize_init(Task, is_block=True)
        
        # 评估解并更新启发式表现
        old_energy = energy.copy()
        old_delay = delay.copy()
        energy = [sol.calc_energy(Task=Task) for sol in population]
        delay = [sol.calc_delay(Task=Task) for sol in population]
        for idx, sol in enumerate(population):
            if delay[idx] < old_delay[idx]:
                quantum_space.pf[heuristic_idx[idx]] += 1
            elif delay[idx] == old_delay[idx] and energy[idx] < old_energy[idx]:
                quantum_space.pf[heuristic_idx[idx]] += 1
        # for idx in selected:
        #     print(f"method {idx}: chosen {quantum_space.usage_count[idx]}, improved {quantum_space.pf[idx]}")
        
        # 重新评估
        fitness = fitness_function(energy, delay)
        current_best_idx = np.argmin(fitness)
        # print(delay)
        # print(current_best_idx)

        # 更新全局最优
        if (population[current_best_idx].calc_energy(Task) <= best_energy and population[current_best_idx].calc_delay(Task) == best_solution.calc_delay(Task))\
                or population[current_best_idx].calc_delay(Task) < best_solution.calc_delay(Task):
            best_solution = copy.deepcopy(population[current_best_idx])
            best_energy = energy[current_best_idx]
        # print(f"Iteration {iteration}: Best Energy = {best_energy:.4f}, {best_solution.calc_energy(Task)}")

        # 记录数据
        convergence['best_energy'].append(best_energy)
        convergence['generation_best_energy'].append(population[current_best_idx].calc_energy(Task))
        convergence['best_delay'].append(best_solution.calc_delay(Task))
        convergence['avg_energy'].append(np.mean(energy))
        convergence['std_energy'].append(np.std(energy))
        convergence['avg_delay'].append(np.mean(delay))
        convergence['feasible_ratio'].append(np.mean(np.array(delay) == 0))
    # print(best_energy)
    # print(f"QHA_ECS scheduler {msg} iter{iterr} energy{best_solution.calc_energy(Task)}")
    if best_solution.calc_delay(Task) == 0:
        best_solution.energy_optimization(Task=Task)
    if ecs_initial:
        best_solution.visualize_init(Task, window_name=f".\\{save_dic}\\QHA_ECS scheduler {msg} iter{iterr} energy{best_solution.calc_energy(Task)} energy_percent{best_solution.calc_energy(Task) / Task.E_max} delay{best_solution.calc_delay(Task)}", is_save=True)
        # 绘制收敛曲线
        plot_convergence(convergence,plot_number=3,save_name=f".\\{save_dic}\\QHA_ECS scheduler {msg} iter{iterr} convergence history.png")
    else:
        best_solution.visualize_init(Task, window_name=f".\\{save_dic}\\QHA scheduler {msg} iter{iterr} energy{best_solution.calc_energy(Task)} energy_percent{best_solution.calc_energy(Task) / Task.E_max} delay{best_solution.calc_delay(Task)}", is_save=True)
        # 绘制收敛曲线
        plot_convergence(convergence,plot_number=3,save_name=f".\\{save_dic}\\QHA scheduler {msg} iter{iterr} convergence history.png")
    return best_solution, best_solution.calc_energy(Task)


if __name__ == "__main__":
    np.random.seed(24)
    random.seed(24)
    random_dict = {"N": 2, "V": 5, "alpha": 1.0, "max_out": 2, "beta": 1, "Tpr": 100, "Tpr_beta": 20, "wDAG": 400,
                   "gamma_GPU": 2, "gamma_hw": 1.5, "P_sw": 90, "P_npu": 120, "P_npu_beta": 20,
                   "P_static": 24, "P_static_beta": 2, "P_dyn": 60, "P_dyn_beta": 20, "NWR": 3.0, "tot": 30}
    Task = TaskConfig()
    Task.init_class_vars(random_dict, mode="random")
    print(Task.H)
    print(Task.app_list[0].edges)
    print(Task.app_list[1].edges)

    # solution = QHAMappingSolution(
    #     Task.CPU_number,
    #     Task.NPU_number,
    #     Task.RP_number,
    #     Task.task_number
    # )
    # solution.ecs_scheduling(Task)
    # solution.visualize_init(Task,window_name="1", is_block=False)
    # llh5(solution, Task)
    # solution.visualize_init(Task,window_name="6", is_block=True)
    # llh4(solution, Task)
    # solution.visualize_init(Task,window_name="5", is_block=True)
    # llh3(solution, Task)
    # solution.visualize_init(Task,window_name="4", is_block=True)
    # llh1(solution,Task)
    # solution.visualize_init(Task,window_name="2", is_block=False)
    # llh2(solution,Task)
    # solution.visualize_init(Task,window_name="3", is_block=True)
    best_solution, best_energy = quantum_inspired_hyper_heuristics_algorithm(Task,pop_size=20, max_iter=1000, phase_iter=10, ecs_initial=False)
    print(best_energy/Task.E_max, best_solution.calc_delay(Task))
    best_solution.energy_optimization(Task=Task)
    print(best_solution.calc_delay(Task))
    # print(best_solution.processor_completed_task)
    # print(calc_subtask_instance_from_position(Task,29))
    processor_completed_task = best_solution.processor_completed_task
    # best_solution.clear_scheduling(task_number=Task.task_number)
    # best_solution.scheduling_for_certain_completed_sequence(Task, processor_completed_task)
    best_solution.visualize_init(Task, is_block=True)
    print(best_solution.calc_energy(Task))
    print(best_solution.calc_energy(Task)/Task.E_max)
