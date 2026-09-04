from solution_utils import (
    MappingSolution,
    assert_mapping_eligible,
    eligible_processor_ids,
    project_mapping_to_eligible_processors,
)
from task_definition import TaskConfig
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import math
from scipy.special import gamma
from GA import plot_convergence, fitness_function
from cuckoo import chaotic_initialization, simplebounds


def _project_hawk_mapping(Task: TaskConfig, raw_mapping):
    bounded = simplebounds(raw_mapping, Task.task_number, Task.processor_number)
    return project_mapping_to_eligible_processors(Task, bounded)


def _build_hawk_from_mapping(Task: TaskConfig, raw_mapping):
    projected = _project_hawk_mapping(Task, raw_mapping)
    hawk = MappingSolution(Task.CPU_number, Task.NPU_number, Task.RP_number, Task.task_number)
    hawk.init_definitely(projected)
    assert_mapping_eligible(Task, hawk.task_mapping, context="HHO candidate")
    hawk.heft_scheduling_for_certain_task_mapping(Task)
    return hawk, projected

def harris_hawks_optimization(Task:TaskConfig, save_dic="test", msg="", iterr=0, n=50, max_iter=10000, E_inital=1,
                              ecs_intial=False, is_modified=False, iter_later=0.7, window_size=5,
                              rapid_change_threshold=0.1):
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
    # 1.随机初始化
    if is_modified:
        hawks = chaotic_initialization_henon(Task, n, ecs_initial=ecs_intial)
    else:
        hawks = chaotic_initialization(n, Task, ecs_initial=ecs_intial)
    # 初始评估
    energy = [sol.calc_energy(Task) for sol in hawks]
    delay = [sol.calc_delay(Task) for sol in hawks]
    fitness = fitness_function(energy, delay)

    # 先找出所有可行解（delay == 0）的索引
    feasible_indices = [i for i, d in enumerate(delay) if d == 0]
    if feasible_indices:  # 如果存在可行解
        # 在可行解中找 fitness 最小的索引
        feasible_fitness = [fitness[i] for i in feasible_indices]
        best_feasible_idx_in_subset = np.argmin(feasible_fitness)
        best_idx = feasible_indices[best_feasible_idx_in_subset]
    else:  # 如果没有可行解，则在整个种群中找 fitness 最小的
        best_idx = np.argmin(fitness)
    best_hawk = hawks[best_idx]
    best_energy = energy[best_idx]

    # print(f"Best Energy: {best_energy:.4f}")
    # print(delay[best_idx])
    # best_hawk.visualize_init(Task,is_block=True)

    # 记录初始状态
    convergence['best_energy'].append(best_energy)
    convergence['generation_best_energy'].append(best_energy)
    convergence['best_delay'].append(delay[best_idx])
    convergence['avg_energy'].append(np.mean(energy))
    convergence['std_energy'].append(np.std(energy))
    convergence['avg_delay'].append(np.mean(delay))
    convergence['feasible_ratio'].append(np.mean(np.array(delay) == 0))

    # 局部最优检测变量
    no_improvement_count = 0
    last_improvement = 0
    stagnation_threshold = max_iter // 20  # 5%的迭代次数作为停滞阈值

    # 主循环
    for iteration in range(max_iter):
        # 2. 计算猎物逃逸能量
        # 条件1：迭代进入后期阶段（如后30%迭代）
        is_later_stage = (iteration >= iter_later * max_iter)

        # 条件2：收敛曲线处于快速收敛状态
        is_rapid_convergence = False
        if len(convergence['best_energy']) >= window_size:
            # 计算最近window_size次迭代的适应度变化率
            recent_changes = np.abs(np.diff(convergence['best_energy'][-window_size:]))
            avg_change = np.mean(recent_changes)
            is_rapid_convergence = (avg_change > rapid_change_threshold)  # 变化率小于阈值则认为收敛停滞

        if is_modified:
            if is_later_stage and is_rapid_convergence:
                e_escaping = 1
            else:
                # 使用间歇性能量调节器
                e_escaping = intermittent_energy_regulator(iteration, max_iter, E0=E_inital)
        else:
            e_escaping = 2 * E_inital * (1-iteration/max_iter) * (2 * np.random.rand() - 1)

        # 判断是否陷入局部最优
        is_stagnated = (iteration - last_improvement) > stagnation_threshold and calculate_diversity(hawks) < 0.1

        # 3.探索阶段
        if abs(e_escaping) >= 1:
            hawks = exploration_phase(Task, hawks, best_hawk, fitness,is_modified=is_modified,is_stagnated=is_stagnated)
            # 4. 开发阶段 (|E| < 1)
        else:
            hawks = exploitation_phase(Task, hawks, best_hawk, e_escaping, iteration, max_iter, is_modified, is_stagnated=is_stagnated)

        # 重新评估
        energy = [sol.calc_energy(Task) for sol in hawks]
        delay = [sol.calc_delay(Task) for sol in hawks]
        fitness = fitness_function(energy, delay)
        current_best_idx = np.argmin(fitness)

        # 更新全局最优
        if (energy[current_best_idx] < best_energy and delay[current_best_idx] <= best_hawk.calc_delay(Task)) or delay[current_best_idx] < best_hawk.calc_delay(Task):
            best_hawk = hawks[current_best_idx]
            best_energy = energy[current_best_idx]
            last_improvement = iteration
            no_improvement_count = 0
        else:
            # if is_stagnated:
            #     last_improvement = iteration
            #     no_improvement_count = 0
            no_improvement_count += 1

        # 记录数据
        convergence['best_energy'].append(best_energy)
        convergence['generation_best_energy'].append(hawks[current_best_idx].calc_energy(Task))
        convergence['best_delay'].append(best_hawk.calc_delay(Task))
        convergence['avg_energy'].append(np.mean(energy))
        convergence['std_energy'].append(np.std(energy))
        convergence['avg_delay'].append(np.mean(delay))
        convergence['feasible_ratio'].append(np.mean(np.array(delay) == 0))

        # print(f"Iteration {iteration + 1}: Best Energy = {best_energy:.4f}")
    # print(best_hawk.calc_energy(Task))
    # best_hawk.energy_optimization(Task)
    if is_modified and ecs_intial:
        plot_convergence(convergence, plot_number=3,
                         save_name=f".\\{save_dic}\\HHO_ECS_modified scheduler {msg} iter{iterr} convergence history.png")
        best_hawk.visualize_init(Task, window_name=f".\\{save_dic}\\HHO_ECS_modified scheduler {msg} iter{iterr} energy{best_hawk.calc_energy(Task)} energy_percent{best_hawk.calc_energy(Task) / Task.E_max} delay{best_hawk.calc_delay(Task)}", is_save=True)
    elif is_modified:
        plot_convergence(convergence, plot_number=3,
                         save_name=f".\\{save_dic}\\HHO_modified scheduler {msg} iter{iterr} convergence history.png")
        best_hawk.visualize_init(Task, window_name=f".\\{save_dic}\\HHO_modified scheduler {msg} iter{iterr} energy{best_hawk.calc_energy(Task)} energy_percent{best_hawk.calc_energy(Task) / Task.E_max} delay{best_hawk.calc_delay(Task)}", is_save=True)
    elif ecs_intial:
        plot_convergence(convergence, plot_number=3,
                         save_name=f".\\{save_dic}\\HHO_ECS scheduler {msg} iter{iterr} convergence history.png")
        best_hawk.visualize_init(Task, window_name=f".\\{save_dic}\\HHO_ECS scheduler {msg} iter{iterr} energy{best_hawk.calc_energy(Task)} energy_percent{best_hawk.calc_energy(Task) / Task.E_max} delay{best_hawk.calc_delay(Task)}", is_save=True)
    else:
        plot_convergence(convergence, plot_number=3,save_name=f".\\{save_dic}\\HHO scheduler {msg} iter{iterr} convergence history.png")
        best_hawk.visualize_init(Task, window_name=f".\\{save_dic}\\HHO scheduler {msg} iter{iterr} energy{best_hawk.calc_energy(Task)} energy_percent{best_hawk.calc_energy(Task) / Task.E_max} delay{best_hawk.calc_delay(Task)}", is_save=True)
    return best_hawk, best_hawk.calc_energy(Task)


def nelder_mead_simplex(Task: TaskConfig, hawks: list[MappingSolution], fitness:list, N_ratio=0.2,
                        max_iter = 100,  # 最大迭代次数
                        alpha = 1.0,  # 反射系数
                        gamma = 2.0,  # 扩展系数
                        rho = 0.5,  # 收缩系数
                        sigma = 0.5  # 缩小系数
                        ) -> list[MappingSolution]:
    """
    Nelder-Mead Simplex方法优化种群中的个体位置
    :param Task: 任务配置对象
    :param hawks: 当前种群（Harris Hawks个体列表）
    :return: 优化后的种群
    """
    # 1. 初始化单纯形顶点（选择种群中适应度最好的n+1个个体）
    N = int(len(hawks) * N_ratio)

    # 论文描述：选择最优、次优、次最差和最差点（见原文Section 3.3步骤1）
    sorted_indices = np.argsort(fitness)

    # 获取关键点索引（注意：原文描述为X_NM-Optimal, X_NM-Suboptimal, X_NM-Subworst, X_NM-Worst）
    idx_optimal = sorted_indices[0]  # X_NM-Optimal
    idx_suboptimal = sorted_indices[1]  # X_NM-Suboptimal
    idx_subworst = sorted_indices[-2]  # X_NM-Subworst（原文明确提及）
    idx_worst = sorted_indices[-1]  # X_NM-Worst（必须包含）

    # 构建初始单纯形顶点（按原文要求包含上述点，其余随机填充至N个）
    elite_indices = [idx_optimal, idx_suboptimal, idx_subworst, idx_worst]
    remaining_indices = [i for i in range(len(hawks)) if i not in elite_indices]
    selected_indices = elite_indices + remaining_indices[:N - len(elite_indices)]

    X_NM = [np.asarray(hawks[i].task_mapping, dtype=float) for i in selected_indices]
    X_NM_hawks= [hawks[i] for i in selected_indices]

    for _ in range(max_iter):
        energy = [sol.calc_energy(Task) for sol in X_NM_hawks]
        delay = [sol.calc_delay(Task) for sol in X_NM_hawks]
        simplex_fitness = fitness_function(energy, delay)
        idx_sorted = np.argsort(simplex_fitness)
        X_NM = [X_NM[i] for i in idx_sorted]
        X_NM_hawks = [X_NM_hawks[i] for i in idx_sorted]

        # 2.2 计算重心（排除最差点）
        Xm = np.mean(X_NM[:-1], axis=0)

        # 2.3 反射操作
        Xr = Xm + alpha * (Xm - X_NM[-1])
        Xr_hawk, Xr = _build_hawk_from_mapping(Task, Xr)
        Xr = np.asarray(Xr, dtype=float)
        X_NM_hawks.append(Xr_hawk)
        energy = [sol.calc_energy(Task) for sol in X_NM_hawks]
        delay = [sol.calc_delay(Task) for sol in X_NM_hawks]
        simplex_fitness = fitness_function(energy, delay)
        fr = simplex_fitness.pop()
        X_NM_hawks.pop()

        if simplex_fitness[0] <= fr < simplex_fitness[-2]:
            X_NM[-1] = Xr  # 替换最差点
            simplex_fitness[-1] = fr
            continue

        # 2.4 扩展操作（如果反射点优于当前最佳）
        if fr < simplex_fitness[0]:
            Xe = Xm + gamma * (Xr - Xm)
            Xe_hawk, Xe = _build_hawk_from_mapping(Task, Xe)
            Xe = np.asarray(Xe, dtype=float)
            X_NM_hawks.append(Xe_hawk)
            X_NM_hawks.append(Xr_hawk)
            energy = [sol.calc_energy(Task) for sol in X_NM_hawks]
            delay = [sol.calc_delay(Task) for sol in X_NM_hawks]
            simplex_fitness = fitness_function(energy, delay)
            fr = simplex_fitness.pop()
            fe = simplex_fitness.pop()
            del X_NM_hawks[-2:]

            if fe < fr:
                X_NM[-1] = Xe
                simplex_fitness[-1] = fe
            else:
                X_NM[-1] = Xr
                simplex_fitness[-1] = fr
            continue

        # 2.5 收缩操作
        if fr < simplex_fitness[-1]:
            # 外收缩
            Xc = Xm + rho * (Xr - Xm)
        else:
            # 内收缩
            Xc = Xm + rho * (X_NM[-1] - Xm)

        Xc_hawk, Xc = _build_hawk_from_mapping(Task, Xc)
        Xc = np.asarray(Xc, dtype=float)
        X_NM_hawks.append(Xc_hawk)
        energy = [sol.calc_energy(Task) for sol in X_NM_hawks]
        delay = [sol.calc_delay(Task) for sol in X_NM_hawks]
        simplex_fitness = fitness_function(energy, delay)
        fc = simplex_fitness.pop()
        X_NM_hawks.pop()

        if fc < simplex_fitness[-1]:
            X_NM[-1] = Xc
            simplex_fitness[-1] = fc
            continue

        # 2.6 缩小操作（向最佳点收缩）
        for i in range(1, N):
            X_NM[i] = X_NM[0] + sigma * (X_NM[i] - X_NM[0])
            X_NM_hawks[i], X_NM[i] = _build_hawk_from_mapping(Task, X_NM[i])
            X_NM[i] = np.asarray(X_NM[i], dtype=float)

    # 3. 用NM优化后的顶点替换原种群中对应个体
    optimized_hawks = []
    for i in range(N):
        new_hawk, _ = _build_hawk_from_mapping(Task, X_NM[i])
        optimized_hawks.append(new_hawk)

    # 保留未参与NM的个体
    other_hawks = [hawks[i] for i in range(len(hawks)) if i not in selected_indices]
    final_hawks = optimized_hawks + other_hawks
    return final_hawks


def exploration_phase(Task: TaskConfig, hawks: list[MappingSolution], best_hawk: MappingSolution, fitness:list, is_modified=False, is_stagnated=False):
    if is_modified and is_stagnated:
        new_hawks = nelder_mead_simplex(Task,hawks,fitness)
    else:
        """探索阶段：全局随机搜索"""
        new_hawks = []
        for i in range(len(hawks)):
            if np.random.rand() < 0.5:
                # 策略1：基于随机个体更新
                rand_hawk = random.choice(hawks)
                r1, r2 = np.random.rand(), np.random.rand()
                new_mapping = [
                    rand_hawk.task_mapping[d] - r1 * abs(rand_hawk.task_mapping[d] - 2 * r2 * hawks[i].task_mapping[d])
                    for d in range(Task.task_number)
                ]
            else:
                # 策略2：基于最优解和平均位置更新
                avg_mapping = np.mean([h.task_mapping for h in hawks], axis=0)
                r3, r4 = np.random.rand(), np.random.rand()
                new_mapping = [
                    (best_hawk.task_mapping[d] - avg_mapping[d]) - r3 * (0 + r4 * (Task.processor_number - 1 - 0))
                    for d in range(Task.task_number)
                ]

            # 边界处理并生成新解
            new_hawk, _ = _build_hawk_from_mapping(Task, new_mapping)
            new_hawks.append(new_hawk)

    return new_hawks

def cardano_beta(t: int, T: int) -> float:
    """根据Cardano公式计算动态beta（公式22）"""
    term = t / T
    cubic_root = np.cbrt(term/27 + np.sqrt(term**2/4 + term**3/27))
    return 1.5 * (1 - cubic_root)

def get_sigma(beta: float) -> float:
    """计算Levy飞行的sigma（公式11）"""
    numerator = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    denominator = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    if beta > 1.2:
        return (numerator / denominator) ** (1 / beta)
    else:
        return (numerator / denominator) ** (1 / beta) * np.random.rand()

def levy_flight(dim: int, beta: float, sigma: float) -> np.ndarray:
    """生成Levy飞行步长（使用预计算的beta和sigma）"""
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / (np.abs(v) ** (1 / beta))
    return 0.01 * step

def compare_solutions(new_sol: MappingSolution, current_sol: MappingSolution, Task: TaskConfig) -> MappingSolution:
    """比较解优劣（返回更优的解）"""
    new_delay, current_delay = new_sol.calc_delay(Task), current_sol.calc_delay(Task)
    new_energy, current_energy = new_sol.calc_energy(Task), current_sol.calc_energy(Task)

    # 优先比较可行性
    if new_delay == 0 and current_delay > 0:
        return new_sol
    elif new_delay > 0 and current_delay == 0:
        return current_sol
    # 都可行时比较能耗
    elif new_delay == 0 and current_delay == 0:
        return new_sol if new_energy < current_energy else current_sol
    # 都不可行时比较延迟
    else:
        return new_sol if new_delay < current_delay else current_sol

def decimal_to_binary(decimal, precision=1000, bits=10):
    """将十进制数转换为10位二进制字符串"""
    integer = int(decimal * precision)
    binary_str = bin(integer)[2:].zfill(bits)
    return binary_str

def binary_to_decimal(binary_str, precision=1000):
    """将二进制字符串转换回十进制数"""
    integer = int(binary_str, 2)
    return integer / precision

def quantum_correction(Task:TaskConfig, hawks:list[MappingSolution], max_rotations=3, theta=0.2)->list[MappingSolution]:
    Lb = 0 * np.ones(Task.task_number)
    Ub = (Task.processor_number - 1) * np.ones(Task.task_number)
    # 1. 初始化量子比特表示
    """将种群位置转换为量子比特表示"""
    N = len(hawks)
    D = Task.task_number
    # 获取所有映射位置并归一化到[0,1]范围
    positions = np.array([h.task_mapping for h in hawks])
    P = (positions - Lb) / (Ub - Lb)
    # 转换为二进制矩阵 (N x D x 10)
    BP = np.zeros((N, D, 10), dtype=int)
    for i in range(N):
        for j in range(D):
            binary_str = decimal_to_binary(P[i, j])
            BP[i, j] = np.array([int(b) for b in binary_str])
    # 转换为量子比特 (N x D x 2 x 10)
    QB = np.zeros((N, D, 2, 10))
    r = np.random.rand(N, D, 10)  # 随机数用于初始化概率幅
    for i in range(N):
        for j in range(D):
            for k in range(10):
                if BP[i, j, k] == 1:
                    alpha = np.sqrt(r[i, j, k])
                    beta = np.sqrt(1 - r[i, j, k])
                else:
                    beta = np.sqrt(r[i, j, k])
                    alpha = np.sqrt(1 - r[i, j, k])
                QB[i, j, 0, k] = alpha  # α概率幅
                QB[i, j, 1, k] = beta  # β概率幅
    # 2. 进行多次量子旋转和测量
    best_hawks = hawks.copy()
    for _ in range(max_rotations):
        # 2.1. 应用量子旋转门
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        for i in range(N):
            for j in range(D):
                for k in range(10):
                    state = QB[i, j, :, k]
                    new_state = np.dot(rotation_matrix, state)
                    QB[i, j, 0, k] = new_state[0]
                    QB[i, j, 1, k] = new_state[1]

        # 2.2. 测量量子比特
        BP = np.zeros((N, D, 10), dtype=int)
        r = np.random.rand(N, D, 10)  # 随机数用于测量

        for i in range(N):
            for j in range(D):
                for k in range(10):
                    alpha = QB[i, j, 0, k]
                    if r[i, j, k] <= alpha ** 2:
                        BP[i, j, k] = 0
                    else:
                        BP[i, j, k] = 1

        # 2.3. 转换回实际位置
        new_positions = np.zeros((N, D))
        for i in range(N):
            for j in range(D):
                binary_str = ''.join([str(b) for b in BP[i, j]])
                decimal = binary_to_decimal(binary_str)
                # 将比例转换回实际位置
                new_positions[i, j] = Lb[j] + decimal * (Ub[j] - Lb[j])


        # 2.4. 创建新种群
        new_hawks = []
        for i in range(len(hawks)):
            new_mapping = new_positions[i].tolist()
            # 边界处理
            new_hawk, _ = _build_hawk_from_mapping(Task, new_mapping)
            new_hawks.append(new_hawk)

        # 2.5. 评估新种群
        combined_hawks = best_hawks + new_hawks
        energy = [h.calc_energy(Task) for h in combined_hawks]
        delay = [h.calc_delay(Task) for h in combined_hawks]
        fitness = fitness_function(energy,delay)
        best_idx = np.argmin(fitness)
        if best_idx >= len(best_hawks):
            best_hawks = new_hawks

    return best_hawks


def quantum_correction_individual(Task: TaskConfig, hawk: MappingSolution, max_rotations=3,
                                  theta=0.2):
    """
    针对单个MappingSolution进行量子校正
    :param Task: 任务配置
    :param hawk: 待校正的个体
    :param max_rotations: 最大旋转次数
    :param theta: 旋转角度
    :return: 校正后的新个体
    """
    Lb = 0 * np.ones(Task.task_number)
    Ub = (Task.processor_number - 1) * np.ones(Task.task_number)

    # 1. 将个体位置转换为量子比特表示
    position = np.array(hawk.task_mapping)
    P = (position - Lb) / (Ub - Lb)

    # 转换为二进制数组 (D x 10)
    BP = np.zeros((Task.task_number, 10), dtype=int)
    for j in range(Task.task_number):
        binary_str = decimal_to_binary(P[j])
        BP[j] = np.array([int(b) for b in binary_str])

    # 转换为量子比特 (D x 2 x 10)
    QB = np.zeros((Task.task_number, 2, 10))
    r = np.random.rand(Task.task_number, 10)  # 随机数用于初始化概率幅

    for j in range(Task.task_number):
        for k in range(10):
            if BP[j, k] == 1:
                alpha = np.sqrt(r[j, k])
                beta = np.sqrt(1 - r[j, k])
            else:
                beta = np.sqrt(r[j, k])
                alpha = np.sqrt(1 - r[j, k])
            QB[j, 0, k] = alpha  # α概率幅
            QB[j, 1, k] = beta  # β概率幅

    # 2. 进行多次量子旋转和测量
    best_position = position.copy()
    best_energy = hawk.calc_energy(Task)
    best_delay = hawk.calc_delay(Task)

    for _ in range(max_rotations):
        # 2.1 应用量子旋转门
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        for j in range(Task.task_number):
            for k in range(10):
                state = QB[j, :, k]
                new_state = np.dot(rotation_matrix, state)
                QB[j, 0, k] = new_state[0]
                QB[j, 1, k] = new_state[1]

        # 2.2 测量量子比特
        BP = np.zeros((Task.task_number, 10), dtype=int)
        r = np.random.rand(Task.task_number, 10)  # 随机数用于测量

        for j in range(Task.task_number):
            for k in range(10):
                alpha = QB[j, 0, k]
                BP[j, k] = 0 if r[j, k] <= alpha ** 2 else 1

        # 2.3 转换回实际位置
        new_position = np.zeros(Task.task_number)
        for j in range(Task.task_number):
            binary_str = ''.join([str(b) for b in BP[j]])
            decimal = binary_to_decimal(binary_str)
            new_position[j] = Lb[j] + decimal * (Ub[j] - Lb[j])

        # 边界处理
        new_hawk, new_position = _build_hawk_from_mapping(Task, new_position)
        new_position = np.array(new_position)

        # 2.4 创建新个体并评估

        # 比较新旧个体的适应度
        new_energy = new_hawk.calc_energy(Task)
        new_delay = new_hawk.calc_delay(Task)
        if new_energy < best_energy and new_delay <= best_delay:
            best_position = new_position
            best_energy = new_energy
            best_delay = new_delay

    # 返回最佳个体
    # best_hawk = MappingSolution(
    #     Task.CPU_number,
    #     Task.NPU_number,
    #     Task.RP_number,
    #     Task.task_number
    # )
    # best_hawk.init_definitely(best_position.tolist())
    # best_hawk.heft_scheduling_for_certain_task_mapping(Task)

    return best_position


def calculate_diversity(population):
    """计算种群多样性"""
    positions = np.array([sol.task_mapping for sol in population])
    mean_position = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - mean_position, axis=1)
    return np.std(distances)  # 使用位置距离的标准差作为多样性指标


def exploitation_phase(Task: TaskConfig, hawks: list[MappingSolution], best_hawk: MappingSolution, e_escaping: float,
                       iterr, max_iter, is_modified=False,is_stagnated=False):
    # if is_stagnated and is_modified:
    #     # print(iterr)
    #     new_hawks = quantum_correction(Task,hawks)
    #     return new_hawks

    """开发阶段：四种局部搜索策略（修正版）"""
    new_hawks = []
    # 计算当前迭代的beta和sigma
    if is_modified:
        beta = cardano_beta(iterr, max_iter)
        sigma = get_sigma(beta)
    else:
        beta = 1.5
        sigma = get_sigma(1.5)
    # ss = (1 - iterr / max_iter) ** 2

    for i in range(len(hawks)):
        r = np.random.rand()  # 随机数决定策略选择
        J = 2 * (1 - np.random.rand())  # 跳跃强度
        delta_mapping = [best_hawk.task_mapping[d] - hawks[i].task_mapping[d] for d in range(Task.task_number)]

        # 策略选择
        if r >= 0.5:
            if abs(e_escaping) >= 0.5:
                # 1. 软包围策略
                new_mapping = [
                    delta_mapping[d] - e_escaping * abs(J * best_hawk.task_mapping[d] - hawks[i].task_mapping[d])
                    for d in range(Task.task_number)
                ]
            else:
                # 2. 硬包围策略
                new_mapping = [
                    best_hawk.task_mapping[d] - e_escaping * abs(delta_mapping[d])
                    for d in range(Task.task_number)
                ]
            if is_stagnated and is_modified:
                new_hawk, _ = _build_hawk_from_mapping(Task, new_mapping)
                new_mapping = quantum_correction_individual(Task, new_hawk)
        else:
            if abs(e_escaping) >= 0.5:
                # 3. 渐进式俯冲策略（标准HHO流程）
                # 生成基本策略解S
                S = [
                    best_hawk.task_mapping[d] - e_escaping * abs(J * best_hawk.task_mapping[d] - hawks[i].task_mapping[d])
                    for d in range(Task.task_number)
                ]
                S_hawk, _ = _build_hawk_from_mapping(Task, S)

                # 比较当前解和S
                if compare_solutions(S_hawk, hawks[i], Task) == S_hawk:
                    new_mapping = S  # 直接采用S
                else:
                    # 在S基础上应用Levy飞行
                    levy_step = levy_flight(Task.task_number,beta=beta,sigma=sigma)
                    new_mapping = [S[d] + levy_step[d] * np.random.rand() for d in range(Task.task_number)]
                    # new_mapping = [S[d] + levy_step[d] * ss for d in range(Task.task_number)]
            else:
                # 4. 高强度俯冲策略（同样流程）
                avg_mapping = np.mean([h.task_mapping for h in hawks], axis=0)

                # 生成基本策略解S
                S = [
                    best_hawk.task_mapping[d] - e_escaping * abs(J * best_hawk.task_mapping[d] - avg_mapping[d])
                    for d in range(Task.task_number)
                ]
                S_hawk, _ = _build_hawk_from_mapping(Task, S)

                if compare_solutions(S_hawk, hawks[i], Task) == S_hawk:
                    new_mapping = S
                else:
                    levy_step = levy_flight(Task.task_number,beta=beta,sigma=sigma)
                    new_mapping = [S[d] + levy_step[d] * np.random.rand() for d in range(Task.task_number)]
                    # new_mapping = [S[d] + levy_step[d] * ss for d in range(Task.task_number)]

        # 边界处理并生成新解
        new_hawk, _ = _build_hawk_from_mapping(Task, new_mapping)
        new_hawks.append(new_hawk)

    return new_hawks


def henon_map(size, a=1.4, b=0.3, x0=0.1, y0=0.1):
    """
    生成Hénon混沌序列
    :param size: 需要生成的序列长度
    :return: 归一化后的混沌序列 (范围[0,1])
    """
    x, y = np.zeros(size), np.zeros(size)
    x[0], y[0] = x0, y0
    for i in range(1, size):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]

    # 归一化到[0,1]区间
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_norm


def chaotic_initialization_henon(Task:TaskConfig, pop_size,ecs_initial=False):
    """
    基于Hénon混沌映射的种群初始化
    :param pop_size: 种群大小
    :param Task: 任务配置对象
    :return: 初始化的种群列表
    """
    if ecs_initial:
        pop_size_exact = pop_size-1
    else:
        pop_size_exact = pop_size
    population = []
    chaos_seq = henon_map(pop_size_exact * Task.task_number)  # 生成足够长的混沌序列

    for i in range(pop_size_exact):
        # 为每个个体生成混沌映射的任务分配
        hawk_mapping = []
        for j in range(Task.task_number):
            # 将混沌值映射到处理器编号
            idx = i * Task.task_number + j
            eligible = eligible_processor_ids(Task, j)
            processor = eligible[int(chaos_seq[idx] * len(eligible)) % len(eligible)]
            hawk_mapping.append(processor)

        # 创建解决方案
        hawk = MappingSolution(Task.CPU_number, Task.NPU_number, Task.RP_number, Task.task_number)
        assert_mapping_eligible(Task, hawk_mapping, context="HHO henon init")
        hawk.init_definitely(hawk_mapping)
        hawk.heft_scheduling_for_certain_task_mapping(Task)
        population.append(hawk)
    if ecs_initial:
        hawk = MappingSolution(Task.CPU_number, Task.NPU_number, Task.RP_number, Task.task_number)
        hawk.ecs_scheduling(Task)
        population.append(hawk)

    return population


def plot_task_processor_distribution(population, Task):
    """
    绘制任务-处理器分配的热图
    :param population: 种群列表
    :param Task: 任务配置对象
    """
    # 初始化计数矩阵 (tasks x processors)
    count_matrix = np.zeros((Task.task_number, Task.processor_number))

    # 统计每个任务被分配到每个处理器的次数
    for hawk in population:
        for task_idx, proc_idx in enumerate(hawk.task_mapping):
            count_matrix[task_idx, proc_idx] += 1

    # 绘制热图
    plt.figure(figsize=(10, 6))
    plt.imshow(count_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Assignment Count')
    plt.xlabel('Processor Index')
    plt.ylabel('Task Index')
    plt.title('Task-to-Processor Assignment Distribution')
    plt.show()


from sklearn.decomposition import PCA

def plot_solution_space(population, Task):
    """
    使用PCA降维后可视化解空间分布
    :param population: 种群列表
    :param Task: 任务配置对象
    """
    # 将所有解的任务映射转换为矩阵
    X = np.array([hawk.task_mapping for hawk in population])

    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.title('Solution Space Projection (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def plot_processor_usage(population, Task):
    """
    绘制处理器使用频率直方图
    :param population: 种群列表
    :param Task: 任务配置对象
    """
    # 统计每个处理器的使用次数
    usage = np.zeros(Task.processor_number)
    for hawk in population:
        for proc in hawk.task_mapping:
            usage[proc] += 1

    plt.figure(figsize=(10, 4))
    plt.bar(range(Task.processor_number), usage)
    plt.xlabel('Processor Index')
    plt.ylabel('Usage Count')
    plt.title('Processor Usage Distribution')
    plt.show()


def intermittent_energy_regulator(t, T, k=5,E0=1):
    """
    实现论文中的间歇性能量调节器（公式16-18）

    参数:
        t (int): 当前迭代次数
        T (int): 最大迭代次数
        k (int): 间歇性周期参数（默认为5）
        E_init (float): 初始能量（默认为1.0）

    返回:
        E_time (float): 更新后的逃逸能量
    """
    E_init = 2 * np.random.rand() - 1
    # 公式16：非线性能量衰减项（指数函数）
    E_nl = 2 * np.exp(-np.pi * t / T)

    # 公式17：间歇性参数（余弦扰动）
    Inf = np.cos(2 * k * np.pi * t / T)

    # 公式18：能量更新规则
    if Inf >= 0:
        E_time = E_init * E_nl * Inf * E0  # 能量非线性衰减
    else:
        E_time = 0  # 能量归零（猎物休息阶段）

    return E_time

if __name__ == "__main__":
    # T = 1000
    # E_values = [intermittent_energy_regulator(t, T) for t in range(T)]
    #
    # # 可视化
    # plt.figure(figsize=(10, 5))
    # plt.plot(E_values, label='E_time (Escape Energy)')
    # plt.xlabel('Iteration (t)')
    # plt.ylabel('Energy Value')
    # plt.title('Intermittent Energy Regulator Dynamics (k=5)')
    # plt.axhline(y=1, color='r', linestyle='--', label='|E_time|=1 (Exploration/Exploitation Threshold)')
    # plt.legend()
    # plt.grid()
    # plt.show()

    np.random.seed(24)
    random.seed(24)
    random_dict = {"N": 2, "V": 10, "alpha": 1.0, "max_out": 2, "beta": 1, "Tpr": 100, "Tpr_beta": 20, "wDAG": 400,
                   "gamma_GPU": 2, "gamma_hw": 1.5, "P_sw": 90, "P_npu": 120, "P_npu_beta": 20,
                   "P_static": 24, "P_static_beta": 2, "P_dyn": 60, "P_dyn_beta": 20, "NWR": 3.0, "tot": 100}
    Task = TaskConfig()
    Task.init_class_vars(random_dict, mode="random")

    # 初始化种群并可视化
    # hawks = chaotic_initialization_henon(Task, 50)
    # energy = [sol.calc_energy(Task) for sol in hawks]
    # delay = [sol.calc_delay(Task) for sol in hawks]
    # fitness = fitness_function(energy, delay)
    # hawks = nelder_mead_simplex(Task,hawks,fitness)
    # print(len(hawks))
    # plot_task_processor_distribution(population, Task)
    # plot_solution_space(population, Task)
    # plot_processor_usage(population, Task)

    # 测试量子校正机制
    # hawks = chaotic_initialization_henon(Task, 10, ecs_initial=True)
    # hawks = quantum_correction(Task,hawks)
    # positions = np.array([h.task_mapping for h in hawks])
    # print(positions)

    start = time.time()
    best_solution, best_energy = harris_hawks_optimization(Task,save_dic="save_test",n=50, max_iter=1000, ecs_intial=False, is_modified=True)
    print(best_energy)
    print(best_solution.calc_delay(Task))
    best_solution.energy_optimization(Task)
    print(best_solution.calc_energy(Task))
    print(best_solution.calc_delay(Task))
    print(best_solution.calc_energy(Task)/Task.E_max)
    print(f"time cost: {time.time() - start:.2f}s")
