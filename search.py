from solution_utils import MappingSolution
from task_definition import TaskConfig
import numpy as np
from scipy.special import gamma
import time
import random
import matplotlib.pyplot as plt

def cuckoo_search( N=0, V=0, tot=0, iterr=0,seed=24,n=25, tol=1.0e-5, max_iter=10000):
    """
    布谷鸟搜索算法Python实现

    参数:
    n -- 巢穴数量(默认25)
    pa -- 发现概率(默认0.25)
    task_number -- 问题维度(任务总数)
    tol -- 容差(默认1e-5)
    max_iter -- 最大迭代次数(默认10000)

    返回:
    bestnest -- 最优解
    fmin -- 最优值
    """
    # # 设置随机数种子
    # np.random.seed(seed)
    # random.seed(seed)

    # ========== 记录收敛数据 ==========
    convergence = {
        'best_energy': [],
        'best_delay': [],
        'avg_energy': [],
        'std_energy': [],
        'avg_delay': [],  # 新增平均延时
        'feasible_ratio': []  # 新增可行解比例
    }

    start = time.time()  # 记录开始时间
    # 搜索空间边界
    Lb = 0 * np.ones(TaskConfig.task_number)
    Ub = (TaskConfig.processor_number - 1) * np.ones(TaskConfig.task_number)

    ### 生成随机初始解
    nest = []
    for _ in range(n):
        solution = MappingSolution(TaskConfig.CPU_number,
                                   TaskConfig.NPU_number,
                                   TaskConfig.RP_number,
                                   TaskConfig.task_number)
        solution.init_randomly()
        solution.scheduling()
        nest.append(solution)

    ### 初始评估
    energy = [solution.calc_energy() for solution in nest]
    time_delay = [solution.calc_delay() for solution in nest]
    fitness = fitness_function(energy, time_delay)
    # 先找出所有可行解（delay == 0）的索引
    feasible_indices = [i for i, d in enumerate(time_delay) if d == 0]
    if feasible_indices:  # 如果存在可行解
        # 在可行解中找 fitness 最小的索引
        feasible_fitness = [fitness[i] for i in feasible_indices]
        best_feasible_idx_in_subset = np.argmin(feasible_fitness)
        best_idx = feasible_indices[best_feasible_idx_in_subset]
    else:  # 如果没有可行解，则在整个种群中找 fitness 最小的
        best_idx = np.argmin(fitness)
    bestnest, fmin = nest[best_idx], energy[best_idx]
    # print(f"最小能耗{fmin}")
    # 记录初始状态（增强）
    convergence['best_energy'].append(fmin)
    convergence['best_delay'].append(time_delay[best_idx])
    convergence['avg_energy'].append(np.mean(energy))
    convergence['std_energy'].append(np.std(energy))
    convergence['avg_delay'].append(np.mean(time_delay))  # 平均延时
    convergence['feasible_ratio'].append(np.mean(np.array(time_delay) == 0))  # 可行解比例

    # bestnest.visualize_init(window_name="best scheduler", is_block=True)
    solution_reserve = nest[0].task_mapping
    count = 0
    best = bestnest

    ### 主循环
    N_iter = 0
    while fmin > tol and N_iter < max_iter:
        # 通过Levy飞行生成新解
        # new_nest = get_cuckoos(nest, bestnest, Lb, Ub)
        new_nest = get_cuckoos(nest, best, Lb, Ub)
        best, nest, fitness = get_best_nest(nest, new_nest, fitness)

        # 发现并丢弃较差解
        new_nest = empty_nests(nest, Lb, Ub, fitness)
        best, nest, fitness = get_best_nest(nest, new_nest, fitness)
        fnew = best.calc_energy()
        N_iter += 1

        # 计算当前种群指标
        current_delay = best.calc_delay()
        pop_energy = [sol.calc_energy() for sol in nest]
        pop_delays = [sol.calc_delay() for sol in nest]

        # 更新全局最优
        if fnew < fmin and best.calc_delay() == 0:
            fmin = fnew
            bestnest = best

        # 记录当前迭代数据
        convergence['best_energy'].append(fmin)
        convergence['best_delay'].append(current_delay)
        convergence['avg_energy'].append(np.mean(pop_energy))
        convergence['std_energy'].append(np.std(pop_energy))
        convergence['avg_delay'].append(np.mean(pop_delays))  # 新增
        convergence['feasible_ratio'].append(np.mean(np.array(pop_delays) == 0))  # 新增

        # for solution in nest:
        #     print(solution.task_mapping)
        # print(nest[0].task_mapping)
        if nest[0].task_mapping != solution_reserve:
            count += 1
            solution_reserve = nest[0].task_mapping
        # print(f"最小能耗{fmin}")
        # bestnest.visualize_init(window_name="best scheduler", is_block=True)


    # print(f"Total number of iterations: {N_iter}")
    # print(f"Minimum value found: {fmin}")
    # print(f"Best solution: {bestnest}")
    end = time.time()  # 记录结束时间
    # print(f"耗时: {end - start:.4f} 秒")  # 保留4位小数
    # print(count)
    bestnest.visualize_init(window_name=f".\\test\\cuckoo scheduler {N}_{V}_{tot} iter{iterr}", is_save=True)

    # 绘制收敛曲线
    # plot_convergence(convergence)

    return bestnest, fmin

def get_cuckoos(nest:list[MappingSolution], best:MappingSolution, Lb, Ub):
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
        s_new = simplebounds(s_new, Lb, Ub)
        new_nest.append(MappingSolution(TaskConfig.CPU_number, TaskConfig.NPU_number, TaskConfig.RP_number, TaskConfig.task_number))
        new_nest[j].init_definitely(s_new.tolist())
        new_nest[j].scheduling()
        # new_nest[j].visualize_init(is_block=True)
        # print(f"solution{j + 1},能耗{new_nest[j].calc_energy()}")

    return new_nest

def empty_nests(nest:list[MappingSolution], Lb, Ub, fitness, pa_max=0.7, pa_min=0.2 ):
    """
    发现并替换较差解
    """
    n = len(nest)
    # 随机决定哪些巢穴会被发现
    if max(fitness) - min(fitness) == 0:
        pa = 0.5
    else:
        pa = [(pa_max-pa_min)*((f - min(fitness))/(max(fitness) - min(fitness))) ** 2 + pa_min for f in fitness]
        pa = np.array([[val] * TaskConfig.task_number for val in pa])
    K = np.random.rand(n, TaskConfig.task_number) < pa
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
        new_s_s = simplebounds(new_s[j, :], Lb, Ub)
        new_nest.append(
            MappingSolution(TaskConfig.CPU_number, TaskConfig.NPU_number, TaskConfig.RP_number, TaskConfig.task_number))
        new_nest[j].init_definitely(new_s_s.tolist())
        new_nest[j].scheduling()
        # new_nest[j].visualize_init(is_block=True)
        # print(f"solution{j + 1},能耗{new_nest[j].calc_energy()}")

    return new_nest

def get_best_nest(nest:list[MappingSolution], new_nest:list[MappingSolution], fitness):
    """
    评估并获取最优解
    """
    updated_fitness = fitness.copy()
    updated_nest = nest.copy()

    energy = [solution.calc_energy() for solution in new_nest]
    time_delay = [solution.calc_delay() for solution in new_nest]
    fnew = fitness_function(energy, time_delay)

    # 输出不可行解
    # print([timeii for timeii in time_delay if timeii > 0])

    # 对于每个个体，若新个体适应度更强，则更有可能选择新个体
    for index in range(len(fnew)):
        x = random.random()
        # print(x)
        if fnew[index] != 0 or fitness[index] != 0:
            if x <= fitness[index]/(fnew[index] + fitness[index]):
                updated_fitness[index] = fnew[index]
                updated_nest[index] = new_nest[index]
        else:
            if x < 0.5:
                updated_fitness[index] = fnew[index]
                updated_nest[index] = new_nest[index]
        # if fnew[index] <= fitness[index]:
        #     updated_fitness[index] = fnew[index]
        #     updated_nest[index] = new_nest[index]

    # 找到当前最优
    best_idx = np.argmin(updated_fitness)
    best = updated_nest[best_idx]

    return best, updated_nest, updated_fitness

# 计算适应度，越小越好
def fitness_function(energy_list, time_delay_list):
    fitness_list = []
    # 部分解可行，对能耗进行归一化（适应度在1以内，超时时间保持求和）
    for index,energy in enumerate(energy_list):
        if time_delay_list[index] == 0:
            fitness = (energy - min(energy_list))/(max(energy_list) - min(energy_list))
        else:
            fitness = 1 + time_delay_list[index]/max(time_delay_list)
        fitness_list.append(fitness)
    return fitness_list

def simplebounds(s, Lb, Ub):
    """
    应用边界约束并保证结果为整数
    """
    s_new = s.copy()
    s_new = np.round(s_new).astype(int)  # 先四舍五入取整
    s_new[s_new < Lb] = Lb[s_new < Lb]
    s_new[s_new > Ub] = Ub[s_new > Ub]
    return s_new


# def plot_convergence(convergence):
#     """绘制增强的收敛曲线（含延时信息）"""
#     plt.figure(figsize=(15, 5))
#
#     # 子图1: 能量相关
#     plt.subplot(1, 3, 1)
#     plt.plot(convergence['best_energy'], 'r-', label='Best Energy')
#     plt.plot(convergence['avg_energy'], 'b--', label='Average Energy')
#     plt.fill_between(range(len(convergence['avg_energy'])),
#                      np.array(convergence['avg_energy']) - np.array(convergence['std_energy']),
#                      np.array(convergence['avg_energy']) + np.array(convergence['std_energy']),
#                      alpha=0.2)
#     plt.xlabel('Iteration')
#     plt.ylabel('Energy')
#     plt.title('Energy Convergence')
#     plt.legend()
#     plt.grid(True)
#
#     # 子图2: 延时相关
#     plt.subplot(1, 3, 2)
#     plt.plot(convergence['best_delay'], 'g-', label='Best Delay')
#     plt.plot(convergence['avg_delay'], 'm--', label='Average Delay')
#     plt.xlabel('Iteration')
#     plt.ylabel('Delay')
#     plt.title('Delay Trends')
#     plt.legend()
#     plt.grid(True)
#
#     # 子图3: 可行解比例
#     plt.subplot(1, 3, 3)
#     plt.plot(np.array(convergence['feasible_ratio']) * 100, 'k-')
#     plt.xlabel('Iteration')
#     plt.ylabel('Feasible Solutions (%)')
#     plt.title('Feasibility Ratio')
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.show(block=True)


def plot_convergence(convergence):
    """绘制增强的收敛曲线（含延时信息）"""
    plt.figure(figsize=(15, 5))

    # 子图1: 能量相关
    plt.subplot(1, 3, 1)
    best_energy = convergence['best_energy']

    # 绘制折线
    line, = plt.plot(best_energy, 'r-', label='Best Energy')
    plt.plot(convergence['avg_energy'], 'b--', label='Average Energy')
    plt.fill_between(range(len(convergence['avg_energy'])),
                     np.array(convergence['avg_energy']) - np.array(convergence['std_energy']),
                     np.array(convergence['avg_energy']) + np.array(convergence['std_energy']),
                     alpha=0.2)

    # 标记变化点
    changes = []
    for i in range(1, len(best_energy)):
        if best_energy[i] != best_energy[i - 1]:
            changes.append(i)

    # 在变化点添加标签
    for idx in changes:
        # plt.scatter(idx, best_energy[idx], color='red', zorder=5)
        plt.text(idx, best_energy[idx],
                 f'{best_energy[idx]:.2f}',
                 ha='center', va='bottom', fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy Convergence')
    plt.legend()
    plt.grid(True)

    # 子图2: 延时相关
    plt.subplot(1, 3, 2)
    plt.plot(convergence['best_delay'], 'g-', label='Best Delay')
    plt.plot(convergence['avg_delay'], 'm--', label='Average Delay')
    plt.xlabel('Iteration')
    plt.ylabel('Delay')
    plt.title('Delay Trends')
    plt.legend()
    plt.grid(True)

    # 子图3: 可行解比例
    plt.subplot(1, 3, 3)
    plt.plot(np.array(convergence['feasible_ratio']) * 100, 'k-')
    plt.xlabel('Iteration')
    plt.ylabel('Feasible Solutions (%)')
    plt.title('Feasibility Ratio')
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=True)
