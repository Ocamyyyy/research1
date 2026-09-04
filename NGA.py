from solution_utils import MappingSolution
from task_definition import TaskConfig
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import math
from GA import fitness_function
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster  # 改用层次聚类以支持自定义距离
from scipy.spatial.distance import squareform  # 用于距离矩阵转换
from sklearn.cluster import SpectralClustering

processor_types = {}
for ii in range(TaskConfig.CPU_number):
    processor_types[ii] = "CPU"
for ii in range(TaskConfig.CPU_number, TaskConfig.CPU_number + TaskConfig.NPU_number):
    processor_types[ii] = "NPU"
for ii in range(TaskConfig.CPU_number + TaskConfig.NPU_number, TaskConfig.processor_number):
    processor_types[ii] = "FPGA"

# 处理器类型差异矩阵
type_distance_matrix = {
    ("CPU", "CPU"): 0,
    ("CPU", "NPU"): TaskConfig.processor_number * 0.5,
    ("CPU", "FPGA"): TaskConfig.processor_number * 0.5,
    ("NPU", "CPU"): TaskConfig.processor_number * 0.5,
    ("NPU", "NPU"): 0,
    ("NPU", "FPGA"): TaskConfig.processor_number * 0.5,
    ("FPGA", "CPU"): TaskConfig.processor_number * 0.5,
    ("FPGA", "NPU"): TaskConfig.processor_number * 0.5,
    ("FPGA", "FPGA"): 0
}

# 每种处理器的编号范围
processor_ranges = {
    "CPU": (0, TaskConfig.CPU_number - 1),  # min=0, max=3
    "NPU": (TaskConfig.CPU_number, TaskConfig.CPU_number + TaskConfig.NPU_number - 1),  # GPU只有一个编号4
    "FPGA": (TaskConfig.CPU_number + TaskConfig.NPU_number, TaskConfig.processor_number - 1)  # min=5, max=8
}


def cyclic_index_distance(a, b):
    """计算同一类型处理器间的循环编号差距"""
    type_a = processor_types[a]
    type_b = processor_types[b]

    if type_a != type_b:
        return 0  # 不同类型，编号差距不适用

    min_idx, max_idx = processor_ranges[type_a]
    total_indices = max_idx - min_idx + 1  # 该类型的处理器数量

    direct_diff = abs(a - b)
    cyclic_diff = total_indices - direct_diff

    return min(direct_diff, cyclic_diff)


def mixed_distance(a, b, alpha=0.7, beta=0.3):
    """计算两个处理器分配的混合距离（含循环编号差距）"""
    type_a = processor_types[a]
    type_b = processor_types[b]

    # 计算类型差异
    type_dist = type_distance_matrix[(type_a, type_b)]

    # 计算循环编号差距（仅适用于同一类型）
    index_dist = cyclic_index_distance(a, b)

    # 综合距离
    return alpha * type_dist + beta * index_dist


def chromosome_distance(chrom1, chrom2, alpha=1.0, beta=0.5):
    """计算两个染色体的混合距离"""
    assert len(chrom1) == len(chrom2), "Chromosomes must have the same length!"
    total_distance = 0.0
    for a, b in zip(chrom1, chrom2):
        total_distance += mixed_distance(a, b, alpha, beta)
    return total_distance / len(chrom1)


def niche_selection(population: list, fitness: list, pop_size: int, niche_radius: float):
    """
    改进的小生境选择，避免相似解过度繁殖

    参数:
        population: 种群列表，每个元素是一个解（如MappingSolution）
        fitness: 适应度列表，与population一一对应
        pop_size: 需要选择的个体数量
        niche_radius: 小生境半径，用于判定解是否相似

    返回:
        选择后的新种群
    """
    selected_indices = []

    # 按适应度排序（降序）
    ranked_indices = np.argsort(fitness)[::-1]  # 使用传入的fitness而不是self.fitness_values

    for idx in ranked_indices:
        if len(selected_indices) == 0:
            selected_indices.append(idx)
        else:
            # 检查是否与已选个体太相似（基于混合距离）
            too_similar = False
            for sel_idx in selected_indices:
                # 使用传入的population而不是self.pop
                dist = chromosome_distance(population[idx], population[sel_idx])
                if dist < niche_radius:  # 使用传入的niche_radius而不是self.niche_radius
                    too_similar = True
                    break
            if not too_similar:
                selected_indices.append(idx)

        if len(selected_indices) >= pop_size:  # 使用传入的pop_size而不是self.pop_size
            break

    return [population[i] for i in selected_indices]  # 返回选择后的新种群


def niche_genetic_algorithm(seed=24, pop_size=50, pop_number=5, subpop_min_scale=0.5,
                            max_iter=1000, crossover_rate=0.8, mutation_rate=0.1,
                            elitism_ratio=0.1, plot_number=4):
    # 初始化随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 记录收敛数据
    start_time = time.time()
    history = {'best_energy_history':[],
               'best_delay_history':[]}
    sub_best_solution=[]
    for i in range(pop_number):
        history[f'best_energy_history_{i+1}'] = []


    # 1. 初始化种群
    population = []
    for _ in range(pop_size):
        solution = MappingSolution(
            TaskConfig.CPU_number,
            TaskConfig.NPU_number,
            TaskConfig.RP_number,
            TaskConfig.task_number
        )
        solution.init_randomly()
        solution.scheduling()
        population.append(solution)

    # 2. 初始评估并初始化最优解
    energy = [sol.calc_energy() for sol in population]
    delay = [sol.calc_delay() for sol in population]
    fitness = fitness_function(energy, delay)
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx]
    best_energy = energy[best_idx]
    best_delay = delay[best_idx]

    print(f"Initial Best Energy: {best_energy:.4f}")
    best_solution.visualize_init(window_name="initial best solution", is_block=True)

    # 3. 使用自定义距离进行初始聚类
    distance_matrix = np.zeros((pop_size, pop_size))
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dist = chromosome_distance(population[i].task_mapping, population[j].task_mapping)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    k = max(2, pop_number)
    # condensed_dist = squareform(distance_matrix)
    # Z = linkage(condensed_dist, method='average')
    # labels: np.ndarray = fcluster(Z, t=k, criterion='maxclust')
    # print(labels)
    similarity_matrix = np.exp(-distance_matrix / distance_matrix.std())  # 使用高斯核转换
    spectral = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",  # 使用自定义相似度矩阵
        random_state=42
    ).fit(similarity_matrix)
    labels = spectral.labels_  # 标签已经是 0, 1, 2, ..., k-1
    labels = labels + 1
    print(labels)
    label_list = labels.tolist()

    # 4. 划分子种群
    subpopulations = defaultdict(list)
    for idx, label in enumerate(label_list):
        subpopulations[label].append(population[idx])

    subpop_sizes = {label: len(subpop) for label, subpop in subpopulations.items()}
    print(f"initial Subpopulation sizes: {subpop_sizes}")

    # 5.0 预计算所有初代适应度信息
    parent_fitness_info = {}
    for label, subpop in subpopulations.items():
        parent_energy = [sol.calc_energy() for sol in subpop]
        parent_delay = [sol.calc_delay() for sol in subpop]
        parent_fitness = fitness_function(parent_energy, parent_delay)
        parent_fitness_info[label] = (parent_energy, parent_delay, parent_fitness)

    # 5. 进化过程
    for iteration in range(max_iter):
        new_subpops = defaultdict(list)
        count_mutate = 0
        count_crossover = 0

        # 5.1 每个子种群独立进化（变异后立即保留父代精英）
        for label, subpop in subpopulations.items():
            parent_energy, parent_delay, parent_fitness = parent_fitness_info[label]

            # 1. 选择 (轮盘赌选择)
            selected_indices = selection(parent_fitness, len(subpop))
            selected_population = [subpop[i] for i in selected_indices]

            # 2. 交叉 (单点交叉)
            offspring = []
            for i in range(0, len(subpop), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1] if i + 1 < len(subpop) else selected_population[0]

                if random.random() < crossover_rate:
                    count_crossover += 1
                    child1, child2 = crossover(parent1, parent2)
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1, parent2])

            # 3. 变异 (随机翻转)
            for i in range(len(offspring)):
                if random.random() < mutation_rate:
                    count_mutate += 1
                    offspring[i] = mutate(offspring[i],is_random=True)

            # 4. 父代精英保留
            elite_size = math.ceil(elitism_ratio * len(subpop))
            elite_indices = np.argsort(parent_fitness)[:elite_size]
            elite = [subpop[i] for i in elite_indices]
            new_subpops[label] = elite + offspring[:len(subpop) - elite_size]

        # === 新增迁移操作 ===
        if iteration % 10 == 0 and iteration < 300:  # 每10代迁移一次
            new_subpops = migration(new_subpops, migration_rate=0.05)

        # 5.2 动态调整子种群规模
        # 计算全局适应度
        all_energy = [sol.calc_energy() for subpop in new_subpops.values() for sol in subpop]
        all_delay = [sol.calc_delay() for subpop in new_subpops.values() for sol in subpop]
        global_fitness = fitness_function(all_energy, all_delay)

        # 计算子种群平均适应度
        subpop_info = []
        index = 0
        for label, subpop in new_subpops.items():
            subfitness = global_fitness[index:index + len(subpop)]
            avg_fitness = np.mean(subfitness)
            index += len(subpop)
            subpop_info.append((label, len(new_subpops[label]), avg_fitness))

        # 分配规模
        total = pop_size
        min_size = max(2, int(pop_size / k * subpop_min_scale))
        allocations = {info[0]: min_size for info in subpop_info}
        total -= len(subpop_info) * min_size

        # 轮盘赌分配剩余个体
        if total > 0 and subpop_info:
            # # 计算反向适应度作为权重
            # max_fitness = max(info[2] for info in subpop_info)
            # min_fitness = min(info[2] for info in subpop_info)
            #
            # # 避免除零，加一个小的偏移量
            # epsilon = (max_fitness - min_fitness) * 0.01 + 1e-10
            # inverse_fitness = [max_fitness - info[2] + epsilon for info in subpop_info]

            # 获取全局适应度的最大值和最小值（而非子种群平均适应度的极值）
            global_max_fitness = np.max(global_fitness)  # 所有个体中的最大适应度
            global_min_fitness = np.min(global_fitness)  # 所有个体中的最小适应度

            # 计算反向适应度权重（确保适应度越小的子种群权重越大）
            epsilon = (global_max_fitness - global_min_fitness) * 0.01 + 1e-10
            # inverse_fitness = [
            #     (global_max_fitness - info[2] + epsilon)  # info[2] 是子种群平均适应度
            #     for info in subpop_info
            # ]
            inverse_fitness = [
                (global_max_fitness - info[2])/(global_max_fitness - global_min_fitness + epsilon)  # info[2] 是子种群平均适应度
                for info in subpop_info
            ]

            # 计算选择概率
            sum_inverse = sum(inverse_fitness)
            probs = [inv / sum_inverse for inv in inverse_fitness]
            subpop_sizes = {info[0]:info[2] for info in subpop_info}
            print(f"avg fitness: {subpop_sizes}")
            print(inverse_fitness)
            print(probs)

            # 使用轮盘赌选择
            all_labels = [info[0] for info in subpop_info]
            selected_labels = np.random.choice(all_labels, size=total, p=probs, replace=True)

            # 统计每个label被选中的次数
            unique_labels, counts = np.unique(selected_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                allocations[label] += count
        print(allocations)

        # 5.3 重组子种群（再次精英保留+轮盘赌选择）
        final_subpops = defaultdict(list)
        current_population = []

        for label in new_subpops.keys():
            target_size = allocations.get(label, min_size)
            current_pop = new_subpops[label]

            # 计算当前子代适应度
            current_energy = [sol.calc_energy() for sol in current_pop]
            current_delay = [sol.calc_delay() for sol in current_pop]
            current_fitness = fitness_function(current_energy, current_delay)

            # 精英选择（确保不超过当前种群大小）
            elite_size = min(len(current_pop), max(1, int(elitism_ratio * target_size)))
            elite_indices = np.argsort(current_fitness)[:elite_size]
            elites = [current_pop[i] for i in elite_indices]

            # 轮盘赌选择所有个体（包含精英）
            # selected_individuals = []
            # remaining_size = target_size - elite_size

            # if remaining_size > 0 and current_pop:
            #     # 直接使用原始适应度（假设越大越好）
            #     sum_fit = sum(current_fitness)
            #     if sum_fit > 0:
            #         probs = [f / sum_fit for f in current_fitness]
            #         # 允许重复选择（replace=True），精英可能被再次选中
            #         selected = np.random.choice(len(current_pop), size=remaining_size, p=probs, replace=True)
            #         selected_individuals = [current_pop[i] for i in selected]
            #     else:
            #         # 如果所有适应度都为0，随机选择（允许重复）
            #         selected_individuals = [random.choice(current_pop) for _ in range(remaining_size)]
            # 随机选择剩余个体（从整个种群中随机选，可能包含精英）
            remaining_size = target_size - elite_size
            if remaining_size > 0 and current_pop:
                # 从整个种群中随机选择（允许重复）
                selected_individuals = [random.choice(current_pop) for _ in range(remaining_size)]
            else:
                selected_individuals = []

            final_subpops[label] = elites + selected_individuals
            current_population.extend(final_subpops[label])

        subpopulations = final_subpops
        population = current_population

        # 6. 评估并更新全局最优
        energy = [sol.calc_energy() for sol in population]
        delay = [sol.calc_delay() for sol in population]
        fitness = fitness_function(energy, delay)

        # 6.1 更新父代适应度信息用于下一次迭代
        index = 0
        for label, subpop in subpopulations.items():
            # 当前子群体的切片范围
            start, end = index, index + len(subpop)
            parent_energy = energy[start:end]
            parent_delay = delay[start:end]
            parent_fitness = fitness_function(parent_energy, parent_delay)
            parent_fitness_info[label] = (parent_energy, parent_delay, parent_fitness)
            index += len(subpop)

        # 6.2 更新每一个种群的最有能量
        index = 0
        for label, subpop in subpopulations.items():
            # 当前子群体的切片范围
            start, end = index, index + len(subpop)

            # 创建布尔掩码：当前子群体中delay等于0的位置
            energy_aa = [energy[i] for i in range(start,end) if delay[i] == 0]

            if energy_aa:  # 如果存在delay为0的个体
                # 获取对应energy的最小值
                min_energy = np.min(energy_aa)
            else:  # 如果没有delay为0的个体
                min_energy = np.nan  # 用nan表示缺失值

            history[f'best_energy_history_{label}'].append(min_energy)
            index += len(subpop)

        # 6.3 更新全局最优解
        # 优先选择可行解（delay=0）
        feasible_mask = np.array(delay) == 0
        if np.any(feasible_mask):
            feasible_fitness = np.array(fitness)[feasible_mask]
            best_feasible_idx = np.argmin(feasible_fitness)
            current_best_idx = np.where(feasible_mask)[0][best_feasible_idx]
        else:
            current_best_idx = np.argmin(fitness)

        # 更新全局最优
        if energy[current_best_idx] < best_energy:
            best_solution = population[current_best_idx]
            best_energy = energy[current_best_idx]
            best_delay = delay[current_best_idx]

        # 记录历史
        history['best_energy_history'].append(best_energy)
        history['best_delay_history'].append(best_delay)

        # 打印进度
        print(f"Iter {iteration}: Best Energy={best_energy:.2f}, Delay={best_delay}")
        if iteration == max_iter - 1:
            index = 0
            for label, subpop in subpopulations.items():
                # 当前子群体的切片范围
                start, end = index, index + len(subpop)

                # 创建布尔掩码：当前子群体中delay等于0的位置
                energy_aa = [energy[i] for i in range(start, end) if delay[i] == 0]
                subpopaa = [subpop[i] for i in range(start, end) if delay[i] == 0]

                if energy_aa:  # 如果存在delay为0的个体
                    # 获取对应energy的最小值
                    sub_best_solution.append(subpopaa[np.argmin(energy_aa)])
                else:  # 如果没有delay为0的个体
                    sub_best_solution.append(np.nan)  # 用nan表示缺失值

    # 返回结果
    print(f"\nFinal Result: Best Energy={best_energy:.2f}, Delay={best_delay}")
    print(f"Total time: {time.time() - start_time:.2f}s")

    plt.figure()
    plt.plot(history['best_energy_history'])
    plt.title("Best Energy History")
    plt.show(block=True)

    plt.figure()
    for label in subpopulations.keys():
        print(history[f'best_energy_history_{label}'])
        plt.plot(history[f'best_energy_history_{label}'], label=f'Best energy of subpop {label}')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Energy Trends')
        plt.legend()
        plt.grid(True)
    plt.show(block=True)
    for i in range(pop_number):
        sub_best_solution[i].visualize_init(window_name=f"Best Solution{i+1}", is_block=True)

    return best_solution, history['best_energy_history'], history['best_delay_history']


def selection(fitness, pop_size):
    """轮盘赌选择"""
    inverted_fitness = 1 / (np.array(fitness) + 1e-6)  # 避免除零
    prob = inverted_fitness / np.sum(inverted_fitness)
    return np.random.choice(range(len(fitness)), size=pop_size, p=prob)


def crossover(parent1, parent2):
    """单点交叉"""
    crossover_point = random.randint(1, TaskConfig.task_number - 1)
    child1_mapping = parent1.task_mapping[:crossover_point] + parent2.task_mapping[crossover_point:]
    child2_mapping = parent2.task_mapping[:crossover_point] + parent1.task_mapping[crossover_point:]

    child1 = MappingSolution(
        TaskConfig.CPU_number,
        TaskConfig.NPU_number,
        TaskConfig.RP_number,
        TaskConfig.task_number
    )
    child1.init_definitely(child1_mapping)
    child1.scheduling()

    child2 = MappingSolution(
        TaskConfig.CPU_number,
        TaskConfig.NPU_number,
        TaskConfig.RP_number,
        TaskConfig.task_number
    )
    child2.init_definitely(child2_mapping)
    child2.scheduling()

    return child1, child2


def mutate(solution, is_random=False):
    """随机变异一个处理器分配"""
    mutated_mapping = solution.task_mapping.copy()
    if is_random:
        choice = random.choice(["swap", "mutate", "block_move"])
        if choice == "swap":
            # 操作1：交换两个任务的映射
            i, j = random.sample(range(len(mutated_mapping)), 2)
            mutated_mapping[i], mutated_mapping[j] = mutated_mapping[j], mutated_mapping[i]
        elif choice == "mutate":
            # 操作2：随机突变一个任务的映射
            mutate_pos = random.randint(0, TaskConfig.task_number - 1)
            mutated_mapping[mutate_pos] = random.randint(0, TaskConfig.processor_number - 1)
        elif choice == "block_move":
            # 操作3：块迁移（连续多个任务一起迁移）
            block_size = random.randint(1, min(5, len(mutated_mapping) // 2))
            start = random.randint(0, len(mutated_mapping) - block_size)
            new_proc = random.randint(0, TaskConfig.processor_number - 1)
            for i in range(start, start + block_size):
                mutated_mapping[i] = new_proc
    else:
        mutate_pos = random.randint(0, TaskConfig.task_number - 1)
        mutated_mapping[mutate_pos] = random.randint(0, TaskConfig.processor_number - 1)

    new_solution = MappingSolution(
        TaskConfig.CPU_number,
        TaskConfig.NPU_number,
        TaskConfig.RP_number,
        TaskConfig.task_number
    )
    new_solution.init_definitely(mutated_mapping)
    new_solution.scheduling()
    return new_solution


def migration(subpopulations, migration_rate=0.1, elite_ratio=0.2):
    migrants = []
    labels = list(subpopulations.keys())

    # 1. 选择待迁移的个体
    for label in labels:
        subpop = subpopulations[label]
        if len(subpop) < 2:  # 过小的子群不参与迁移
            continue

        # 选择精英移民 (适应度最好的前elite_ratio比例)
        n_elite = max(1, int(elite_ratio * len(subpop) * migration_rate))
        elite_indices = np.argsort([sol.calc_energy() for sol in subpop])[:n_elite]
        migrants.extend([(label, subpop[i]) for i in elite_indices])

        # 选择随机移民
        n_random = max(1, int((1 - elite_ratio) * len(subpop) * migration_rate))
        random_indices = np.random.choice(len(subpop), size=n_random, replace=False)
        migrants.extend([(label, subpop[i]) for i in random_indices])

    # 2. 执行迁移 (避免源子群和目标子群相同)
    for src_label, migrant in migrants:
        possible_dst = [l for l in labels if l != src_label]
        if possible_dst:
            dst_label = random.choice(possible_dst)
            subpopulations[dst_label].append(migrant)

    return subpopulations