from solution_utils import (
    MappingSolution,
    assert_mapping_eligible,
    eligible_processor_ids,
    find_valid_swap_pair,
)
from task_definition import TaskConfig
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import math
import copy

def genetic_algorithm(Task:TaskConfig,save_dic="save_test",seed=24, msg="", iterr=0,pop_size=50, max_iter=1000,
                      crossover_rate=0.8, mutation_rate=0.1, elitism_ratio=0.1, plot_number=4,ecs_initial=False):
    """
    遗传算法实现任务调度优化

    参数:
    pop_size -- 种群大小 (默认50)
    max_iter -- 最大迭代次数 (默认1000)
    crossover_rate -- 交叉概率 (默认0.8)
    mutation_rate -- 变异概率 (默认0.1)
    elitism_ratio -- 精英保留比例 (默认0.1)
    """
    # np.random.seed(seed)
    # random.seed(seed)

    # 记录收敛数据
    convergence = {
        'best_energy': [],
        'generation_best_energy':[],
        'best_delay': [],
        'avg_energy': [],
        'std_energy': [],
        'avg_delay': [],
        'feasible_ratio': [],
        'count_crossover':[],
        'count_mutate':[]
    }

    start_time = time.time()

    # 初始化种群
    population = []
    for i in range(pop_size):
        solution = MappingSolution(
            Task.CPU_number,
            Task.NPU_number,
            Task.RP_number,
            Task.task_number
        )
        if i== 0 and ecs_initial:
            solution.ecs_scheduling(Task)
        else:
            solution.init_randomly(Task)
            # solution.scheduling(Task)
            solution.heft_scheduling_for_certain_task_mapping(Task)
        population.append(solution)

    # 初始评估
    energy = [sol.calc_energy(Task) for sol in population]
    delay = [sol.calc_delay(Task) for sol in population]
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
    best_solution = population[best_idx]
    best_energy = energy[best_idx]

    # print(f"Best Energy: {best_energy:.4f}")
    # best_solution.visualize_init(is_block=True)

    # 记录初始状态
    convergence['best_energy'].append(best_energy)
    convergence['generation_best_energy'].append(best_energy)
    convergence['best_delay'].append(delay[best_idx])
    convergence['avg_energy'].append(np.mean(energy))
    convergence['std_energy'].append(np.std(energy))
    convergence['avg_delay'].append(np.mean(delay))
    convergence['feasible_ratio'].append(np.mean(np.array(delay) == 0))

    # 主循环
    for iteration in range(max_iter):
        # 1. 选择 (轮盘赌选择)
        selected_indices = selection(fitness, pop_size)
        selected_population = [population[i] for i in selected_indices]

        # 2. 交叉 (单点交叉)
        offspring = []
        count_crossover = 0
        for i in range(0, pop_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < pop_size else selected_population[0]

            if random.random() < crossover_rate:
                count_crossover += 1
                child1, child2 = crossover(Task,parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        # 3. 变异 (随机翻转)
        count_mutate = 0
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                count_mutate += 1
                offspring[i] = mutate(Task,offspring[i], is_random=True)

        # 4. 父代精英保留
        elite_size = int(elitism_ratio * pop_size)
        elite_indices = np.argsort(fitness)[:elite_size]
        elite = [copy.deepcopy(population[i]) for i in elite_indices]
        offspring = elite + offspring[:pop_size - elite_size]
        # 随机抽取自代
        # offspring_selected = random.sample(offspring, pop_size - elite_size)
        # offspring = elite + offspring_selected
        # 显示排序子代
        # offspring_sorted = sorted(offspring, key=lambda x: x.calc_fitness())
        # offspring = elite + offspring_sorted[:pop_size - elite_size]

        # 更新种群
        population = offspring

        # 重新评估
        energy = [sol.calc_energy(Task) for sol in population]
        delay = [sol.calc_delay(Task) for sol in population]
        fitness = fitness_function(energy, delay)
        current_best_idx = np.argmin(fitness)

        # 更新全局最优
        if (population[current_best_idx].calc_energy(Task) <= best_solution.calc_energy(Task) and population[current_best_idx].calc_delay(Task) <= best_solution.calc_delay(Task))\
                or population[current_best_idx].calc_delay(Task) < best_solution.calc_delay(Task):
            best_solution = population[current_best_idx]
            best_energy = energy[current_best_idx]

        # 记录数据
        convergence['best_energy'].append(best_energy)
        convergence['generation_best_energy'].append(population[current_best_idx].calc_energy(Task))
        convergence['best_delay'].append(best_solution.calc_delay(Task))
        convergence['avg_energy'].append(np.mean(energy))
        convergence['std_energy'].append(np.std(energy))
        convergence['avg_delay'].append(np.mean(delay))
        convergence['feasible_ratio'].append(np.mean(np.array(delay) == 0))
        convergence['count_crossover'].append(count_crossover)
        convergence['count_mutate'].append(count_mutate)
        # if iteration % 100==0:
        #     print(f"Iteration {iteration + 1}: Best Energy = {best_energy:.4f}, {best_solution.calc_energy(Task)},number of crossover = {count_crossover}")

    # 输出结果
    # print(f"Total number of iterations: {max_iter}")
    # print(f"Best Energy: {best_energy:.4f}")
    # print(f"Best Delay: {best_solution.calc_delay()}")
    # print(f"Time Cost: {time.time() - start_time:.2f}s")
    # print(f"Crossover Average: {np.mean(convergence['count_crossover'])}")
    # print(f"Mutation Average: {np.mean(convergence['count_mutate'])}")

    # 绘制收敛曲线
    # best_solution.energy_optimization(Task)
    plot_convergence(convergence,plot_number=3,save_name=f".\\{save_dic}\\GA scheduler {msg} iter{iterr} convergence history.png")
    best_solution.visualize_init(Task,window_name=f".\\{save_dic}\\GA scheduler {msg} iter{iterr} energy{best_solution.calc_energy(Task)} energy_percent{best_solution.calc_energy(Task) / Task.E_max} delay{best_solution.calc_delay(Task)}", is_save=True)

    return best_solution, best_solution.calc_energy(Task)

def adaptive_genetic_algorithm(Task:TaskConfig, seed=24, pop_size=50, max_iter=1000, crossover_rate=0.8, mutation_rate=0.1,
                               elitism_ratio=0.1, plot_number=4):
    """
    遗传算法实现任务调度优化

    参数:
    pop_size -- 种群大小 (默认50)
    max_iter -- 最大迭代次数 (默认1000)
    crossover_rate -- 交叉概率 (默认0.8)
    mutation_rate -- 变异概率 (默认0.1)
    elitism_ratio -- 精英保留比例 (默认0.1)
    """
    np.random.seed(seed)
    random.seed(seed)

    # 记录收敛数据
    convergence = {
        'best_energy': [],
        'best_delay': [],
        'avg_energy': [],
        'std_energy': [],
        'avg_delay': [],
        'feasible_ratio': [],
        'count_crossover':[],
        'count_mutate':[]
    }

    start_time = time.time()

    # 初始化种群
    population = []
    for _ in range(pop_size):
        solution = MappingSolution(
            Task.CPU_number,
            Task.NPU_number,
            Task.RP_number,
            Task.task_number
        )
        solution.init_randomly(Task)
        solution.scheduling(Task)
        population.append(solution)

    # 初始评估
    energy = [sol.calc_energy(Task) for sol in population]
    delay = [sol.calc_delay(Task) for sol in population]
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
    best_solution = population[best_idx]
    best_energy = energy[best_idx]

    print(f"Best Energy: {best_energy:.4f}")
    best_solution.visualize_init(Task,is_block=True)

    # 记录初始状态
    convergence['best_energy'].append(best_energy)
    convergence['best_delay'].append(delay[best_idx])
    convergence['avg_energy'].append(np.mean(energy))
    convergence['std_energy'].append(np.std(energy))
    convergence['avg_delay'].append(np.mean(delay))
    convergence['feasible_ratio'].append(np.mean(np.array(delay) == 0))

    # 主循环
    for iteration in range(max_iter):
        # 1. 选择 (轮盘赌选择)
        selected_indices = selection(fitness, pop_size)
        selected_population = [population[i] for i in selected_indices]
        # 适应度计算 (用于自适应交叉概率计算)
        energy = [sol.calc_energy(Task) for sol in selected_population]
        delay = [sol.calc_delay(Task) for sol in selected_population]
        selected_fitness = fitness_function(energy, delay)
        selected_fitness_average= np.mean(selected_fitness)
        selected_fitness_min = min(selected_fitness)
        selected_fitness_max = max(selected_fitness)

        # 2. 交叉 (单点交叉)
        offspring = []
        count_crossover = 0
        for i in range(0, pop_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < pop_size else selected_population[0]
            # if iteration > 0.5* max_iter:
            # 自适应交叉概率计算
            fitness1 = selected_fitness[i]
            fitness2 = selected_fitness[i + 1] if i + 1 < pop_size else selected_fitness[0]
            fitness_choose = min(fitness1,fitness2)
            # pc = crossover_rate*(fitness_choose - selected_fitness_min)/(selected_fitness_average - selected_fitness_min) if fitness_choose < selected_fitness_average else crossover_rate
            pc = crossover_rate - crossover_rate * 0.5 * (selected_fitness_max - fitness_choose) / (
                        selected_fitness_max-selected_fitness_average) if fitness_choose < selected_fitness_average else crossover_rate
            # else:
            #     pc = crossover_rate

            if random.random() < pc:
                count_crossover += 1
                child1, child2 = crossover(Task,parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        energy = [sol.calc_energy(Task) for sol in offspring]
        delay = [sol.calc_delay(Task) for sol in offspring]
        offspring_fitness = fitness_function(energy, delay)
        offspring_fitness_average = np.mean(offspring_fitness)
        offspring_fitness_max = max(offspring_fitness)
        offspring_fitness_min = min(offspring_fitness)

        # print(offspring_fitness)

        # 3. 变异 (随机翻转)
        count_mutate = 0
        for i in range(len(offspring)):
            if offspring_fitness_max != offspring_fitness_average:
                pm = (mutation_rate - mutation_rate * 0.5 * (offspring_fitness_max - offspring_fitness[i])
                      / (offspring_fitness_max - offspring_fitness_average)) if offspring_fitness[i] < offspring_fitness_average else mutation_rate
            else:
                pm = mutation_rate
            if random.random() < pm:
                count_mutate+=1
                offspring[i] = mutate(Task,offspring[i])

        # 4. 父代精英保留
        elite_size = int(elitism_ratio * pop_size)
        elite_indices = np.argsort(fitness)[:elite_size]
        elite = [population[i] for i in elite_indices]
        offspring = elite + offspring[:pop_size - elite_size]
        # 随机抽取自代
        # offspring_selected = random.sample(offspring, pop_size - elite_size)
        # offspring = elite + offspring_selected
        # 显示排序子代
        # offspring_sorted = sorted(offspring, key=lambda x: x.calc_fitness())
        # offspring = elite + offspring_sorted[:pop_size - elite_size]

        # 更新种群
        population = offspring

        # 重新评估
        energy = [sol.calc_energy(Task) for sol in population]
        delay = [sol.calc_delay(Task) for sol in population]
        fitness = fitness_function(energy, delay)
        current_best_idx = np.argmin(fitness)

        # 更新全局最优
        if population[current_best_idx].calc_energy(Task) < best_solution.calc_energy(Task):
            best_solution = population[current_best_idx]
            best_energy = energy[current_best_idx]

        # 记录数据
        convergence['best_energy'].append(best_energy)
        convergence['best_delay'].append(delay[current_best_idx])
        convergence['avg_energy'].append(np.mean(energy))
        convergence['std_energy'].append(np.std(energy))
        convergence['avg_delay'].append(np.mean(delay))
        convergence['feasible_ratio'].append(np.mean(np.array(delay) == 0))
        convergence['count_crossover'].append(count_crossover)
        convergence['count_mutate'].append(count_mutate)

        print(f"Iteration {iteration + 1}: Best Energy = {best_energy:.4f}, number of crossover = {count_crossover}"
              f", number of mutation = {count_mutate}")

    # 输出结果
    print(f"Total number of iterations: {max_iter}")
    print(f"Best Energy: {best_energy:.4f}")
    print(f"Best Delay: {best_solution.calc_delay(Task)}")
    print(f"Time Cost: {time.time() - start_time:.2f}s")
    print(f"Crossover Average: {np.mean(convergence['count_crossover'])}")
    print(f"Mutation Average: {np.mean(convergence['count_mutate'])}")

    # 绘制收敛曲线
    plot_convergence(convergence, plot_number)
    best_solution.visualize_init(Task,window_name="GA Best Solution", is_block=True)

    return best_solution, best_energy

def genetic_simulated_annealing(Task:TaskConfig, seed=24, pop_size=50, max_iter=1000, crossover_rate=0.8, mutation_rate=0.1, elitism_ratio=0.1, plot_number=4):
    # np.random.seed(seed)
    # random.seed(seed)

    # 记录收敛数据
    convergence = {
        'best_energy': [],
        'generation_best_energy': [],
        'best_delay': [],
        'avg_energy': [],
        'std_energy': [],
        'avg_delay': [],
        'feasible_ratio': [],
        'count_crossover': [],
        'count_mutate': []
    }

    start_time = time.time()

    # 初始化种群
    population = []
    for _ in range(pop_size):
        solution = MappingSolution(
            Task.CPU_number,
            Task.NPU_number,
            Task.RP_number,
            Task.task_number
        )
        solution.init_randomly(Task)
        solution.scheduling(Task)
        population.append(solution)

    # 初始评估
    energy = [sol.calc_energy(Task) for sol in population]
    delay = [sol.calc_delay(Task) for sol in population]
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
    best_solution = population[best_idx]
    best_energy = energy[best_idx]

    print(f"Best Energy: {best_energy:.4f}")
    best_solution.visualize_init(Task,is_block=True)

    # 记录初始状态
    convergence['best_energy'].append(best_energy)
    convergence['generation_best_energy'].append(best_energy)
    convergence['best_delay'].append(delay[best_idx])
    convergence['avg_energy'].append(np.mean(energy))
    convergence['std_energy'].append(np.std(energy))
    convergence['avg_delay'].append(np.mean(delay))
    convergence['feasible_ratio'].append(np.mean(np.array(delay) == 0))

    # 主循环
    for iteration in range(max_iter):
        # 1. 选择 (轮盘赌选择)
        selected_indices = selection(fitness, pop_size)
        selected_population = [population[i] for i in selected_indices]

        # 2. 交叉 (单点交叉)
        offspring = []
        count_crossover = 0
        for i in range(0, pop_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < pop_size else selected_population[0]

            if random.random() < crossover_rate:
                count_crossover += 1
                child1, child2 = crossover(Task,parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        # 3. 变异 (随机翻转)
        count_mutate = 0
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                # mutate_individual = mutate(offspring[i])
                # mutate_energy = mutate_individual.calc_energy()
                # original_energy = offspring[i].calc_energy()
                # if mutate_energy < original_energy:
                #     count_mutate += 1
                #     offspring[i] = mutate_individual
                # else:
                #     if random.random() < original_energy/(mutate_energy + original_energy):
                #         count_mutate += 1
                #         offspring[i] = mutate_individual
                count_mutate += 1
                offspring[i] = mutate(Task,offspring[i])

        # 4. 父代精英保留
        elite_size = int(elitism_ratio * pop_size)
        elite_indices = np.argsort(fitness)[:elite_size]
        elite = [population[i] for i in elite_indices]
        offspring = elite + offspring[:pop_size - elite_size]

        if max_iter - iteration < 10 or iteration<10:
            energy = [sol.calc_energy(Task) for sol in offspring]
            delay = [sol.calc_delay(Task) for sol in offspring]
            offspring_fitness = fitness_function(energy, delay)
            current_best_idx = np.argmin(offspring_fitness)

            # 5. 模拟退火局部搜索
            energy_reserve = energy[current_best_idx]
            # print(delay[current_best_idx])
            new_individual,_ = simulated_annealing(Task,offspring[current_best_idx])
            offspring[current_best_idx] = new_individual
            energy[current_best_idx] = new_individual.calc_energy(Task)
            delay[current_best_idx] = new_individual.calc_delay(Task)
            print(f"SA before:{energy_reserve}, SA after:{energy[current_best_idx]}")
            # print(delay[current_best_idx])

            # 更新种群
            population = offspring

        # 重新评估
        else:
            population = offspring
            energy = [sol.calc_energy() for sol in population]
            delay = [sol.calc_delay() for sol in population]
        fitness = fitness_function(energy, delay)
        current_best_idx = np.argmin(fitness)

        # 更新全局最优
        if population[current_best_idx].calc_energy(Task) < best_solution.calc_energy(Task):
            best_solution = population[current_best_idx]
            best_energy = energy[current_best_idx]

        # 记录数据
        convergence['best_energy'].append(best_energy)
        convergence['generation_best_energy'].append(population[current_best_idx].calc_energy(Task))
        convergence['best_delay'].append(delay[current_best_idx])
        convergence['avg_energy'].append(np.mean(energy))
        convergence['std_energy'].append(np.std(energy))
        convergence['avg_delay'].append(np.mean(delay))
        convergence['feasible_ratio'].append(np.mean(np.array(delay) == 0))
        convergence['count_crossover'].append(count_crossover)
        convergence['count_mutate'].append(count_mutate)

        print(f"Iteration {iteration + 1}: Best Energy = {best_energy:.4f}, number of crossover = {count_crossover}")

    # 输出结果
    print(f"Total number of iterations: {max_iter}")
    print(f"Best Energy: {best_energy:.4f}")
    print(f"Best Delay: {best_solution.calc_delay(Task)}")
    print(f"Time Cost: {time.time() - start_time:.2f}s")
    print(f"Crossover Average: {np.mean(convergence['count_crossover'])}")
    print(f"Mutation Average: {np.mean(convergence['count_mutate'])}")

    # 绘制收敛曲线
    plot_convergence(convergence, plot_number)
    best_solution.visualize_init(Task,window_name="GA Best Solution", is_block=True)

    return best_solution, best_energy


def selection(fitness, pop_size):
    """轮盘赌选择"""
    inverted_fitness = 1 / (np.array(fitness) + 1e-6)  # 避免除零
    prob = inverted_fitness / np.sum(inverted_fitness)
    return np.random.choice(range(len(fitness)), size=pop_size, p=prob)


def crossover(Task:TaskConfig, parent1, parent2):
    """单点交叉"""
    crossover_point = random.randint(1, Task.task_number - 1)
    child1_mapping = parent1.task_mapping[:crossover_point] + parent2.task_mapping[crossover_point:]
    child2_mapping = parent2.task_mapping[:crossover_point] + parent1.task_mapping[crossover_point:]
    assert_mapping_eligible(Task, child1_mapping, context="GA crossover child1")
    assert_mapping_eligible(Task, child2_mapping, context="GA crossover child2")
    # child1_mapping = simplebounds(child1_mapping,Task.task_number,Task.processor_number)
    # child2_mapping = simplebounds(child2_mapping, Task.task_number, Task.processor_number)

    child1 = MappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number
    )
    child1.init_definitely(child1_mapping)
    # child1.scheduling(Task)
    child1.heft_scheduling_for_certain_task_mapping(Task)

    child2 = MappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number
    )
    child2.init_definitely(child2_mapping)
    # child2.scheduling(Task)
    child2.heft_scheduling_for_certain_task_mapping(Task)

    return child1, child2


def mutate(Task:TaskConfig, solution, is_random=False):
    """随机变异一个处理器分配"""
    mutated_mapping = solution.task_mapping.copy()
    if is_random:
        choice = random.choice(["swap", "mutate", "block_move"])
        if choice == "swap":
            swap_pair = find_valid_swap_pair(Task, mutated_mapping, max_attempts=100)
            if swap_pair is not None:
                i, j = swap_pair
                mutated_mapping[i], mutated_mapping[j] = mutated_mapping[j], mutated_mapping[i]
        elif choice == "mutate":
            mutate_pos = random.randint(0, Task.task_number - 1)
            original_value = mutated_mapping[mutate_pos]
            candidates = [p for p in eligible_processor_ids(Task, mutate_pos) if p != original_value]
            if candidates:
                mutated_mapping[mutate_pos] = random.choice(candidates)
        elif choice == "block_move":
            for _ in range(100):
                block_size = random.randint(1, min(5, len(mutated_mapping) // 2))
                start = random.randint(0, len(mutated_mapping) - block_size)
                block = list(range(start, start + block_size))
                common_eligible = set(eligible_processor_ids(Task, block[0]))
                for task_pos in block[1:]:
                    common_eligible &= set(eligible_processor_ids(Task, task_pos))
                current_processors = {mutated_mapping[task_pos] for task_pos in block}
                candidate_processors = [
                    p for p in sorted(common_eligible)
                    if p not in current_processors or len(current_processors) > 1
                ]
                if not candidate_processors:
                    continue
                new_proc = random.choice(candidate_processors)
                for task_pos in block:
                    mutated_mapping[task_pos] = new_proc
                break
    else:
        mutate_pos = random.randint(0, Task.task_number - 1)
        original_value = mutated_mapping[mutate_pos]
        candidates = [p for p in eligible_processor_ids(Task, mutate_pos) if p != original_value]
        if candidates:
            mutated_mapping[mutate_pos] = random.choice(candidates)
    assert_mapping_eligible(Task, mutated_mapping, context="GA mutate")
    # mutated_mapping = simplebounds(mutated_mapping, Task.task_number, Task.processor_number)
    new_solution = MappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number
    )
    new_solution.init_definitely(mutated_mapping)
    # new_solution.scheduling(Task)
    new_solution.heft_scheduling_for_certain_task_mapping(Task)
    return new_solution


def simulated_annealing(Task:TaskConfig, solution: MappingSolution, save_dict="save_test",msg="", iterr=0, alpha=0.95, max_iter=5000,
                        steps_per_temp=10, p0=0.8, is_show_best=False, is_plot_history=False,is_save_best = False):
    current_mapping = solution.task_mapping.copy()
    current_energy = solution.calc_energy(Task)
    current_delay = solution.calc_delay(Task)
    n = len(current_mapping)

    # === 1. 动态估算初始温度 temp ===
    def estimate_initial_temp(solution, p0, num_samples=100):
        delta_es = []
        for _ in range(num_samples):
            # 随机选择一种邻域操作生成新解
            new_mapping = generate_neighbor(current_mapping)

            # 计算新解的目标函数差值
            new_solution = MappingSolution(
                Task.CPU_number,
                Task.NPU_number,
                Task.RP_number,
                Task.task_number
            )
            new_solution.init_definitely(new_mapping)
            new_solution.heft_scheduling_for_certain_task_mapping(Task)
            delta_e = new_solution.calc_energy(Task) - solution.calc_energy(Task)
            delta_es.append(abs(delta_e))

        avg_delta_e = sum(delta_es) / num_samples
        temp = avg_delta_e / (-math.log(p0))  # T0 = <|ΔE|> / ln(1/p0)
        return max(temp, 1e-6)  # 避免除零

    # === 2. 定义多种邻域生成方式 ===
    def generate_neighbor(mapping):
        new_mapping = mapping.copy()
        choice = random.choice(["swap", "mutate", "block_move"])
        # choice = random.choice(["swap", "mutate"])

        if choice == "swap":
            swap_pair = find_valid_swap_pair(Task, new_mapping, max_attempts=100)
            if swap_pair is not None:
                i, j = swap_pair
                new_mapping[i], new_mapping[j] = new_mapping[j], new_mapping[i]
        elif choice == "mutate":
            mutate_pos = random.randint(0, Task.task_number - 1)
            candidates = [p for p in eligible_processor_ids(Task, mutate_pos) if p != new_mapping[mutate_pos]]
            if candidates:
                new_mapping[mutate_pos] = random.choice(candidates)
        elif choice == "block_move":
            for _ in range(100):
                block_size = random.randint(1, min(5, n // 2))
                start = random.randint(0, n - block_size)
                block = list(range(start, start + block_size))
                common_eligible = set(eligible_processor_ids(Task, block[0]))
                for task_pos in block[1:]:
                    common_eligible &= set(eligible_processor_ids(Task, task_pos))
                current_processors = {new_mapping[task_pos] for task_pos in block}
                candidate_processors = [
                    p for p in sorted(common_eligible)
                    if p not in current_processors or len(current_processors) > 1
                ]
                if not candidate_processors:
                    continue
                new_proc = random.choice(candidate_processors)
                for task_pos in block:
                    new_mapping[task_pos] = new_proc
                break
        assert_mapping_eligible(Task, new_mapping, context="SA neighbor")
        return new_mapping

    # 初始化温度参数
    temp = estimate_initial_temp(solution, p0)
    temp_end = 1e-7 * temp  # 最终温度与初始温度成比例
    # print(temp, temp_end)
    current_temp = temp

    history = []
    best_history = []
    best_history_delay = []
    best_mapping = current_mapping.copy()
    best_energy = current_energy
    best_delay = current_delay

    while current_temp > temp_end and len(history) < max_iter:
        for _ in range(steps_per_temp):
            # 随机选择一种邻域操作生成新解
            new_mapping = generate_neighbor(current_mapping)

            # 评估新解
            new_solution = MappingSolution(
                Task.CPU_number,
                Task.NPU_number,
                Task.RP_number,
                Task.task_number
            )
            new_solution.init_definitely(new_mapping)
            new_solution.heft_scheduling_for_certain_task_mapping(Task)
            new_energy = new_solution.calc_energy(Task)
            new_delay = new_solution.calc_delay(Task)

            # === 硬约束优化：仅接受延迟不劣化的解 ===
            if new_delay <= current_delay:
                energy_diff = new_energy - current_energy
                accept_prob = math.exp(-energy_diff / current_temp) if energy_diff > 0 else 1.0
                if random.random() < accept_prob:
                    current_mapping = new_mapping.copy()
                    current_energy = new_energy
                    current_delay = new_delay
                    # 更新历史最优解
                    if current_energy < best_energy or (
                            current_energy == best_energy and current_delay < best_delay
                    ):
                        best_mapping = current_mapping.copy()
                        best_energy = current_energy
                        best_delay = current_delay
            # if is_plot_history or is_save_best:
            #     if len(history) % 100 == 0:
            #         print(f'iter: {len(history)}, current temperature: {current_temp}, best energy: {best_energy}, '
            #               f'current energy: {current_energy}, best delay:{best_delay}')
            history.append((current_energy, current_delay))
            best_history.append(best_energy)
            best_history_delay.append(best_delay)

        current_temp *= alpha  # 温度下降

    # 返回最优解
    best_solution = MappingSolution(
        Task.CPU_number,
        Task.NPU_number,
        Task.RP_number,
        Task.task_number
    )
    best_solution.init_definitely(best_mapping)
    best_solution.heft_scheduling_for_certain_task_mapping(Task)
    if is_show_best:
        best_solution.visualize_init(Task,window_name=f".\\{save_dict}\\SA scheduler {msg} iter{iterr}",is_block=False)
    elif is_save_best:
        best_solution.visualize_init(Task,window_name=f".\\{save_dict}\\SA scheduler {msg} iter{iterr}",is_save=True)
        plt.close("all")

        # 重新创建新窗口
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        energy_list = [his[0] for his in history]
        plt.plot(energy_list, label=f'energy')
        plt.plot(best_history, label="best energy")
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Energy Trends')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(best_history_delay, 'g-', label='Best Delay')
        # plt.plot(convergence['avg_delay'], 'm--', label='Average Delay')
        plt.xlabel('Iteration')
        plt.ylabel('Delay')
        plt.title('Delay Trends')
        plt.legend()
        plt.grid(True)
        plt.savefig(f".\\{save_dict}\\SA scheduler {msg} iter{iterr} convergence history.png")
        plt.close("all")
    if is_plot_history:
        energy_list = [his[0] for his in history]
        plt.plot(energy_list, label=f'energy')
        plt.plot(best_history, label="best energy")
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Energy Trends')
        plt.legend()
        plt.grid(True)
        plt.show(block=True)
    return best_solution, best_energy

def fitness_function(energy_list, time_delay_list):
    """适应度函数（与CS算法保持一致）"""
    fitness_list = []
    for i, energy in enumerate(energy_list):
        if time_delay_list[i] == 0:
            fitness = energy / max(energy_list)
        else:
            fitness = 1 + time_delay_list[i] / max(time_delay_list)
        fitness_list.append(fitness)
    return fitness_list


def plot_convergence(convergence, plot_number=3,save_name="convergence history.png"):
    """绘制收敛曲线（与CS算法相同）"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, plot_number, 1)
    plt.plot(convergence['best_energy'], 'r-', label='Best Energy')
    # if 'generation_best_energy' in convergence:
    #     plt.plot(convergence['generation_best_energy'],'y-',label='Best Energy in Generation')
    # plt.plot(convergence['avg_energy'], 'b--', label='Average Energy')
    # plt.fill_between(range(len(convergence['avg_energy'])),
    #                  np.array(convergence['avg_energy']) - np.array(convergence['std_energy']),
    #                  np.array(convergence['avg_energy']) + np.array(convergence['std_energy']),
    #                  alpha=0.2)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy Convergence')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, plot_number, 2)
    plt.plot(convergence['best_delay'], 'g-', label='Best Delay')
    # plt.plot(convergence['avg_delay'], 'm--', label='Average Delay')
    plt.xlabel('Iteration')
    plt.ylabel('Delay')
    plt.title('Delay Trends')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, plot_number, 3)
    plt.plot(np.array(convergence['feasible_ratio']) * 100, 'k-')
    plt.xlabel('Iteration')
    plt.ylabel('Feasible Solutions (%)')
    plt.title('Feasibility Ratio')
    plt.grid(True)

    # 子图4：Crossover Count
    if plot_number == 4:
        plt.subplot(1, plot_number, 4)
        plt.plot(convergence['count_crossover'], 'c-', label='Crossover Count')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Crossovers')
        plt.title('Crossover Operations per Iteration')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_name)

# if __name__ == "__main__":
#     np.random.seed(24)
#     random.seed(24)
#     random_dict = {"N": 2, "V": 10, "alpha": 1.0, "max_out": 2, "beta": 1, "Tpr": 100, "Tpr_beta": 20, "wDAG": 400,
#                    "gamma_GPU": 2, "gamma_hw": 1.5, "P_sw": 90, "P_npu": 120, "P_npu_beta": 20,
#                    "P_static": 24, "P_static_beta": 2, "P_dyn": 60, "P_dyn_beta": 20, "NWR": 5.0, "tot": 100}
#     Task = TaskConfig()
#     Task.init_class_vars(random_dict, mode="random")
#     print(Task.H)
#
#     best_solution, best_energy = genetic_algorithm(Task,
#                                                    pop_size=50,
#                                                    max_iter=1000,
#                                                    crossover_rate=0.8,
#                                                    mutation_rate=0.1,
#                                                    plot_number=3)
#     print(best_energy)
#     print(best_solution.calc_delay(Task))

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
