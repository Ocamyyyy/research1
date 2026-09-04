import multiprocessing
import random
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from task_definition import TaskConfig
from solution_utils import MappingSolution
from cuckoo import cuckoo_search
from GA import genetic_algorithm
from HHO import harris_hawks_optimization


def run_single_experiment(args):
    """处理单个实验(N, V, tot, iterr)"""
    N, V, tot, iterr, save_dic = args

    # 每个进程需要有自己的随机种子
    random.seed(2 + hash((N, V, tot, iterr)) % 10000)
    np.random.seed(2 + hash((N, V, tot, iterr)) % 10000)

    # 创建随机字典
    random_dict = {
        "N": N, "V": V, "alpha": 1.0, "max_out": 2, "beta": 1, "Tpr": 100,
        "Tpr_beta": 20, "wDAG": 400, "gamma_GPU": 2, "gamma_hw": 1.5,
        "P_sw": 400, "P_npu": 120, "P_npu_beta": 20, "P_static": 24,
        "P_static_beta": 2, "P_dyn": 60, "P_dyn_beta": 20, "NWR": 1.0, "tot": tot
    }

    msg = f"{N}_{V}_{tot}"
    Task = TaskConfig()
    Task.init_class_vars(random_dict, mode="random")
    print(f"{N} {V} {tot} iter:{iterr} Task H:{Task.H} Task E Max:{Task.E_max}")

    results = {}

    # Cuckoo scheduler
    start = time.time()
    best_solution, best_energy = cuckoo_search(Task, save_dic=save_dic, msg=msg, iterr=iterr, n=50, max_iter=1000)
    print(
        f"{N} {V} {tot} iter:{iterr} cuckoo:{best_energy:.2f} delay:{best_solution.calc_delay(Task):.2f} time:{time.time() - start:.2f}s")
    results['cuckoo'] = best_energy / Task.E_max

    # GA scheduler
    start = time.time()
    best_solution, best_energy = genetic_algorithm(Task, save_dic=save_dic, msg=msg, iterr=iterr,
                                                   seed=24 + iterr,
                                                   pop_size=50,
                                                   max_iter=1000,
                                                   crossover_rate=0.8,
                                                   mutation_rate=0.1,
                                                   plot_number=3)
    best_solution.energy_optimization(Task)
    print(
        f"{N} {V} {tot} iter:{iterr} GA:{best_solution.calc_energy(Task):.2f} delay:{best_solution.calc_delay(Task):.2f} time:{time.time() - start:.2f}s")
    results['ga'] = best_solution.calc_energy(Task) / Task.E_max

    # HHO scheduler
    start = time.time()
    best_solution, best_energy = harris_hawks_optimization(Task, save_dic=save_dic, msg=msg, iterr=iterr,
                                                   max_iter=1000)
    best_solution.energy_optimization(Task)
    print(
        f"{N} {V} {tot} iter:{iterr} HHO:{best_solution.calc_energy(Task):.2f} delay:{best_solution.calc_delay(Task):.2f} time:{time.time() - start:.2f}s")
    results['hho'] = best_solution.calc_energy(Task) / Task.E_max

    # HEFT scheduler
    solution = MappingSolution(Task.CPU_number, Task.NPU_number, Task.RP_number, Task.task_number)
    start = time.time()
    solution.heft_scheduling(Task)
    solution.visualize_init(Task, window_name=f".\\{save_dic}\\heft scheduler {N}_{V}_{tot}_iter{iterr}", is_save=True)
    print(
        f"{N} {V} {tot} iter:{iterr} HEFT:{solution.calc_energy(Task):.2f} delay:{solution.calc_delay(Task):.2f} time:{time.time() - start:.2f}s")
    results['heft'] = solution.calc_energy(Task) / Task.E_max

    # ECS scheduler
    solution = MappingSolution(Task.CPU_number, Task.NPU_number, Task.RP_number, Task.task_number)
    start = time.time()
    solution.ecs_scheduling(Task)
    solution.visualize_init(Task, window_name=f".\\{save_dic}\\ecs scheduler {N}_{V}_{tot}_iter{iterr}", is_save=True)
    print(
        f"{N} {V} {tot} iter:{iterr} ECS:{solution.calc_energy(Task):.2f} delay:{solution.calc_delay(Task):.2f} time:{time.time() - start:.2f}s")
    results['ecs'] = solution.calc_energy(Task) / Task.E_max

    return (N, V, tot, iterr), results


if __name__ == "__main__":
    random.seed(2)
    np.random.seed(2)

    test_list = [(2, 10, 100), (2, 10, 200), (2, 10, 300), (2, 10, 400), (2, 10, 500)]
    # test_list = [(2, 10, 100)]
    tttt = 5
    start_tot = time.time()
    save_dic = "test1-4(heft for GA scheduling and ecs for GA HHO initial, P sw added, Dg decreased, HHO modified)"

    if not os.path.exists(save_dic):
        os.makedirs(save_dic)


    # 设置日志
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'a', encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            pass


    sys.stdout = Logger(f'.\\{save_dic}\\my_document.txt')

    # 准备所有实验参数
    experiments = [(N, V, tot, iterr, save_dic)
                   for N, V, tot in test_list
                   for iterr in range(tttt)]

    # 使用进程池处理所有实验
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        all_results = pool.map(run_single_experiment, experiments)

    # 重新组织结果
    NE_data = defaultdict(lambda: defaultdict(list))
    for (N, V, tot, iterr), results in all_results:
        NE_data[(N, V, tot)]['heft'].append(results['heft'])
        NE_data[(N, V, tot)]['ecs'].append(results['ecs'])
        NE_data[(N, V, tot)]['cuckoo'].append(results['cuckoo'])
        NE_data[(N, V, tot)]['ga'].append(results['ga'])
        NE_data[(N, V, tot)]['hho'].append(results['hho'])

    # 计算平均值并填充NE_list
    NE_list = [[] for _ in range(5)]
    test_case_order = sorted(NE_data.keys(), key=lambda x: x[2])  # 按tot排序

    for N, V, tot in test_case_order:
        avg_heft = np.mean(NE_data[(N, V, tot)]['heft']) * 100
        avg_ecs = np.mean(NE_data[(N, V, tot)]['ecs']) * 100
        avg_cuckoo = np.mean(NE_data[(N, V, tot)]['cuckoo']) * 100
        avg_ga = np.mean(NE_data[(N, V, tot)]['ga']) * 100
        avg_hho = np.mean(NE_data[(N, V, tot)]['hho']) * 100

        NE_list[0].append(avg_heft)
        NE_list[1].append(avg_ecs)
        NE_list[2].append(avg_cuckoo)
        NE_list[3].append(avg_ga)
        NE_list[4].append(avg_hho)

        print(f"Average NE for ({N}, {V}, {tot}): "
              f"HEFT={avg_heft:.2f}%, ECS={avg_ecs:.2f}%, "
              f"Cuckoo={avg_cuckoo:.2f}%, GA={avg_ga:.2f}%, HHO={avg_hho:.2f}%")

    # 恢复标准输出
    sys.stdout = sys.__stdout__

    # 绘制结果图
    x_values = [str(tot) for _, _, tot in test_case_order]
    algorithms = ["HEFT", "ECS", "GACSM", "HECRS", "HHO"]

    # 调整参数
    bar_width = 0.15  # 减少宽度以容纳更多柱子
    group_gap = 0.4  # 控制每组之间的总间隔
    index = np.arange(len(x_values)) * (5 * bar_width + group_gap)  # 动态调整x轴位置

    # 颜色方案（第4个改为浅紫色）
    colors = [
        "#8DA0CB",  # 浅蓝
        "#66C2A5",  # 浅绿
        "#FC8D62",  # 浅橙
        "#CBC3E3",  # 浅紫（原#EE82EE改为更柔和的#DDA0DD）
        "#FFFACD",  # 浅黄
    ]

    plt.figure(figsize=(12, 6))  # 加宽画布

    # 绘制柱状图
    for i in range(5):
        plt.bar(
            index + i * bar_width,
            NE_list[i],
            width=bar_width,
            label=algorithms[i],
            color=colors[i],
            edgecolor='grey',
            alpha=0.8,
        )

    # 调整坐标轴和标签
    plt.xlabel('Task Number', fontsize=12)
    plt.ylabel('Normalized Energy-consumption (%)', fontsize=12)
    plt.xticks(
        index + 2 * bar_width,  # 将刻度标签居中显示
        x_values,
        fontsize=10
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
    # plt.grid(axis='y', linestyle='--', alpha=0.6)  # 横向网格线

    plt.tight_layout()
    plt.savefig(f".\\{save_dic}\\experiment1.png", dpi=300, bbox_inches='tight')  # 高分辨率保存
    print(f"Total time cost: {time.time() - start_tot:.2f}s")