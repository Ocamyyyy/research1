import multiprocessing
import random
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd  # Added for Excel export
from task_definition import TaskConfig
from solution_utils import MappingSolution
from cuckoo import cuckoo_search
from GA import genetic_algorithm, simulated_annealing
from HHO import harris_hawks_optimization
from QHA import quantum_inspired_hyper_heuristics_algorithm


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
        "P_sw": 90, "P_npu": 120, "P_npu_beta": 20, "P_static": 24,
        "P_static_beta": 2, "P_dyn": 60, "P_dyn_beta": 20, "NWR": 3.0, "tot": tot
    }

    msg = f"{N}_{V}_{tot}"
    Task = TaskConfig()
    Task.init_class_vars(random_dict, mode="random")
    print(f"{N} {V} {tot} iter:{iterr} Task H:{Task.H} Task E Max:{Task.E_max}")

    results = {}
    
    # HEFT scheduler
    solution = MappingSolution(Task.CPU_number, Task.NPU_number, Task.RP_number, Task.task_number)
    start = time.time()
    solution.heft_scheduling(Task)
    solution.visualize_init(Task, window_name=f".\\{save_dic}\\heft scheduler {N}_{V}_{tot}_iter{iterr}", is_save=True)
    heft_energy = solution.calc_energy(Task)
    heft_delay = solution.calc_delay(Task)
    print(
        f"{N} {V} {tot} iter:{iterr} HEFT:{heft_energy:.2f} delay:{heft_delay:.2f} time:{time.time() - start:.2f}s")
    results['heft_energy'] = heft_energy / Task.E_max
    results['heft_delay'] = heft_delay

    # GA scheduler
    start = time.time()
    best_solution, best_energy = genetic_algorithm(Task, save_dic=save_dic, msg=msg, iterr=iterr, ecs_initial=False,
                                                   seed=24 + iterr,
                                                   pop_size=50,
                                                   max_iter=1000,
                                                   crossover_rate=0.8,
                                                   mutation_rate=0.1,
                                                   plot_number=3)
    # best_solution.energy_optimization(Task)
    ga_energy = best_solution.calc_energy(Task)
    ga_delay = best_solution.calc_delay(Task)
    print(
        f"{N} {V} {tot} iter:{iterr} GAA:{ga_energy:.2f} delay:{ga_delay:.2f} time:{time.time() - start:.2f}s")
    results['ga_energy'] = ga_energy / Task.E_max
    results['ga_delay'] = ga_delay

    # HHO scheduler
    start = time.time()
    best_solution, best_energy = harris_hawks_optimization(Task, n=50, save_dic=save_dic, msg=msg, iterr=iterr,
                                                           max_iter=1000, ecs_intial=False, is_modified=True)
    # best_solution.energy_optimization(Task)
    hho_energy = best_solution.calc_energy(Task)
    hho_delay = best_solution.calc_delay(Task)
    print(
        f"{N} {V} {tot} iter:{iterr} QC-HHO:{hho_energy:.2f} delay:{hho_delay:.2f} time:{time.time() - start:.2f}s")
    results['hho_modified_energy'] = hho_energy / Task.E_max
    results['hho_modified_delay'] = hho_delay

    # Cuckoo scheduler
    start = time.time()
    best_solution, best_energy = cuckoo_search(Task, save_dic=save_dic, msg=msg, iterr=iterr, n=50, max_iter=1000,
                                               ecs_initial=False)
    cuckoo_delay = best_solution.calc_delay(Task)
    print(
        f"{N} {V} {tot} iter:{iterr} GACSM:{best_energy:.2f} delay:{cuckoo_delay:.2f} time:{time.time() - start:.2f}s")
    results['cuckoo_energy'] = best_energy / Task.E_max
    results['cuckoo_delay'] = cuckoo_delay

    # QHA scheduler
    start = time.time()
    best_solution, best_energy = quantum_inspired_hyper_heuristics_algorithm(Task, pop_size=50, save_dic=save_dic,
                                                                             msg=msg, iterr=iterr,
                                                                             max_iter=1000, phase_iter=10,
                                                                             ecs_initial=False)
    # print(best_solution.processor_completed_task)
    # best_solution.energy_optimization(Task)
    qha_energy = best_solution.calc_energy(Task)
    qha_delay = best_solution.calc_delay(Task)
    print(
        f"{N} {V} {tot} iter:{iterr} QMES:{qha_energy:.2f} delay:{qha_delay:.2f} time:{time.time() - start:.2f}s")
    results['qha_energy'] = qha_energy / Task.E_max
    results['qha_delay'] = qha_delay

    return (N, V, tot, iterr), results


if __name__ == "__main__":
    random.seed(2)
    np.random.seed(2)

    test_list = [(2,10,40),(2,10,70),(2, 10, 100), (2, 10, 200), (2, 10, 300), (2, 10, 400), (2, 10, 500)]
    # test_list = [(2, 10, 40)]
    # test_list = [(2, 5, 30), (2, 5, 40), (2, 5, 50), (2, 5, 60), (2, 5, 70), (2, 5, 80), (2, 5, 90), (2, 5, 100)]
    tttt = 5
    start_tot = time.time()
    save_dic = "test1-formal, 3"
    print(save_dic)

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

    # experiments = [(2,10,40,3, save_dic)]

    # 使用进程池处理所有实验
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        all_results = pool.map(run_single_experiment, experiments)

    # 创建DataFrame来存储结果
    columns = ["N", "V", "tot", "iterr",
               "HEFT energy", "HEFT delay",
               "GAA energy", "GAA delay",
               "QC-HHO energy", "QC-HHO delay",
               "GACSM energy", "GACSM delay",
               "QHA energy","QHA delay"]

    data = []
    for (N, V, tot, iterr), results in all_results:
        row = [
            N, V, tot, iterr,
            results['heft_energy'] * 100, results['heft_delay'],
            results['ga_energy'] * 100, results['ga_delay'],
            results['hho_modified_energy'] * 100, results['hho_modified_delay'],
            results['cuckoo_energy'] * 100, results['cuckoo_delay'],
            results['qha_energy']*100, results['qha_delay'],
        ]
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    excel_path = f".\\{save_dic}\\experiment_results.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

    # 重新组织结果用于绘图 (保持原有绘图逻辑不变)
    NE_data = defaultdict(lambda: defaultdict(list))
    delay_data = defaultdict(lambda: defaultdict(list))
    for (N, V, tot, iterr), results in all_results:
        NE_data[(N, V, tot)]['heft'].append(results['heft_energy'])
        NE_data[(N, V, tot)]['ga'].append(results['ga_energy'])
        NE_data[(N, V, tot)]['hho_modified'].append(results['hho_modified_energy'])
        NE_data[(N, V, tot)]['cuckoo'].append(results['cuckoo_energy'])
        NE_data[(N, V, tot)]['qha'].append(results['qha_energy'])
        delay_data[(N, V, tot)]['heft'].append(results['heft_delay'])
        delay_data[(N, V, tot)]['ga'].append(results['ga_delay'])
        delay_data[(N, V, tot)]['hho_modified'].append(results['hho_modified_delay'])
        delay_data[(N, V, tot)]['cuckoo'].append(results['cuckoo_delay'])
        delay_data[(N, V, tot)]['qha'].append(results['qha_delay'])

    algorithms = ["HEFT", "GAA", "QC-HHO", "GACSM", "QMES"]
    # 计算平均值并填充NE_list
    NE_list = [[] for _ in range(len(algorithms))]
    delay_list = [[] for _ in range(len(algorithms))]
    test_case_order = sorted(NE_data.keys(), key=lambda x: x[2])  # 按tot排序

    for N, V, tot in test_case_order:
        avg_heft = np.mean(NE_data[(N, V, tot)]['heft']) * 100
        avg_cuckoo = np.mean(NE_data[(N, V, tot)]['cuckoo']) * 100
        avg_ga = np.mean(NE_data[(N, V, tot)]['ga']) * 100
        avg_hho_modified = np.mean(NE_data[(N, V, tot)]['hho_modified']) * 100
        avg_qha = np.mean(NE_data[(N, V, tot)]['qha']) * 100
        avg_heft_delay = np.mean(delay_data[(N, V, tot)]['heft'])
        avg_cuckoo_delay = np.mean(delay_data[(N, V, tot)]['cuckoo'])
        avg_ga_delay = np.mean(delay_data[(N, V, tot)]['ga'])
        avg_hho_modified_delay = np.mean(delay_data[(N, V, tot)]['hho_modified'])
        avg_qha_delay = np.mean(delay_data[(N, V, tot)]['qha'])

        NE_list[0].append(avg_heft)
        NE_list[1].append(avg_ga)
        NE_list[2].append(avg_hho_modified)
        NE_list[3].append(avg_cuckoo)
        NE_list[4].append(avg_qha)

        delay_list[0].append(avg_heft_delay)
        delay_list[1].append(avg_ga_delay)
        delay_list[2].append(avg_hho_modified_delay)
        delay_list[3].append(avg_cuckoo_delay)
        delay_list[4].append(avg_qha_delay)

        print(f"Average NE for ({N}, {V}, {tot}): "
              f"HEFT={avg_heft:.2f}%, "
              f"GAA={avg_ga:.2f}%, "
              f"QC-HHO={avg_hho_modified:.2f}%, ",
              f"GACSM={avg_cuckoo:.2f}%, "
              f"QHA={avg_qha:.2f}%")
        print(f"Average delay for ({N}, {V}, {tot}): "
              f"HEFT={avg_heft_delay}, "
              f"GAA={avg_ga_delay}, "
              f"QC-HHO={avg_hho_modified_delay}, ",
              f"GACSM={avg_cuckoo_delay}, "
              f"QHA={avg_qha_delay}")

    # 恢复标准输出
    sys.stdout = sys.__stdout__

    # 绘制结果图
    x_values = [str(tot) for _, _, tot in test_case_order]


    # 调整参数
    bar_width = 0.2  # 减少宽度以容纳更多柱子
    group_gap = 0.2  # 控制每组之间的总间隔
    index = np.arange(len(x_values)) * (len(algorithms) * bar_width + group_gap)  # 动态调整x轴位置

    # colors = [
    #     "#F08080",  # Misty Rose (低饱和粉红)
    #     "#FFA07A",  # Light Salmon (浅鲑鱼红)
    #     "#FFB347",  # Pastel Orange (柔和橙)
    #     "#FFD700",  # Gold (金色)
    #     "#FFF8DC"   # Cornsilk (玉米丝黄)
    # ]
    colors = [
        "#F5D9E6",
        "#DFE1E2",
        "#CEE2F5",
        "#CBE0BB",
        "#F7E5CA"
    ]

    plt.figure(figsize=(16, 8))  # 加宽画布

    # 绘制柱状图
    for i in range(len(algorithms)):
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
    plt.ylim(10, max(max(NE_list))* 1.1)  # 纵坐标从 10 开始，顶部留 10% 空间
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
    # plt.grid(axis='y', linestyle='--', alpha=0.6)  # 横向网格线

    plt.tight_layout()
    plt.savefig(f".\\{save_dic}\\experiment1-NE.png", dpi=300, bbox_inches='tight')  # 高分辨率保存

    plt.figure(figsize=(16, 8))  # 加宽画布

    # 绘制柱状图
    for i in range(len(algorithms)):
        plt.bar(
            index + i * bar_width,
            delay_list[i],
            width=bar_width,
            label=algorithms[i],
            color=colors[i],
            edgecolor='grey',
            alpha=0.8,
        )

    # 调整坐标轴和标签
    plt.xlabel('Task Number', fontsize=12)
    plt.ylabel('Average Delay', fontsize=12)
    plt.xticks(
        index + 2 * bar_width,  # 将刻度标签居中显示
        x_values,
        fontsize=10
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
    # plt.grid(axis='y', linestyle='--', alpha=0.6)  # 横向网格线

    plt.tight_layout()
    plt.savefig(f".\\{save_dic}\\experiment1-delay.png", dpi=300, bbox_inches='tight')  # 高分辨率保存
    print(f"Total time cost: {time.time() - start_tot:.2f}s")