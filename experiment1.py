import time
import random
import numpy as np
from task_definition import TaskConfig
from solution_utils import MappingSolution
# from search import cuckoo_search
from cuckoo import cuckoo_search
from GA import genetic_algorithm, adaptive_genetic_algorithm, genetic_simulated_annealing, simulated_annealing
# from NGA import niche_genetic_algorithm
import matplotlib.pyplot as plt
import os
import sys

if __name__== "__main__" :
    random.seed(2)
    np.random.seed(2)
    # 显式初始化类变量
    # TaskConfig.init_class_vars()
    # test_list = [(2,5,30), (2,5,40), (2,5,50), (2,5,60), (2,5,70), (2,5,80), (2,5,90), (2, 5, 100)]
    # test_list = [(2, 20, 500)]
    # test_list = [(2, 5, 100)]
    # test_list = [(2,10,100),(2,10,200),(2,10,300),(2,10,400), (2,10,500)]
    test_list = [(2,10,100),(2,10,200),(2,10,300),(2,10,400), (2,10,500)]
    tttt = 5
    start_tot = time.time()
    save_dic = "test1"
    if not os.path.exists(save_dic):
        os.makedirs(save_dic)
    with (open(f'.\\{save_dic}\\my_document.txt', 'w', encoding='utf-8') as f):
        # 定义一个类，用于同时写入文件和打印到控制台
        class Logger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, 'a', encoding='utf-8')

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()  # 实时刷新写入文件

            def flush(self):
                pass  # 需要定义flush方法以兼容文件操作


        # 替换sys.stdout
        sys.stdout = Logger(f'.\\{save_dic}\\my_document.txt')

        NE_list = [[] for _ in range(4)]
        for N, V, tot in test_list:
            NE = [0] * 4
            for iterr in range(tttt):
                msg = f"{N} {V} {tot}"
                print("--------------------------------------------------------------------")
                print(f"\n{N} {V} {tot} iter:{iterr}")
                random_dict = {"N": N, "V": V, "alpha": 1.0, "max_out": 2, "beta": 1, "Tpr": 100, "Tpr_beta": 20, "wDAG": 400,
                               "gamma_GPU": 2, "gamma_hw": 1.5, "P_sw": 90, "P_npu": 120, "P_npu_beta": 20,
                               "P_static": 24, "P_static_beta": 2, "P_dyn": 60, "P_dyn_beta": 20, "NWR": 1.0, "tot":tot}
                Task = TaskConfig()
                Task.init_class_vars(random_dict, mode="random")
                # TaskConfig.init_class_vars()
                print(Task.E_max)

                start = time.time()
                print("\ncuckoo scheduler:")
                best_solution, best_energy = cuckoo_search(Task, save_dic=save_dic,msg=msg, iterr=iterr,n=50, max_iter=1000)
                print(best_energy)
                print(best_solution.calc_delay(Task))
                print(f"time cost: {time.time() - start:.2f}s")
                NE[2] += best_energy/Task.E_max

                start = time.time()
                print("\nGA scheduler:")
                best_solution, best_energy = genetic_algorithm(Task, save_dic=save_dic,msg=msg, iterr=iterr,
                    seed=24,
                    pop_size=50,
                    max_iter=1000,
                    crossover_rate=0.8,
                    mutation_rate=0.1,
                    plot_number=3
                )
                best_solution.energy_optimization(Task)
                print(best_solution.calc_energy(Task))
                print(best_solution.calc_delay(Task))
                print(f"time cost: {time.time() - start:.2f}s")
                NE[3] += best_solution.calc_energy(Task)/Task.E_max

                solution = MappingSolution(
                    Task.CPU_number,
                    Task.NPU_number,
                    Task.RP_number,
                    Task.task_number
                )
                start = time.time()
                print("\nheft scheduler:")
                # heft算法
                solution.heft_scheduling(Task)
                solution.visualize_init(Task,window_name=f".\\{save_dic}\\heft scheduler {N}_{V}_{tot} iter{iterr}", is_save=True)
                print(f"time cost: {time.time() - start:.2f}s")
                print(solution.calc_energy(Task))
                print(solution.calc_delay(Task))
                # solution.energy_optimization()
                # solution.visualize_init(window_name="heft scheduler optimization", is_save=True)
                # print(f"time cost: {time.time() - start:.2f}s")
                NE[0] += solution.calc_energy(Task)/Task.E_max

                solution = MappingSolution(
                    Task.CPU_number,
                    Task.NPU_number,
                    Task.RP_number,
                    Task.task_number
                )
                start = time.time()
                print("\necs scheduler:")
                # ecs算法
                solution.ecs_scheduling(Task)
                solution.visualize_init(Task,window_name=f".\\{save_dic}\\ecs scheduler {N}_{V}_{tot} iter{iterr}", is_save=True)
                print(f"time cost: {time.time() - start:.2f}s")
                print(solution.calc_energy(Task))
                print(solution.calc_delay(Task))
                NE[1] += solution.calc_energy(Task)/Task.E_max

            NE = [ne/tttt*100 for ne in NE]
            for idx, ne in enumerate(NE):
                NE_list[idx].append(ne)
            print(f"average NE:{NE}")
        # 恢复标准输出
        sys.stdout = sys.__stdout__
        x_values = [str(x) for _, _, x in test_list]  # 转换为字符串方便显示
        algorithms = ["HEFT", "ECS", "GACSM", "HECRS" ]

        # 设置柱状图的宽度和位置
        bar_width = 0.2
        index = np.arange(len(x_values))

        colors = [
            "#8DA0CB",  # 浅蓝色
            "#66C2A5",  # 浅绿色
            "#FC8D62",  # 浅橙色
            "#FFFACD",  # 浅黄色
        ]

        plt.figure(figsize=(10, 6))

        # 为每种算法绘制柱状图，并指定颜色
        for i in range(4):
            plt.bar(
                index + i * bar_width,
                NE_list[i],
                bar_width,
                label=algorithms[i],
                color=colors[i],  # 使用低饱和度颜色
                # edgecolor='grey',  # 边框颜色（可选）
                alpha=0.8,  # 透明度（可选）
            )

        plt.xlabel('Task Number')
        plt.ylabel('Normalized Energy-consumption (%)')
        plt.xticks(index + bar_width * 1.5, x_values)
        plt.legend()
        # plt.grid(True, axis='y', linestyle='--', alpha=0.6)  # 更柔和的网格线
        plt.tight_layout()
        plt.savefig(f".\\{save_dic}\\experiment1.png")
        print(f"time cost: {time.time() - start_tot:.2f}s")
