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
import gc


def run_single_experiment(args):
    """处理单个实验(N, V, tot, iterr)"""
    N, V, tot, alpha, max_out, NWR, RP_number,iterr, save_dic = args

    # 每个进程需要有自己的随机种子
    random.seed(2 + hash((N, V, tot, alpha, max_out, NWR, RP_number, iterr)) % 10000)
    np.random.seed(2 + hash((N, V, tot, iterr)) % 10000)

    # 创建随机字典
    random_dict = {
        "N": N, "V": V, "alpha": alpha, "max_out": max_out, "beta": 1, "Tpr": 100,
        "Tpr_beta": 20, "wDAG": 400, "gamma_GPU": 2, "gamma_hw": 1.5,
        "P_sw": 90, "P_npu": 120, "P_npu_beta": 20, "P_static": 24,
        "P_static_beta": 2, "P_dyn": 60, "P_dyn_beta": 20, "NWR": NWR, "tot": tot
    }
    # print(f"NWR:{random_dict['NWR']}")

    msg = f"{N}_{V}_{tot}_alpha{alpha}_max_out{max_out}_NWR{NWR}_RP{RP_number}"
    Task = TaskConfig()
    Task.init_class_vars(random_dict, mode="random")
    Task.change_RP_number(RP_number, random_dict)
    # print(f"RP number:{Task.RP_number}")
    print(f"{N} {V} {tot} alpha{alpha} max_out{max_out} NWR{NWR} RP{RP_number} iter:{iterr} Task H:{Task.H} Task E Max:{Task.E_max}")

    results = {}

    # HEFT scheduler
    solution = MappingSolution(Task.CPU_number, Task.NPU_number, Task.RP_number, Task.task_number)
    start = time.time()
    solution.heft_scheduling(Task)

    heft_energy = solution.calc_energy(Task)
    heft_delay = solution.calc_delay(Task)
    solution.visualize_init(Task, window_name=f".\\{save_dic}\\heft scheduler {msg} iter{iterr} energy{heft_energy} energy_percent{heft_energy / Task.E_max} delay{heft_delay}", is_save=True)
    print(
        f"{N} {V} {tot} alpha{alpha} max_out{max_out} NWR{NWR} RP{RP_number} iter:{iterr} HEFT:{heft_energy:.2f} delay:{heft_delay:.2f} time:{time.time() - start:.2f}s")
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
        f"{N} {V} {tot} alpha{alpha} max_out{max_out} NWR{NWR} RP{RP_number} iter:{iterr} GAA:{ga_energy:.2f} delay:{ga_delay:.2f} time:{time.time() - start:.2f}s")
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
        f"{N} {V} {tot} alpha{alpha} max_out{max_out} NWR{NWR} RP{RP_number} iter:{iterr} QC-HHO:{hho_energy:.2f} delay:{hho_delay:.2f} time:{time.time() - start:.2f}s")
    results['hho_modified_energy'] = hho_energy / Task.E_max
    results['hho_modified_delay'] = hho_delay

    # Cuckoo scheduler
    start = time.time()
    best_solution, best_energy = cuckoo_search(Task, save_dic=save_dic, msg=msg, iterr=iterr, n=50, max_iter=1000,
                                               ecs_initial=False)
    cuckoo_delay = best_solution.calc_delay(Task)
    print(
        f"{N} {V} {tot} alpha{alpha} max_out{max_out} NWR{NWR} RP{RP_number} iter:{iterr} GACSM:{best_energy:.2f} delay:{cuckoo_delay:.2f} time:{time.time() - start:.2f}s")
    results['cuckoo_energy'] = best_energy / Task.E_max
    results['cuckoo_delay'] = cuckoo_delay

    # QHA scheduler
    start = time.time()
    best_solution, best_energy = quantum_inspired_hyper_heuristics_algorithm(Task, pop_size=50, save_dic=save_dic,
                                                                             msg=msg, iterr=iterr,
                                                                             max_iter=1000, phase_iter=10,
                                                                             ecs_initial=False)
    # best_solution.energy_optimization(Task)
    qha_energy = best_solution.calc_energy(Task)
    qha_delay = best_solution.calc_delay(Task)
    print(
        f"{N} {V} {tot} alpha{alpha} max_out{max_out} NWR{NWR} RP{RP_number} iter:{iterr} QMES:{qha_energy:.2f} delay:{qha_delay:.2f} time:{time.time() - start:.2f}s")
    results['qha_energy'] = qha_energy / Task.E_max
    results['qha_delay'] = qha_delay

    print(f"{N} {V} {tot} alpha{alpha} max_out{max_out} NWR{NWR} RP{RP_number} iter:{iterr} finishes!")
    gc.collect()
    return (N, V, tot, alpha, max_out, NWR, RP_number,iterr), results


if __name__ == "__main__":
    random.seed(2)
    np.random.seed(2)

    # test_list_tot = [(2,5,30),(2, 5, 40), (2, 5, 50), (2, 5, 60), (2, 5, 70), (2, 5, 80), (2, 5, 90), (2, 5, 100),
    #                  (2, 10, 40), (2, 10, 70), (2, 10, 100), (2, 10, 200), (2, 10, 300), (2, 10, 400), (2, 10, 500),
    #                  (2, 20, 100), (2, 20, 200), (2, 20, 300), (2, 20, 400), (2, 20, 500)]
    test_list_tot = [(2, 20, 300),(2, 20, 200),(2, 20, 100)]
    for tt in test_list_tot:
        NWR_list_tot = [0.5, 1.0, 2.0, 3.0, 4.0]
        # NWR_list_tot = [0.5]
        for NWRR in NWR_list_tot:
            alpha_list_tot=[0.5, 1.0, 2.0]
            # alpha_list_tot = [1.0]
            for aa in alpha_list_tot:
                alpha_list =[aa]
                test_list = [tt]
                NWR_list = [NWRR]
                max_out_list = [1,2,4]
                RP_number_list = [2, 4]

                # test_list = [(2, 5, 30)]
                # alpha_list = [0.5]
                # max_out_list = [1]
                # NWR_list = [0.5]
                # RP_number_list = [2]

                tttt = 5
                start_tot = time.time()
                save_dic = f"test4-formal, 2\\{tt}\\{NWRR}"
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
                experiments = [(N, V, tot, alpha, max_out, NWR, RP_number,iterr, save_dic)
                               for N, V, tot in test_list
                               for alpha in alpha_list
                               for max_out in max_out_list
                               for NWR in NWR_list
                               for RP_number in RP_number_list
                               for iterr in range(tttt)]

                # 使用进程池处理所有实验
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                    all_results = pool.map(run_single_experiment, experiments)

                # 创建DataFrame来存储结果
                columns = ["N", "V", "tot", "alpha", "max_out", "NWR", "RP_number", "iterr",
                           "HEFT energy", "HEFT delay",
                           "GAA energy", "GAA delay",
                           "QC-HHO energy", "QC-HHO delay",
                           "GACSM energy", "GACSM delay",
                           "QHA energy", "QHA delay"]

                data = []
                for (N, V, tot, alpha, max_out, NWR, RP_number, iterr), results in all_results:
                    row = [
                        N, V, tot, alpha, max_out, NWR, RP_number, iterr,
                        results['heft_energy'] * 100, results['heft_delay'],
                        results['ga_energy'] * 100, results['ga_delay'],
                        results['hho_modified_energy'] * 100, results['hho_modified_delay'],
                        results['cuckoo_energy'] * 100, results['cuckoo_delay'],
                        results['qha_energy'] * 100, results['qha_delay'],
                    ]
                    data.append(row)

                df = pd.DataFrame(data, columns=columns)
                excel_path = f".\\{save_dic}\\experiment_results_{tt}_{NWRR}_{aa}.xlsx"
                df.to_excel(excel_path, index=False)
                print(f"Results saved to {excel_path}")

                # 恢复标准输出
                sys.stdout = sys.__stdout__

                print(f"Total time cost: {time.time() - start_tot:.2f}s")
                print("finishes!!")
                gc.collect()
            print("finishes!!!")