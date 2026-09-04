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

if __name__== "__main__" :
    random.seed(2)
    np.random.seed(2)
    # 显式初始化类变量
    # TaskConfig.init_class_vars()
    # test_list = [(2,5,30), (2,5,40), (2,5,50), (2,5,60), (2,5,70), (2,5,80), (2,5,90), (2, 5, 100)]
    # test_list = [(2, 10, 500)]
    # test_list = [(2, 5, 100)]
    test_list = [(2,10,100),(2,10,200),(2,10,300),(2,10,400), (2,10,500)]
    # test_list = [(2,10,100),(2,10,200),(2,10,300),(2,10,400), (2,10,500)]
    tttt = 1
    start_tot = time.time()
    save_dic = "test1"
    if not os.path.exists(save_dic):
        os.makedirs(save_dic)
    with (open('my_document.txt', 'w', encoding='utf-8') as f):
        NE_list = [[] for _ in range(4)]
        for N, V, tot in test_list:
            NE = [0] * 4
            for iterr in range(tttt):
                print("--------------------------------------------------------------------")
                print(f"\n{N} {V} {tot} iter:{iterr}")
                random_dict = {"N": N, "V": V, "alpha": 1.0, "max_out": 2, "beta": 1, "Tpr": 100, "Tpr_beta": 20, "wDAG": 400,
                               "gamma_GPU": 2, "gamma_hw": 1.5, "P_sw": 90, "P_npu": 120, "P_npu_beta": 20,
                               "P_static": 24, "P_static_beta": 2, "P_dyn": 60, "P_dyn_beta": 20, "NWR": 1.0, "tot":tot}
                Task=TaskConfig()
                Task.init_class_vars(random_dict, mode="random")
                # TaskConfig.init_class_vars()
                print(Task.E_max)
                # print(TaskConfig.app_list)
                # s1 = MappingSolution(TaskConfig.CPU_number, TaskConfig.NPU_number, TaskConfig.RP_number, TaskConfig.task_number)
                # s1.init_randomly()
                # s1.scheduling()
                # s1.visualize_init()

                # start = time.time()
                # print("\ncuckoo scheduler:")
                # best_solution, best_energy = cuckoo_search(save_dic=save_dic,N=N, V=V, tot=tot, iterr=iterr,n=50, max_iter=1000)
                # print(best_energy)
                # print(best_solution.calc_delay())
                # print(f"time cost: {time.time() - start:.2f}s")
                # NE[0] += best_energy/TaskConfig.E_max


                # best_solution, best_energy = adaptive_genetic_algorithm(
                #     seed=2,
                #     pop_size=50,
                #     max_iter=1000,
                #     crossover_rate=0.8,
                #     mutation_rate=0.1,
                #     plot_number=3
                # )

                start = time.time()
                print("\nGA scheduler:")
                best_solution, best_energy = genetic_algorithm(Task, save_dic=save_dic,msg=f"{N}_{V}_{tot}", iterr=iterr,
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
                NE[1] += best_solution.calc_energy(Task)/Task.E_max

                # best_solution, best_energy = genetic_simulated_annealing(
                #     seed=24,
                #     pop_size=50,
                #     max_iter=500,
                #     crossover_rate=0.8,
                #     mutation_rate=0.1,
                #     plot_number=3
                # )

                # niche_genetic_algorithm(seed=24, pop_size=150, pop_number=3, subpop_min_scale=0.5,
                #                         max_iter=1000, crossover_rate=0.8, mutation_rate=0.1,
                #                         elitism_ratio=0.1, plot_number=3)

                # solution = MappingSolution(
                #     TaskConfig.CPU_number,
                #     TaskConfig.NPU_number,
                #     TaskConfig.RP_number,
                #     TaskConfig.task_number
                # )
                # start = time.time()
                # print("\nheft scheduler:")
                # # heft算法
                # solution.heft_scheduling()
                # solution.visualize_init(window_name=f".\\{save_dic}\\heft scheduler {N}_{V}_{tot} iter{iterr}", is_save=True)
                # print(f"time cost: {time.time() - start:.2f}s")
                # print(solution.calc_energy())
                # print(solution.calc_delay())
                # # solution.energy_optimization()
                # # solution.visualize_init(window_name="heft scheduler optimization", is_save=True)
                # # print(f"time cost: {time.time() - start:.2f}s")
                # NE[2] += solution.calc_energy()/TaskConfig.E_max
                #
                # solution = MappingSolution(
                #     TaskConfig.CPU_number,
                #     TaskConfig.NPU_number,
                #     TaskConfig.RP_number,
                #     TaskConfig.task_number
                # )
                # start = time.time()
                # print("\necs scheduler:")
                # # ecs算法
                # solution.ecs_scheduling()
                # solution.visualize_init(window_name=f".\\{save_dic}\\ecs scheduler {N}_{V}_{tot} iter{iterr}", is_save=True)
                # print(f"time cost: {time.time() - start:.2f}s")
                # print(solution.calc_energy())
                # print(solution.calc_delay())
                # NE[3] += solution.calc_energy()/TaskConfig.E_max

                start = time.time()
                # solution = MappingSolution(
                #     TaskConfig.CPU_number,
                #     TaskConfig.NPU_number,
                #     TaskConfig.RP_number,
                #     TaskConfig.task_number
                # )
                # solution.init_randomly()
                # solution.scheduling()
                # print(solution.calc_energy())
                # print(solution.calc_delay())
                # print(f"time cost: {time.time() - start:.2f}s")
                # solution.visualize_init(window_name="initial randomly",is_save=True)

                # best_solution, history = simulated_annealing(solution, N=N, V=V, tot=tot, iterr=iterr, alpha=0.995, max_iter=1000, steps_per_temp=10, p0=0.8,
                #                                              is_show_best=False, is_plot_history=False, is_save_best=True)
                # print(best_solution.calc_energy())
                # print(best_solution.calc_delay())
                # print(f"time cost: {time.time() - start:.2f}s")
                # best_solution.energy_optimization()
                # print(best_solution.calc_energy())
                # print(best_solution.calc_delay())
                # best_solution.visualize_init(window_name=f".\\test\\SA scheduler optimization {N}_{V}_{tot} iter{iterr}", is_save=True)
                # NE += best_solution.calc_energy()/TaskConfig.E_max
                # print(f"NE={best_solution.calc_energy()/TaskConfig.E_max}")
            NE = [ne/tttt for ne in NE]
            for idx, ne in enumerate(NE):
                NE_list[idx].append(ne)
            print(f"average NE:{NE}")
        x_values = [x for _,_,x in test_list]
        plt.figure()
        plt.plot(x_values, NE_list[0], label="GACSM")
        plt.plot(NE_list[1], label="HESAL")
        plt.plot(NE_list[2], label="HEFT")
        plt.plot(NE_list[3], label="ECS")
        plt.xlabel('Iteration')
        plt.ylabel('NE')
        plt.title('Energy Trends')
        plt.legend()
        plt.grid(True)
        plt.savefig("result.png")
        print(f"time cost: {time.time() - start_tot:.2f}s")