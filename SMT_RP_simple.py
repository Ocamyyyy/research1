from z3 import *
from task_definition import TaskConfig
import time
import threading
import random
from time_slot_plot import ProcessorTimelinePlotter, colors


if __name__== "__main__" :
    # 显式初始化类变量
    TaskConfig.init_class_vars()

    # 创建任务映射变量字典
    # 应用i的第q个实例的第j个任务是否映射到第k个CPU
    X = {(i, q, j, k): Bool(f"X_{i}_{q}_{j}_{k}")
         for i in range(len(TaskConfig.app_list))
         for q in range(TaskConfig.instance_number_list[i])
         for j in range(TaskConfig.app_list[i].node_num)
         for k in range(TaskConfig.CPU_number)}
    # 应用i的第q个实例的第j个任务是否映射到第r个NPU
    Y = {(i, q, j, r): Bool(f"Y_{i}_{q}_{j}_{r}")
         for i in range(len(TaskConfig.app_list))
         for q in range(TaskConfig.instance_number_list[i])
         for j in range(TaskConfig.app_list[i].node_num)
         for r in range(TaskConfig.NPU_number)}
    # 应用i的第q个实例的第j个任务是否映射到第eta个RP
    Z = {(i, q, j, eta): Bool(f"Z_{i}_{q}_{j}_{eta}")
         for i in range(len(TaskConfig.app_list))
         for q in range(TaskConfig.instance_number_list[i])
         for j in range(TaskConfig.app_list[i].node_num)
         for eta in range(TaskConfig.RP_number)}
    # 应用i的第q个实例的第j个任务的开始时间
    t = {(i, q, j): Int(f"t_{i}_{q}_{j}")
         for i in range(len(TaskConfig.app_list))
         for q in range(TaskConfig.instance_number_list[i])
         for j in range(TaskConfig.app_list[i].node_num)}
    # 应用i的第q个实例的第j个任务的重构开始时间（不映射到FPGA则为0）
    tpr = {(i, q, j): Int(f"tpr_{i}_{q}_{j}")
         for i in range(len(TaskConfig.app_list))
         for q in range(TaskConfig.instance_number_list[i])
         for j in range(TaskConfig.app_list[i].node_num)}
    # 辅助变量A1：判断任务T(i,q,j)是否在任务T(i2,q2,j2)开始执行之后开始执行
    A1 = {(i, q, j, i2, q2, j2): Bool(f"A1_{i}_{q}_{j}_{i2}_{q2}_{j2}")
          for i in range(len(TaskConfig.app_list))
          for q in range(TaskConfig.instance_number_list[i])
          for j in range(TaskConfig.app_list[i].node_num)
          for i2 in range(len(TaskConfig.app_list))
          for q2 in range(TaskConfig.instance_number_list[i2])
          for j2 in range(TaskConfig.app_list[i2].node_num)}
    # 辅助变量RPMap：判断两个任务是否都映射到某个RP区
    RPMap = {(i, q, j, i2, q2, j2, eta): Bool(f"RPMap_{i}_{q}_{j}_{i2}_{q2}_{j2}_{eta}")
             for i in range(len(TaskConfig.app_list))
             for q in range(TaskConfig.instance_number_list[i])
             for j in range(TaskConfig.app_list[i].node_num)
             for i2 in range(len(TaskConfig.app_list))
             for q2 in range(TaskConfig.instance_number_list[i2])
             for j2 in range(TaskConfig.app_list[i2].node_num)
             for eta in range(TaskConfig.RP_number)}
    # 辅助变量A3：判断任务T(i,q,j)是否在任务T(i2,q2,j2)开始重构之后开始重构
    A3 = {(i, q, j, i2, q2, j2): Bool(f"A3_{i}_{q}_{j}_{i2}_{q2}_{j2}")
          for i in range(len(TaskConfig.app_list))
          for q in range(TaskConfig.instance_number_list[i])
          for j in range(TaskConfig.app_list[i].node_num)
          for i2 in range(len(TaskConfig.app_list))
          for q2 in range(TaskConfig.instance_number_list[i2])
          for j2 in range(TaskConfig.app_list[i2].node_num)}

    # 定义大M常数
    M1 = RealVal(10000)
    M2 = RealVal(10000)
    M3 = RealVal(10000)
    # M4 = RealVal(10000)
    M5 = RealVal(10000)

    opt = Optimize()

    # 辅助变量A1约束
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                for i2 in range(len(TaskConfig.app_list)):
                    for q2 in range(TaskConfig.instance_number_list[i2]):
                        for j2 in range(TaskConfig.app_list[i2].node_num):
                            a1 = A1[(i, q, j, i2, q2, j2)]
                            t1 = t[(i, q, j)]
                            t2 = t[(i2, q2, j2)]
                            # 第一个约束：t1 - t2 ≤ M2·A1
                            opt.add(t1 - t2 <= M2 * If(a1, 1, 0))
                            # 第二个约束：t1 - t2 > -M2·(1 - A1)
                            # 注意：Z3中严格大于要用Not(<=)表示
                            opt.add(Not(t1 - t2 <= -M2 * (1 - If(a1, 1, 0))))
    # 辅助变量RPMap约束
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                for i2 in range(len(TaskConfig.app_list)):
                    for q2 in range(TaskConfig.instance_number_list[i2]):
                        for j2 in range(TaskConfig.app_list[i2].node_num):
                            for eta in range(TaskConfig.RP_number):
                                # 实现RPMap = Z ∧ Z'
                                opt.add(
                                    RPMap[(i, q, j, i2, q2, j2, eta)] == And(
                                        Z[(i, q, j, eta)],
                                        Z[(i2, q2, j2, eta)]
                                    )
                                )
    # 辅助变量A3约束
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                for i2 in range(len(TaskConfig.app_list)):
                    for q2 in range(TaskConfig.instance_number_list[i2]):
                        for j2 in range(TaskConfig.app_list[i2].node_num):
                            a1 = A3[(i, q, j, i2, q2, j2)]
                            t1 = tpr[(i, q, j)]
                            t2 = tpr[(i2, q2, j2)]
                            # 第一个约束：t1 - t2 ≤ M2·A1
                            opt.add(t1 - t2 <= M3 * If(a1, 1, 0))
                            # 第二个约束：t1 - t2 > -M2·(1 - A1)
                            # 注意：Z3中严格大于要用Not(<=)表示
                            opt.add(Not(t1 - t2 <= -M3 * (1 - If(a1, 1, 0))))
    ## 首先预计算所有可能的三元组(i3,q3,j3)
    ALL_TASKS = [
        (i3, q3, j3)
        for i3 in range(len(TaskConfig.app_list))
        for q3 in range(TaskConfig.instance_number_list[i3])
        for j3 in range(TaskConfig.app_list[i3].node_num)
    ]
    ## 引入辅助变量存储中间结果(任意两个RP区的任务之间有别的任务在执行)
    Indicator = {(i, q, j, i2, q2, j2, eta): Bool(f"Ind_{i}_{q}_{j}_{i2}_{q2}_{j2}_{eta}")
                 for i in range(len(TaskConfig.app_list))
                 for q in range(TaskConfig.instance_number_list[i])
                 for j in range(TaskConfig.app_list[i].node_num)
                 for i2 in range(len(TaskConfig.app_list))
                 for q2 in range(TaskConfig.instance_number_list[i2])
                 for j2 in range(TaskConfig.app_list[i2].node_num)
                 for eta in range(TaskConfig.RP_number) if not (i2 == i and j2 == j and q2 == q)}
    ## 预先计算所有indicator
    for i, q, j, i2, q2, j2, eta in Indicator:
        if i == i2 and j == j2 and q == q2: continue
        # 修正后的sum计算，使用预计算的ALL_TASKS
        total = Sum([If(And(
            RPMap.get((i, q, j, i3, q3, j3, eta), False),
            RPMap.get((i2, q2, j2, i3, q3, j3, eta), False),
            A3.get((i, q, j, i3, q3, j3), False),
            A3.get((i3, q3, j3, i2, q2, j2), False)
        ), 1, 0) for i3, q3, j3 in ALL_TASKS
            if not ((i3 == i and j3 == j and q3 == q) or (i3 == i2 and j3 == j2 and q3 == q2))])

        opt.add(Indicator[(i, q, j, i2, q2, j2, eta)] == (total > 0))

    # 没有映射到fpga的任务其重构开始时间为0
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                sum_Z = Sum([If(Z[(i, q, j, eta)], 1, 0) for eta in range(TaskConfig.RP_number)])
                # 添加约束
                opt.add(tpr[(i,q,j)] <= M1 * sum_Z)
    # 开始执行时间和开始重构时间都要大于等于0
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                # 添加约束
                opt.add(tpr[(i, q, j)] >= 0)
                opt.add(t[(i, q, j)] >= 0)
    # 任务约束：所有任务只执行一次，只映射到一个计算单元
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                # 创建三个求和部分
                sum_X = Sum([If(X[(i, q, j, k)], 1, 0) for k in range(TaskConfig.CPU_number)])
                sum_Y = Sum([If(Y[(i, q, j, r)], 1, 0) for r in range(TaskConfig.NPU_number)])
                sum_Z = Sum([If(Z[(i, q, j, eta)], 1, 0) for eta in range(TaskConfig.RP_number)])

                # 添加约束
                opt.add(sum_X + sum_Y + sum_Z == 1)
    # 任务约束：任务实例在应用实例到达后开始执行
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                arr = TaskConfig.instance_dict_list[i][q + 1]["arr"]
                opt.add(t[(i, q, j)] >= arr)
    # 任务约束：同一应用实例的任务存在依赖关系
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                successors = TaskConfig.app_list[i].successor_of_node(j)
                if successors:
                    for successor in successors:
                        sum_X = Sum([If(X[(i, q, j, k)], 1, 0) for k in range(TaskConfig.CPU_number)])
                        sum_Y = Sum([If(Y[(i, q, j, r)], 1, 0) for r in range(TaskConfig.NPU_number)])
                        sum_Z = Sum([If(Z[(i, q, j, eta)], 1, 0) for eta in range(TaskConfig.RP_number)])
                        cpu_time = TaskConfig.app_list[i].sw_exe[j]
                        npu_time = TaskConfig.app_list[i].npu_exe[j]
                        fpga_time = TaskConfig.app_list[i].hw_exe[j]
                        opt.add(t[(i, q, j)] + cpu_time * sum_X + npu_time * sum_Y + fpga_time * sum_Z <= t[(i, q, successor)])
    # 任务约束：截止时间约束
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                if TaskConfig.app_list[i].out_degree[j] == 0:
                    ddl = TaskConfig.instance_dict_list[i][q + 1]["ddl"]
                    sum_X = Sum([If(X[(i, q, j, k)], 1, 0) for k in range(TaskConfig.CPU_number)])
                    sum_Y = Sum([If(Y[(i, q, j, r)], 1, 0) for r in range(TaskConfig.NPU_number)])
                    sum_Z = Sum([If(Z[(i, q, j, eta)], 1, 0) for eta in range(TaskConfig.RP_number)])
                    cpu_time = TaskConfig.app_list[i].sw_exe[j]
                    npu_time = TaskConfig.app_list[i].npu_exe[j]
                    fpga_time = TaskConfig.app_list[i].hw_exe[j]
                    opt.add(t[(i, q, j)] + cpu_time * sum_X + npu_time * sum_Y + fpga_time * sum_Z <= ddl)
    # 资源约束：一个CPU每一时刻只能执行一个任务
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                for i2 in range(len(TaskConfig.app_list)):
                    for q2 in range(TaskConfig.instance_number_list[i2]):
                        for j2 in range(TaskConfig.app_list[i2].node_num):
                            for k in range(TaskConfig.CPU_number):
                                if (i, q, j) == (i2, q2, j2):  # 跳过同一任务
                                    continue
                                a = A1[(i, q, j, i2, q2, j2)]
                                sw_time = TaskConfig.app_list[i2].sw_exe[j2]
                                condition = And(a, X[(i, q, j, k)], X[(i2, q2, j2, k)])
                                opt.add(Implies(condition, t[(i, q, j)] >= t[(i2, q2, j2)] + sw_time))
                                sw_time = TaskConfig.app_list[i].sw_exe[j]
                                condition = And(Not(a), X[(i, q, j, k)], X[(i2, q2, j2, k)])
                                opt.add(Implies(condition, t[(i2, q2, j2)] >= t[(i, q, j)] + sw_time))
    # 资源约束：一个NPU每一时刻只能执行一个任务
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                for i2 in range(len(TaskConfig.app_list)):
                    for q2 in range(TaskConfig.instance_number_list[i2]):
                        for j2 in range(TaskConfig.app_list[i2].node_num):
                            for r in range(TaskConfig.NPU_number):
                                if (i, q, j) == (i2, q2, j2):  # 跳过同一任务
                                    continue
                                a = A1[(i, q, j, i2, q2, j2)]
                                npu_time = TaskConfig.app_list[i2].npu_exe[j2]
                                condition = And(a, Y[(i, q, j, r)], Y[(i2, q2, j2, r)])
                                opt.add(Implies(condition, t[(i, q, j)] >= t[(i2, q2, j2)] + npu_time))
                                npu_time = TaskConfig.app_list[i].npu_exe[j]
                                condition = And(Not(a), Y[(i, q, j, r)], Y[(i2, q2, j2, r)])
                                opt.add(Implies(condition, t[(i2, q2, j2)] >= t[(i, q, j)] + npu_time))
    # 资源约束：映射到FPGA的任务在重构完成后开始执行
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                for eta in range(TaskConfig.RP_number):
                    # 获取任务处理时间基值
                    base_processing = TaskConfig.app_list[i].tpr[j]
                    # 情况1：无冲突时使用完整处理时间
                    opt.add(Implies(Z[(i, q, j, eta)], t[(i, q, j)] >= tpr[(i, q, j)] + base_processing))
    # 资源约束：FPGA同一时刻只能完成一个可重构区的重构配置
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                for i2 in range(len(TaskConfig.app_list)):
                    for q2 in range(TaskConfig.instance_number_list[i2]):
                        for j2 in range(TaskConfig.app_list[i2].node_num):
                            # 跳过相同任务实例
                            if i == i2 and q == q2 and j == j2:
                                continue

                            # 获取相关变量
                            a = A3[(i, q, j, i2, q2, j2)]
                            tpr1 = tpr[(i, q, j)]
                            tpr2 = tpr[(i2, q2, j2)]
                            pr_time1 = TaskConfig.app_list[i].tpr[j]
                            pr_time2 = TaskConfig.app_list[i2].tpr[j2]
                            sum_Z1 = If(Sum([If(Z[(i, q, j, eta)], 1, 0)
                                             for eta in range(TaskConfig.RP_number)]) == 1, True, False)
                            sum_Z2 = If(Sum([If(Z[(i2, q2, j2, eta)], 1, 0)
                                             for eta in range(TaskConfig.RP_number)]) == 1, True, False)

                            opt.add(Implies(And(a,  sum_Z1, sum_Z2), tpr2 + pr_time2<= tpr1))
                            opt.add(Implies(And(Not(a), sum_Z1, sum_Z2), tpr1 + pr_time1<= tpr2))
    # 资源约束：同一重构区，后执行任务重构开始时间大于等于先执行任务的结束时间
    for i in range(len(TaskConfig.app_list)):
        for q in range(TaskConfig.instance_number_list[i]):
            for j in range(TaskConfig.app_list[i].node_num):
                for i2 in range(len(TaskConfig.app_list)):
                    for q2 in range(TaskConfig.instance_number_list[i2]):
                        for j2 in range(TaskConfig.app_list[i2].node_num):
                            if (i, q, j) == (i2, q2, j2):  # 跳过同一任务
                                continue
                            for eta in range(TaskConfig.RP_number):
                                a = A1[(i, q, j, i2, q2, j2)]
                                hw_time = TaskConfig.app_list[i2].hw_exe[j2]
                                condition = And(a, Z[(i, q, j, eta)], Z[(i2, q2, j2, eta)])
                                opt.add(Implies(condition, tpr[(i, q, j)] >= t[(i2, q2, j2)] + hw_time))
                                hw_time = TaskConfig.app_list[i].hw_exe[j]
                                condition = And(Not(a), Z[(i, q, j, eta)], Z[(i2, q2, j2, eta)])
                                opt.add(Implies(condition, tpr[(i2, q2, j2)] >= t[(i, q, j)] + hw_time))

    # 定义目标函数：最小化能耗
    energy_CPU = Int("energy_CPU")
    opt.add(energy_CPU == Sum([If(X[(i,q,j,k)], 1, 0) * TaskConfig.app_list[i].sw_exe[j] * TaskConfig.app_list[i].P_sw[j]
               for i in range(len(TaskConfig.app_list))
               for q in range(TaskConfig.instance_number_list[i])
               for j in range(TaskConfig.app_list[i].node_num)
               for k in range(TaskConfig.CPU_number)]))
    energy_NPU = Int("energy_NPU")
    opt.add(
        energy_NPU == Sum([If(Y[(i, q, j, r)], 1, 0) * TaskConfig.app_list[i].npu_exe[j] * TaskConfig.app_list[i].P_npu[j]
                           for i in range(len(TaskConfig.app_list))
                           for q in range(TaskConfig.instance_number_list[i])
                           for j in range(TaskConfig.app_list[i].node_num)
                           for r in range(TaskConfig.NPU_number)]))
    energy_RP_dyn = Int("energy_RP_dyn")
    opt.add(
        energy_RP_dyn == Sum(
            [If(Z[(i, q, j, eta)], 1, 0) * TaskConfig.app_list[i].hw_exe[j] * TaskConfig.app_list[i].P_dyn[j]
             for i in range(len(TaskConfig.app_list))
             for q in range(TaskConfig.instance_number_list[i])
             for j in range(TaskConfig.app_list[i].node_num)
             for eta in range(TaskConfig.RP_number)]))
    energy_RP_sta = Int("energy_RP_sta")
    energy_RP_sta_terms = []
    for (i, q, j, i2, q2, j2, eta) in RPMap:
        if (i, q, j) == (i2, q2, j2): continue
        # 基础条件
        condition = And(
            RPMap[(i, q, j, i2, q2, j2, eta)],
            A3[(i, q, j, i2, q2, j2)],
            Not(Indicator[(i, q, j, i2, q2, j2, eta)])
        )
        P_sta = int(TaskConfig.app_list[i2].P_static[j2])
        term = If(condition, (tpr[(i, q, j)] - tpr[(i2, q2, j2)]) * P_sta,0)
        energy_RP_sta_terms.append(term)
    for i,q,j,eta in Z:
        sum_after = Sum(If(And(RPMap[(i,q,j,i2,q2,j2,eta)], A1[(i2,q2,j2,i,q,j)]),1,0) for i2, q2, j2 in ALL_TASKS
            if not (i2 == i and j2 == j and q2 == q))
        is_end = If(sum_after > 0, False, True)
        P_sta = int(TaskConfig.app_list[i].P_static[j])
        term = If(And(Z[(i,q,j,eta)],is_end),(TaskConfig.H-tpr[(i,q,j)])* P_sta,0)
        energy_RP_sta_terms.append(term)
    opt.add(energy_RP_sta == Sum(energy_RP_sta_terms))
    energy = Int("energy")
    opt.add(energy == energy_CPU + energy_NPU + energy_RP_dyn + energy_RP_sta)
    # opt.add(energy == energy_CPU + energy_NPU + energy_RP_dyn)
    opt.minimize(energy)

    # 4. 创建计时线程
    def print_time():
        start = time.time()
        while not solver_finished[0]:
            elapsed = time.time() - start
            print(f"\r已求解时间: {elapsed:.2f}秒", end="", flush=True)
            time.sleep(1)
        print()  # 换行


    solver_finished = [False]
    timer_thread = threading.Thread(target=print_time)

    # 5. 开始求解
    print("开始求解...")
    timer_thread.start()

    try:
        result = opt.check()
        solver_finished[0] = True
        timer_thread.join()

        # 6. 输出结果
        print("\n=== 求解结果 ===")
        if result == sat:
            print("找到可行解")
            model = opt.model()

            # 简洁输出关键信息
            print("\n=== 资源分配方案 ===")

            # 1. 输出所有激活的映射关系
            print("\n[激活的资源映射]")
            for i, q, j, k in X:
                if model.eval(X[i, q, j, k]):
                    print(f"应用{i}实例{q}任务{j} → CPU{k}")

            for i, q, j, k in Y:
                if model.eval(Y[i, q, j, k]):
                    print(f"应用{i}实例{q}任务{j} → NPU{k}")

            for i, q, j, k in Z:
                if model.eval(Z[i, q, j, k]):
                    print(f"应用{i}实例{q}任务{j} → RP{k}")

            # 2. 输出关键时间信息
            print("\n[关键时间节点]")
            for (i, q, j), var in t.items():
                print(f"应用{i}实例{q}任务{j} 开始执行时间: {model.eval(var)}")
                print(f"应用{i}实例{q}任务{j} 开始重构时间: {model.eval(tpr[(i, q, j)])}")

            # 随机生成每个应用的颜色
            app_color_list = random.sample(colors, len(TaskConfig.app_list))
            # print(f"application colors:{app_color_list}")
            subtask_colors = [0] * TaskConfig.task_number
            position = 0
            subtask_labels = [""] * TaskConfig.task_number
            subtask_RP_info = [[0, 0, 0] for _ in range(TaskConfig.task_number)]
            task_mapping = [0] * TaskConfig.task_number
            task_start_time = [0] * TaskConfig.task_number
            task_end_time = [0] * TaskConfig.task_number
            for i in range(len(TaskConfig.app_list)):
                for q in range(TaskConfig.instance_number_list[i]):
                    for j in range(TaskConfig.app_list[i].node_num):
                        processor_condition = []
                        for k in range(TaskConfig.CPU_number):
                            # 获取X[(i,q,j,k)]在模型中的实际取值（True/False）
                            x_val = model.eval(X[(i, q, j, k)])
                            # 根据取值向列表添加1或0
                            processor_condition.append(1 if is_true(x_val) else 0)
                        for r in range(TaskConfig.NPU_number):
                            # 获取X[(i,q,j,k)]在模型中的实际取值（True/False）
                            y_val = model.eval(Y[(i, q, j, r)])
                            # 根据取值向列表添加1或0
                            processor_condition.append(1 if is_true(y_val) else 0)
                        for eta in range(TaskConfig.RP_number):
                            # 获取X[(i,q,j,k)]在模型中的实际取值（True/False）
                            z_val = model.eval(Z[(i, q, j, eta)])
                            # 根据取值向列表添加1或0
                            processor_condition.append(1 if is_true(z_val) else 0)
                        # print(i, q, j, processor_condition, sum_SIMap)
                        if processor_condition.index(1) >= TaskConfig.CPU_number + TaskConfig.NPU_number:
                            # print("yes")
                            subtask_RP_info[position][0] = 1
                            subtask_RP_info[position][1] = model.eval(tpr[(i, q, j)]).as_long()
                            subtask_RP_info[position][2] = model.eval(tpr[(i, q, j)]).as_long() + TaskConfig.app_list[i].tpr[j]
                        subtask_colors[position] = app_color_list[i]
                        subtask_labels[position] = f"({i},{q + 1},{j})"
                        task_start_time[position] = model.eval(t[(i, q, j)]).as_long()
                        processor = processor_condition.index(1)
                        task_mapping[position] = processor
                        if processor < TaskConfig.CPU_number:
                            exe = TaskConfig.app_list[i].sw_exe[j]
                        elif processor < TaskConfig.CPU_number + TaskConfig.NPU_number:
                            exe = TaskConfig.app_list[i].npu_exe[j]
                        else:
                            exe = TaskConfig.app_list[i].hw_exe[j]
                        task_end_time[position] = task_start_time[position] + exe
                        position += 1
            plotter = ProcessorTimelinePlotter(TaskConfig.processor_number, TaskConfig.CPU_number, TaskConfig.NPU_number,
                                               TaskConfig.RP_number, time_axis_length=TaskConfig.H)
            task_blocks = [
                {'processor': task_mapping[i], 'start': task_start_time[i], 'end': task_end_time[i],
                 'color': subtask_colors[i],
                 'label': subtask_labels[i], 'RP': subtask_RP_info[i]} for i in range(TaskConfig.task_number)
            ]
            print(f"task blocks:{task_blocks}")
            plotter.set("SMT scheduling4", task_blocks=task_blocks, is_block=True)
            print(f"能耗：{model.eval(energy).as_long()}")
            print(f"FPGA静态功耗：{model.eval(energy_RP_sta).as_long()}")

        elif result == unsat:
            print("无解")
        else:
            # m = opt.model()
            # if m:
            #     print("=== 部分模型 ===")
            #     for var in m:
            #         print(f"{var.name()} = {m[var]}")
            # else:
            #     print("无法获取部分模型")
            # print("求解完成但状态未知")
            solver_finished[0] = True  # 显式设置结束标志
            timer_thread.join()  # 确保计时线程结束
            m = opt.model()
            if m:
                print("=== 部分模型 ===", flush=True)  # 强制刷新输出
                for var in m:
                    print(f"{var.name()} = {m[var]}", flush=True)
            else:
                print("无法获取部分模型", flush=True)
            print("求解完成但状态未知", flush=True)
            print(f"求解器状态: {opt.reason_unknown()}", flush=True)

    except KeyboardInterrupt:
        solver_finished[0] = True
        timer_thread.join()
        print("\n用户中断求解过程")
