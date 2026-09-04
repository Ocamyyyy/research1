from time_slot_plot import ProcessorTimelinePlotter, colors
from DAGs import DAG
import math
from functools import reduce
import random
from collections import deque

CPU_number = 4
NPU_number = 1
RP_number = 4

app_list = []
instance_number_list = []
instance_dict_list = {}
H = 0

# 计算超周期
def calc_superperiod(application_list:list[DAG]):
    periods = [application.period for application in application_list]
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    return reduce(lcm, periods)

# 计算子任务实例在决策变量中的位置(instance_index从1开始计数方式)
def calc_subtask_instance_position(app_index,instance_index,subtask_index):
    pos = 0
    for pre_app_index in range(app_index):
        pos += instance_number_list[pre_app_index] * app_list[pre_app_index].node_num
    pos += (instance_index-1) * app_list[app_index].node_num
    pos += subtask_index
    return pos

# 记录决策变量位置对应的子任务实例
def calc_subtask_instance_from_position(pos):
    i=0
    q=0
    pos_total = 0
    for app_index in range(len(app_list)):
        if pos_total + instance_number_list[app_index] * app_list[app_index].node_num -1 < pos:
            pos_total += instance_number_list[app_index] * app_list[app_index].node_num
        else:
            i = app_index
            break
    for instance_index in range(instance_number_list[i]):
        if pos_total + app_list[i].node_num -1 <pos:
            pos_total += app_list[i].node_num
        else:
            q = instance_index + 1
            break
    j = pos - pos_total
    return i,q,j


# 判断任务是否重构
def define_RP(task:int, processor, task_before):
    i, q, j = calc_subtask_instance_from_position(task)
    if processor >= CPU_number + NPU_number:  # 如果映射到fpga
        # 判断是否需要重构
        if task_before:  # 先前没有任务执行，必定重构
            i_prev, q_prev, j_prev = calc_subtask_instance_from_position(task_before[-1])
            if i_prev == i and j_prev == j:
                return False
        return True
    return False

# 计算任务的最早开始执行时间
# 输入：任务编号,任务结束时间（list），任务映射的处理器,任务映射的处理器的空余时间，任务映射的处理器已执行的任务，重构配置接口的空余时间
def calc_EST(task:int, end_time:list, processor, avail:int, task_before, avail_RP):
    i, q, j = calc_subtask_instance_from_position(task)
    predecessor_of_task = app_list[i].predecessor_of_node(j)
    predecessor_FT = []
    if predecessor_of_task:
        predecessor_FT = [end_time[calc_subtask_instance_position(i, q, index)] for index in predecessor_of_task]
    # 判断是否需要重构
    RP_flag = define_RP(task, processor, task_before)
    if RP_flag:
        if predecessor_FT:
            return max(max(avail,avail_RP)+app_list[i].tpr[j],instance_dict_list[i][q]["arr"], max(predecessor_FT)),True
        return max(max(avail,avail_RP)+app_list[i].tpr[j],instance_dict_list[i][q]["arr"]),True
    if predecessor_FT:
        return max(avail,instance_dict_list[i][q]["arr"], max(predecessor_FT)),False
    return max(avail,instance_dict_list[i][q]["arr"]),False

# 计算任务的最迟开始执行时间
# 输入：任务编号，所有任务映射的处理器，截止时间，任务是否重构
def calc_LST(task, processors,ddl):
    i,q,j = calc_subtask_instance_from_position(task)
    ou_degree = app_list[i].out_degree.copy()
    nodes = list(range(app_list[i].node_num))

    # 初始化最早开始时间
    latest_time = [ddl] * app_list[i].node_num

    # 找到所有入度为0的节点（起始节点）
    queue = deque([node for node in nodes if ou_degree[node] == 0])
    for node in queue:
        pos = calc_subtask_instance_position(i,q,node)
        if processors[pos] < CPU_number:
            latest_time[node] = ddl- app_list[i].sw_exe[node]
        elif processors[pos] < CPU_number+NPU_number:
            latest_time[node] = ddl- app_list[i].npu_exe[node]
        else:
            latest_time[node] = ddl- app_list[i].hw_exe[node] - app_list[i].tpr[node]

    # 拓扑排序
    while queue:
        u = queue.popleft()
        u_pos =calc_subtask_instance_position(i,q,u)
        if u_pos == task:
            return latest_time[u]
        u_time = latest_time[u]
        if processors[u_pos] >= CPU_number+NPU_number:
            u_time -= app_list[i].tpr[u]
        # 遍历u的所有邻接节点
        for v in app_list[i].reverse_graph[u]:
            # 更新v的最早开始时间
            pos = calc_subtask_instance_position(i, q, v)
            if processors[pos] < CPU_number:
                latest_time[v] = min(latest_time[v],u_time - app_list[i].sw_exe[v])
            elif processors[pos] < CPU_number + NPU_number:
                latest_time[v] = min(latest_time[v],u_time - app_list[i].npu_exe[v])
            else:
                latest_time[v] = min(latest_time[v],u_time - app_list[i].hw_exe[v])
            # 减少v的入度
            ou_degree[v] -= 1
            # 如果v的入度为0，加入队列
            if ou_degree[v] == 0:
                queue.append(v)
    # 如果目标节点不在图中（理论上不会，因为输入合法）
    raise ValueError(f"Task {task} not found in the graph.")

# （若任务后续节点时间不动）计算任务的实际最晚执行结束时间
def calc_ALFT(task, processor_completed_task_list, task_start_time, task_DPR_start_time=None, task_DPR=None, mode_DPR:bool = False):
    succ = list_successors(task)
    if succ:
        min_succ = min([task_start_time[mm] for mm in succ])
    else:
        min_succ = instance_dict_list[i][q]["ddl"]
    next_task = calc_next_task_on_processor(task, processor_completed_task_list)
    if next_task:
        if mode_DPR:
            if task_DPR[next_task]:
                next_task_start_time = task_DPR_start_time[next_task]
            else:
                next_task_start_time = task_start_time[next_task]
        else:
            next_task_start_time = task_start_time[next_task]
        LFT = min(min_succ, next_task_start_time)
    else:
        LFT = min_succ
    return LFT

# 计算固定区间内可行的重构区间
# 输入参数：intervals当前重构占用，c可行重构开始时间，d可行重构结束时间，e重构所需时间
def calc_last_RP_interval(intervals, c, d, e):
    prev_end = d
    n = len(intervals)

    for i in range(n - 1, -1, -1):
        a, b = intervals[i]
        if b < prev_end:
            gap_start = b
            gap_end = prev_end
            gap_length = gap_end - gap_start
            if gap_length >= e:
                start = gap_end - e
                return [start, gap_end]
        prev_end = min(prev_end, a)

    if prev_end > c:
        gap_length = prev_end - c
        if gap_length >= e:
            start = prev_end - e
            return [start, prev_end]

    return None

# 往原来的重构队列插入新的重构区间
def insert_RP_interval(intervals, new_RP_interval, task_list, task):
    c, d = new_RP_interval
    i = 0
    n = len(intervals)

    # 找到插入位置
    while i < n and intervals[i][1] <= c:
        i += 1

    # 插入合并后的区间
    intervals.insert(i, [c, d])
    task_list.insert(i, task)

# 返回某个任务同一处理器上的下一个执行的任务
# 输入：任务，任务所在处理器执行的任务list
def calc_next_task_on_processor(task, task_list):
    if not isinstance(task_list, list):
        task_list = [task_list]
    if task!=task_list[-1]:
        task_index = task_list.index(task)
        return task_list[task_index + 1]
    else:
        return None

def calc_CPU_dynamic_energy(task_start_time, task_end_time, task_list):
    sum = 0
    if task_list:
        for task in task_list:
            i,q,j = calc_subtask_instance_from_position(task)
            sum += (task_end_time[task]-task_start_time[task])*app_list[i].P_sw[j]
    return sum

def calc_NPU_dynamic_energy(task_start_time, task_end_time, task_list):
    sum = 0
    if task_list:
        for task in task_list:
            i,q,j = calc_subtask_instance_from_position(task)
            sum += (task_end_time[task]-task_start_time[task])*app_list[i].P_npu[j]
    return sum

def calc_RP_dynamic_energy(task_start_time, task_end_time, task_list, RP_start_time, RP_list):
    sum = 0
    if task_list:
        for index,task in enumerate(task_list):
            i, q, j = calc_subtask_instance_from_position(task)
            e_dyn = (task_end_time[task]-task_start_time[task])*app_list[i].P_dyn[j]
            if index < len(task_list) - 1:
                if task_list[index + 1] in RP_list and task in RP_list:
                    e_sta = (RP_start_time[task_list[index + 1]] - RP_start_time[task]) * app_list[i].P_static[j]
                elif task in RP_list:
                    e_sta = (task_start_time[task_list[index + 1]] - RP_start_time[task]) * app_list[i].P_static[j]
                elif task_list[index + 1] in RP_list:
                    e_sta = (RP_start_time[task_list[index + 1]] - task_start_time[task]) * app_list[i].P_static[j]
                else:
                    e_sta = (task_start_time[task_list[index + 1]] - task_start_time[task]) * app_list[i].P_static[j]
            else:
                if task in RP_list:
                    e_sta = (H - RP_start_time[task] + RP_start_time[0]) * app_list[i].P_static[j]
                else:
                    e_sta = (H - task_start_time[task] + RP_start_time[0]) * app_list[i].P_static[j]
            sum = sum + e_dyn + e_sta
    return sum

# 计算调度方案的能耗
def calc_energy(task_start_time, task_end_time, processor_task, RP_start_time, processor_RP):
    sum = 0
    for processor in range(processor_number):
        if processor < CPU_number:
            print(processor_task[processor])
            sum += calc_CPU_dynamic_energy(task_start_time, task_end_time, processor_task[processor])
        elif processor < CPU_number + NPU_number:
            sum += calc_NPU_dynamic_energy(task_start_time, task_end_time, processor_task[processor])
        else:
            sum += calc_RP_dynamic_energy(task_start_time, task_end_time, processor_task[processor], RP_start_time, processor_RP)
    return sum

# 返回某个任务实例的后续任务节点list
def list_successors(task):
    i,q,j=calc_subtask_instance_from_position(task)
    successors=app_list[i].successor_of_node(j)
    return [calc_subtask_instance_position(i,q,successor) for successor in successors]

# 返回受某个任务实例开始执行时间的影响的后续所有任务节点list(包括该任务自身)
def list_affected_tasks(task, reverse=False):
    i,q,j=calc_subtask_instance_from_position(task)
    successors=app_list[i].find_affected_tasks(j,reverse=reverse)
    return [calc_subtask_instance_position(i,q,successor) for successor in successors]

def sort_lists(RPpr_list, subtask_list):
    """
    对 RPpr_list 和 subtask_list 进行排序。
    当 RPpr_list 中的元素大小相同时，按照 subtask_list 中的元素大小排序。

    参数:
        RPpr_list (list): 第一个列表。
        subtask_list (list): 第二个列表。

    返回:
        RPpr_list_sorted (list): 排序后的 RPpr_list。
        subtask_list_sorted (list): 排序后的 subtask_list。
    """
    # 将两个列表打包成元组列表
    zipped_lists = list(zip(RPpr_list, subtask_list))

    # 自定义排序规则：先按 RPpr_list 排序，如果相同则按 subtask_list 排序
    sorted_pairs = sorted(zipped_lists, key=lambda x: (-x[0], x[1]))


    # 解包排序后的元组列表，得到排序后的 RPpr_list 和 subtask_list
    if sorted_pairs:
        RPpr_list_sorted, subtask_list_sorted = zip(*sorted_pairs)
        # 将结果转换回列表
        RPpr_list_sorted = list(RPpr_list_sorted)
        subtask_list_sorted = list(subtask_list_sorted)
    else:
        RPpr_list_sorted, subtask_list_sorted = [], []  # 或者根据你的需求处理空情况

    return RPpr_list_sorted, subtask_list_sorted

# 判断某个应用实例是否满足ddl
def judge_instance_ddl(app_index, instance_index,end_time_list):
    sink_nodes = app_list[app_index].sink_nodes()
    sink_nodes = [calc_subtask_instance_position(app_index,instance_index + 1,node) for node in sink_nodes]
    end_time = [end_time_list[node] for node in sink_nodes]
    # print(app_index, instance_index, max(end_time))
    return max(end_time) <= instance_dict_list[app_index][instance_index + 1]["ddl"]

if __name__== "__main__" :
    ### 定义应用集
    generate_dict_list = []
    generate_dict_list.append({"node num": 7, "edges": [(0, 4), (1, 4), (2, 3), (2, 5), (3, 6), (4, 6)], "period": 20,
                "sw exe": [4] * 7, "npu exe": [5] * 7, "hw exe": [3] * 7, "tpr": [2] * 7,
                "P static":[15,17,19,21,23,25,27], "P dyn":[10,9,8,7,6,5,4], "P sw":[10]*7, "P npu":[15]*7})
    generate_dict_list.append({"node num": 7, "edges": [(0, 4), (1, 4), (2, 3), (2, 5), (3, 6), (4, 6)], "period": 30,
               "sw exe": [3] * 7, "npu exe": [1,1,2,1,1,1,1], "hw exe": [2] * 7, "tpr": [1] * 7,
               "P static":[27,25,23,21,19,17,15], "P dyn":[4,5,6,7,8,9,10], "P sw":[10]*7, "P npu":[15]*7})
    RP_P_idle = 5              # 可重构分区无比特流时的静态能耗

    for generate_dict in generate_dict_list:
        app_list.append(DAG(generate_dict))

    ### 定义调度周期和应用实例
    H = calc_superperiod(app_list)  # 计算超周期
    instance_number_list = [int(H / app.period) for app in app_list]  # 计算每个应用的实例个数
    # 生成实例的到达时间、截止时间
    for i in range(len(app_list)):
        instance_dict_list[i] = {}
        for q in range(instance_number_list[i]):
            instance_dict_list[i][q + 1] = {}
            instance_dict_list[i][q + 1]["arr"] = q * app_list[i].period
            instance_dict_list[i][q + 1]["ddl"] = (q + 1) * app_list[i].period
    print(instance_dict_list)

    ### 定义处理器
    processor_number = CPU_number + NPU_number + RP_number
    processor_avail = [0] * processor_number    # 处理器空闲时间
    processor_RP_config_time = 0                        # 重构配置接口的空余时间
    processor_completed_task = [[] for _ in range(processor_number)]  # 处理器上已调度的任务
    processor_RP_task_list = []                                       # 记录每次重构的任务
    processor_RP_time_list = []                                       # 记录每次重构的[开始时间,截止时间]

    ### 定义决策变量
    task_number = 0  # 任务实例总数
    for i in range(len(app_list)):
        task_number += instance_number_list[i] * app_list[i].node_num
    task_mapping = [0] * task_number            # 任务映射
    task_start_time = [0] * task_number         # 任务开始执行时间
    task_end_time = [0] * task_number           # 任务结束时间
    task_DPR = [0] * task_number                # 任务是否重构
    task_DPR_start_time = [0] * task_number     # 任务重构开始时间
    task_DPR_end_time = [0] * task_number
    task_not_scheduled = list(range(task_number))   # 未被调度的任务
    task_into_degree = [0] * task_number        # 任务入度
    position = 0
    for i in range(len(app_list)):
        for q in range(instance_number_list[i]):
            for j in range(app_list[i].node_num):
                task_into_degree[position] = app_list[i].into_degree[j]
                position += 1
    print(task_into_degree)

    ### 生成随机映射
    position = 0
    for i in range(len(app_list)):
        for q in range(instance_number_list[i]):
            for j in range(app_list[i].node_num):
                task_mapping[position] = random.randrange(processor_number)
                position += 1

    # 根据任务映射进行调度
    CT = 0 # 当前时间
    while task_not_scheduled:
        print(f"current time:{CT}")
        # 获取当前可调度的任务集
        task_list = []
        mapping_list = {}
        for task in task_not_scheduled:
            if task_into_degree[task] == 0:     # 入度为0的任务才可能被调度
                i, q, j = calc_subtask_instance_from_position(task)
                RP_flag = define_RP(task, task_mapping[task],processor_completed_task[task_mapping[task]])
                # 只有任务所在应用实例(包含重构时间后)已经被触发，任务才可调度
                if RP_flag and CT >= instance_dict_list[i][q]["arr"] - app_list[i].tpr[j]:
                    task_list.append(task)
                elif CT >= instance_dict_list[i][q]["arr"]:
                    task_list.append(task)
        #         else:  # 任务不可调度，存储潜在的处理器移动值
        #             if task_mapping[task] not in mapping_list.keys():
        #                 mapping_list[task_mapping[task]] = max(0,instance_dict_list[i][q]["arr"]-int(RP_flag) * app_list[i].tpr[j])
        #             else:
        #                 mapping_list[task_mapping[task]] = min(mapping_list[task_mapping[task]], instance_dict_list[i][q]["arr"]-int(RP_flag) * app_list[i].tpr[j])
        # if not task_list:
        #     min_key, min_value = min(mapping_list.items(), key=lambda x: x[1])
        #     processor_avail[min_key] = min_value
        #     continue
        # 计算每个处理器上可调度的任务集内所有任务的优先级，并以此为依据进行排序
        processor_list = list(range(processor_number))
        random.shuffle(processor_list)
        for processor in processor_list:
            task_sublist = [task for task in task_list if task_mapping[task] == processor]
            if CT>= processor_avail[processor] and task_sublist:     # 当前时刻处理器空闲
                task_prio=[0] * len(task_sublist)
                for index,task in enumerate(task_sublist):
                    # 计算任务的最早开始执行时间
                    EST,task_DPR[task] = calc_EST(task,task_end_time, task_mapping[task],processor_avail[task_mapping[task]],processor_completed_task[task_mapping[task]],processor_RP_config_time)
                    # EST = max(EST,CT)
                    EST = CT
                    # 计算任务的最晚开始执行时间
                    i,q,j = calc_subtask_instance_from_position(task)
                    LST = calc_LST(task,task_mapping,instance_dict_list[i][q]["ddl"])
                    print(task, EST,LST)
                    # 计算任务执行时间
                    if task_mapping[task] < CPU_number:
                        exe = app_list[i].sw_exe[j]
                    elif task_mapping[task] < CPU_number + NPU_number:
                        exe = app_list[i].npu_exe[j]
                    else:
                        exe = app_list[i].hw_exe[j]
                    # 计算任务优先级
                    if LST-EST <= 0:
                        task_prio[index] = 9999
                    else:
                        task_prio[index] = exe/(LST-EST)
                task_prio_sorted, task_list_sorted = sort_lists(task_prio,task_sublist)
                # 取优先级最高的任务开始调度
                scheduling_task = task_list_sorted[0]
                i, q, j = calc_subtask_instance_from_position(scheduling_task)
                AST, task_DPR[scheduling_task] = calc_EST(scheduling_task, task_end_time, task_mapping[scheduling_task], processor_avail[task_mapping[scheduling_task]],
                                               processor_completed_task[task_mapping[scheduling_task]], processor_RP_config_time)
                                                                            # 任务实际开始时间、任务是否重构
                print(AST,CT)
                AST = max(AST, CT)
                if task_mapping[scheduling_task] < CPU_number:
                    exe = app_list[i].sw_exe[j]
                elif task_mapping[scheduling_task] < CPU_number + NPU_number:
                    exe = app_list[i].npu_exe[j]
                else:
                    exe = app_list[i].hw_exe[j]
                processor_avail[task_mapping[scheduling_task]] = AST + exe  # 处理器空闲时间更新
                if task_DPR[scheduling_task]:
                    if processor_completed_task[task_mapping[scheduling_task]]:
                        processor_RP_config_time_start = max(task_end_time[processor_completed_task[task_mapping[scheduling_task]][-1]],processor_RP_config_time)
                    else:
                        processor_RP_config_time_start = processor_RP_config_time
                    task_DPR_start_time[scheduling_task] = processor_RP_config_time_start           # 任务开始重构时间更新
                    task_DPR_end_time[scheduling_task] = processor_RP_config_time_start + app_list[i].tpr[j]
                    processor_RP_config_time = processor_RP_config_time_start + app_list[i].tpr[j]  # 可重构配置接口空余时间更新
                    processor_RP_task_list.append(scheduling_task)                                  # 任务需要重构，将任务放入重构记录队列
                    processor_RP_time_list.append([processor_RP_config_time_start, processor_RP_config_time])   # 任务需要重构，记录任务的开始重构时间和结束重构时间
                processor_completed_task[task_mapping[scheduling_task]].append(scheduling_task)  # 处理器已执行任务列表更新

                task_start_time[scheduling_task] = AST                                              # 任务开始执行时间更新
                task_end_time[scheduling_task] = AST + exe                                          # 任务结束执行时间更新
                task_not_scheduled.remove(scheduling_task)                                          # 从未调度的任务列表中删去已调度的任务
                scheduling_task_successors = list_successors(scheduling_task)
                for successor in scheduling_task_successors:
                    task_into_degree[successor] -= 1                                                # 后续节点入度减1
                print(f"任务{scheduling_task}，映射{task_mapping[scheduling_task]}，开始时间{task_start_time[scheduling_task]}，结束时间{task_end_time[scheduling_task]}")
        CT += 1

    # 判断是否满足截止时间约束
    flag1 = True
    overtime_list = []
    for i in range(len(app_list)):
        for q in range(instance_number_list[i]):
            flag = judge_instance_ddl(i,q,task_end_time)
            if not flag:
                flag1 = False
                overtime_list.append([i,q + 1])
    if flag1:
        print("所有应用实例满足截止时间约束！")
    else:
        print(f"不满时间约束的应用实例为{overtime_list}")

    ### 绘制调度结果
    # 随机生成每个应用的颜色
    app_color_list = random.sample(colors, len(app_list)+1)
    print(f"application colors:{app_color_list}")
    subtask_colors = [0] * task_number
    position = 0
    subtask_labels = [""] * task_number
    subtask_RP_info = [[0, 0, 0] for _ in range(task_number)]
    for i in range(len(app_list)):
        for q in range(instance_number_list[i]):
            for j in range(app_list[i].node_num):
                if task_DPR[position]:
                    subtask_RP_info[position][0] = 1
                    subtask_RP_info[position][1] = task_DPR_start_time[position]
                    subtask_RP_info[position][2] = task_DPR_start_time[position] + app_list[i].tpr[j]
                subtask_colors[position] = app_color_list[i]
                subtask_labels[position] = f"({i},{q + 1},{j})"
                position += 1
    plotter = ProcessorTimelinePlotter(processor_number, CPU_number, NPU_number, RP_number, time_axis_length=H)
    task_blocks = [
        {'processor': task_mapping[i], 'start': task_start_time[i], 'end': task_end_time[i], 'color': subtask_colors[i],
         'label': subtask_labels[i], 'RP': subtask_RP_info[i]} for i in range(task_number)
    ]
    print(f"task blocks:{task_blocks}")
    print(len(task_blocks))
    plotter.set("initial scheduling",task_blocks=task_blocks,is_save=True)
    energy = calc_energy(task_start_time, task_end_time, processor_completed_task, task_DPR_start_time, processor_RP_task_list)
    print(f"初始能耗为{energy}")

    ### 优化能耗
    RP_task_list = sum(processor_completed_task[-RP_number:], [])  # 在fpga上调度的任务list
    RP_task_list = [task for task in RP_task_list if task_DPR[task]]
    print(RP_task_list)
    # 获取任务的静态能耗比例list
    P_ratio_list = []
    for task in RP_task_list:
        task_index = processor_completed_task[task_mapping[task]].index(task)
        i, q, j = calc_subtask_instance_from_position(task)
        curr_P = RP_P_idle + app_list[i].P_static[j]
        if task_index == 0:
            prev_task = processor_completed_task[task_mapping[task]][-1]
            i_prev, q_prev, j_prev = calc_subtask_instance_from_position(prev_task)
            prev_P = RP_P_idle + app_list[i_prev].P_static[j_prev]
        else:
            prev_task = processor_completed_task[task_mapping[task]][task_index - 1]
            i_prev, q_prev, j_prev = calc_subtask_instance_from_position(prev_task)
            prev_P = RP_P_idle + app_list[i_prev].P_static[j_prev]
        P_ratio_list.append(curr_P / prev_P)
    print(P_ratio_list)
    P_ratio_list, RP_task_list = sort_lists(P_ratio_list, RP_task_list)
    print(P_ratio_list)
    print(RP_task_list)
    flag = True
    while flag:
        flag1 = False
        for index, task in enumerate(RP_task_list):
            print(f"任务{task},能耗比例{ P_ratio_list[index]}")
            if P_ratio_list[index] > 1:
                # 只是为了可视化
                i,q,j = calc_subtask_instance_from_position(task)
                for j in range(app_list[i].node_num):
                    # if calc_subtask_instance_position(i,q,j) in RP_task_list:
                    #     print(calc_subtask_instance_position(i,q,j), P_ratio_list[RP_task_list.index(calc_subtask_instance_position(i,q,j))])
                    task_blocks[calc_subtask_instance_position(i,q,j)]['color'] = app_color_list[-1]
                # plotter.set(f"initial scheduling {task}", task_blocks=task_blocks, is_block=True)
                ## 调整执行时间，重构时间
                # 调整任务图后续任务的执行时间（只调整不在fpga执行的任务）
                affected_task_list = list_affected_tasks(task, reverse=True)
                affected_task_list = [affected_task for affected_task in affected_task_list if task_mapping[affected_task] < CPU_number + NPU_number]
                # task_start_time_copy = task_start_time.copy()
                # task_end_time_copy = task_end_time.copy()
                # processor_completed_task_copy = processor_completed_task.copy()
                # for affected_task in affected_task_list:
                #     # 计算每个受影响任务的最晚执行时间
                #     affected_task_LFT = calc_ALFT(affected_task, processor_completed_task_copy[task_mapping[affected_task]],task_start_time_copy)
                #     print(affected_task, affected_task_LFT)
                #     task_end_time_copy[affected_task] = affected_task_LFT
                #     task_exe = task_end_time[affected_task] - task_start_time[affected_task]
                #     task_start_time_copy[affected_task] = affected_task_LFT - task_exe
                for affected_task in affected_task_list:
                    # 计算每个受影响任务的最晚执行时间
                    affected_task_LFT = calc_ALFT(affected_task, processor_completed_task[task_mapping[affected_task]],task_start_time)
                    # print(affected_task, affected_task_LFT)
                    if affected_task_LFT != task_end_time[affected_task]:
                        task_start_time_copy, task_end_time_copy = task_start_time[affected_task],task_end_time[affected_task]
                        task_end_time[affected_task] = affected_task_LFT
                        task_exe = task_end_time_copy - task_start_time_copy
                        task_start_time[affected_task] = affected_task_LFT - task_exe
                        # 只是为了可视化
                        print(f"任务{affected_task}可调整时间，原执行区间[{task_start_time_copy},{task_end_time_copy}]，"
                              f"现执行区间可调整为[{task_start_time[affected_task]},{task_end_time[affected_task]}]")
                        task_blocks[affected_task]["start"], task_blocks[affected_task]["end"] =task_start_time[affected_task], task_end_time[affected_task]
                        # plotter.set(f"initial scheduling {task}-{affected_task}", task_blocks=task_blocks, is_block=False)
                # 最晚结束时间是ddl，任务图后续节点开始时间，同一RP区下一任务开始时间的最小值
                LFT = calc_ALFT(task, processor_completed_task[task_mapping[task]], task_start_time, task_DPR_start_time, task_DPR,True)
                # 如果计算所得的任务最晚结束时间大于当前任务的实际结束时间，则考虑改变任务执行时间和任务重构时间
                if task_end_time[task] < LFT:
                    print(f"可以调整重构时间！{LFT}")
                    print(processor_RP_task_list)
                    LST = LFT - (task_end_time[task] - task_start_time[task])
                    # 计算前一任务的执行结束时间，缩小重构时间的可行区间
                    task_index = processor_completed_task[task_mapping[task]].index(task)
                    if task_index != 0:
                        prev_task = processor_completed_task[task_mapping[task]][task_index - 1]
                        prev_task_end_time = task_end_time[prev_task]
                    else:
                        prev_task_end_time = 0
                    # 删除现有的重构时间，在可行的重构时间内寻找最靠后的时间
                    # processor_RP_time_list_copy = processor_RP_time_list.copy()
                    # processor_RP_task_list_copy = processor_RP_task_list.copy()
                    del processor_RP_time_list[processor_RP_task_list.index(task)]
                    processor_RP_task_list.remove(task)
                    RP_interval = calc_last_RP_interval(processor_RP_time_list,prev_task_end_time, LST, app_list[i].tpr[j])
                    if RP_interval[0] != task_DPR_start_time[task]:
                        print(f"任务{task}原重构时间：[{task_DPR_start_time[task]},{task_DPR_start_time[task] + app_list[i].tpr[j]}],新重构时间：{RP_interval}")
                        print(f"任务{task}原执行时间：[{task_start_time[task]},{task_end_time[task]}],新执行时间：[{LST},{LFT}]")
                        task_start_time[task], task_end_time[task] = LST, LFT
                        task_DPR_start_time[task] = RP_interval[0]
                        task_DPR_end_time[task] = RP_interval[1]
                        # 只是为了可视化
                        task_blocks[task]["start"], task_blocks[task]["end"] = task_start_time[task], task_end_time[task]
                        task_blocks[task]["RP"] = [1, task_DPR_start_time[task], task_DPR_start_time[task] + app_list[i].tpr[j]]
                        flag1 = True
                    # 重新插入重构区间
                    insert_RP_interval(processor_RP_time_list, RP_interval, processor_RP_task_list, task)
                    if flag1:
                        break
                # 只是为了可视化
                # plotter.set(f"initial scheduling {task}", task_blocks=task_blocks, is_block=True)
                for j in range(app_list[i].node_num):
                    task_blocks[calc_subtask_instance_position(i,q,j)]['color'] = app_color_list[i]
            else:
                print(f"任务{task}能耗比小于0")
                break
        if flag1:
            continue
        else:
            flag = False

    plotter.set(f"initial scheduling final", task_blocks=task_blocks, is_save=True)
    energy = calc_energy(task_start_time, task_end_time, processor_completed_task, task_DPR_start_time,
                         processor_RP_task_list)
    print(f"优化后能耗为{energy}")
