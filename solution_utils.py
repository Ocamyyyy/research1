import random
import math

import numpy as np

from task_definition import (
    TaskConfig,
    average_eligible_execution_time_for_priority,
)
from collections import deque
from time_slot_plot import ProcessorTimelinePlotter, colors
import time
import bisect

# 计算子任务实例在决策变量中的位置(instance_index从1开始计数方式)
def calc_subtask_instance_position(Task:TaskConfig, app_index,instance_index,subtask_index):
    pos = 0
    for pre_app_index in range(app_index):
        pos += Task.instance_number_list[pre_app_index] * Task.app_list[pre_app_index].node_num
    pos += (instance_index-1) * Task.app_list[app_index].node_num
    pos += subtask_index
    return pos

# 记录决策变量位置对应的子任务实例
def calc_subtask_instance_from_position(Task:TaskConfig, pos):
    i=0
    q=0
    pos_total = 0
    for app_index in range(len(Task.app_list)):
        if pos_total + Task.instance_number_list[app_index] * Task.app_list[app_index].node_num -1 < pos:
            pos_total += Task.instance_number_list[app_index] * Task.app_list[app_index].node_num
        else:
            i = app_index
            break
    for instance_index in range(Task.instance_number_list[i]):
        if pos_total + Task.app_list[i].node_num -1 <pos:
            pos_total += Task.app_list[i].node_num
        else:
            q = instance_index + 1
            break
    j = pos - pos_total
    return i,q,j


def is_processor_eligible(Task: TaskConfig, task_pos: int, processor: int):
    i, _q, j = calc_subtask_instance_from_position(Task, task_pos)
    app = Task.app_list[i]
    if 0 <= processor < Task.CPU_number:
        return bool(app.cpu_eligible[j])
    if Task.CPU_number <= processor < Task.CPU_number + Task.NPU_number:
        return bool(app.npu_eligible[j])
    if Task.CPU_number + Task.NPU_number <= processor < Task.processor_number:
        return bool(app.hw_eligible[j])
    return False


def eligible_processor_ids(Task: TaskConfig, task_pos: int):
    i, _q, j = calc_subtask_instance_from_position(Task, task_pos)
    app = Task.app_list[i]
    eligible = list(range(Task.CPU_number))
    if app.npu_eligible[j]:
        eligible.extend(range(Task.CPU_number, Task.CPU_number + Task.NPU_number))
    if app.hw_eligible[j]:
        eligible.extend(range(Task.CPU_number + Task.NPU_number, Task.processor_number))
    return eligible


def is_mapping_eligible(Task: TaskConfig, mapping):
    return all(mapping[task] in eligible_processor_ids(Task, task) for task in range(Task.task_number))


def assert_mapping_eligible(Task: TaskConfig, mapping, context=""):
    for task in range(Task.task_number):
        processor = mapping[task]
        if processor in eligible_processor_ids(Task, task):
            continue
        app_index, instance_index, node_index = calc_subtask_instance_from_position(Task, task)
        processor_type = "CPU"
        if processor >= Task.CPU_number + Task.NPU_number:
            processor_type = "RP"
        elif processor >= Task.CPU_number:
            processor_type = "NPU"
        raise ValueError(
            "Illegal mapping: "
            f"task={task}, app={app_index}, instance={instance_index}, node={node_index}, "
            f"processor={processor} ({processor_type}), context={context}"
        )


def project_mapping_to_eligible_processors(Task: TaskConfig, raw_mapping):
    projected = []
    for task, raw_value in enumerate(raw_mapping):
        eligible = eligible_processor_ids(Task, task)
        rounded = int(round(raw_value))
        if rounded in eligible:
            projected.append(rounded)
            continue
        projected.append(min(eligible, key=lambda processor: (abs(processor - rounded), processor)))
    return projected


def find_valid_swap_pair(Task: TaskConfig, mapping, max_attempts=100):
    for _ in range(max_attempts):
        i, j = random.sample(range(Task.task_number), 2)
        pi = mapping[i]
        pj = mapping[j]
        if pi == pj:
            continue
        if pj in eligible_processor_ids(Task, i) and pi in eligible_processor_ids(Task, j):
            return i, j
    return None

# 判断任务是否重构
def define_RP(Task:TaskConfig, task:int, processor, task_before):
    i, q, j = calc_subtask_instance_from_position(Task,task)
    if processor >= Task.CPU_number + Task.NPU_number:  # 如果映射到fpga
        # 判断是否需要重构
        if task_before:  # 先前没有任务执行，必定重构
            i_prev, q_prev, j_prev = calc_subtask_instance_from_position(Task, task_before[-1])
            if i_prev == i and j_prev == j:
                return False
        return True
    return False

# 计算任务的最早开始执行时间
# 输入：任务编号,任务结束时间（list），任务映射的处理器,任务映射的处理器的空余时间，任务映射的处理器已执行的任务，重构配置接口的空余时间
def calc_EST(Task:TaskConfig, task:int, end_time:list, processor, avail:int, task_before, avail_RP):
    i, q, j = calc_subtask_instance_from_position(Task, task)
    predecessor_of_task = Task.app_list[i].predecessor_of_node(j)
    predecessor_FT = []
    if predecessor_of_task:
        predecessor_FT = [end_time[calc_subtask_instance_position(Task, i, q, index)] for index in predecessor_of_task]
    # 判断是否需要重构
    RP_flag = define_RP(Task, task, processor, task_before)
    if RP_flag:
        if predecessor_FT:
            return max(max(avail,avail_RP)+Task.app_list[i].tpr[j],Task.instance_dict_list[i][q]["arr"], max(predecessor_FT)),True
        return max(max(avail,avail_RP)+Task.app_list[i].tpr[j],Task.instance_dict_list[i][q]["arr"]),True
    if predecessor_FT:
        return max(avail,Task.instance_dict_list[i][q]["arr"], max(predecessor_FT)),False
    return max(avail,Task.instance_dict_list[i][q]["arr"]),False

# 计算任务的最迟开始执行时间
# 输入：任务编号，所有任务映射的处理器，截止时间，任务是否重构
def calc_LST(Task:TaskConfig, task, processors,ddl):
    i,q,j = calc_subtask_instance_from_position(Task, task)
    ou_degree = Task.app_list[i].out_degree.copy()
    nodes = list(range(Task.app_list[i].node_num))

    # 初始化最早开始时间
    latest_time = [ddl] * Task.app_list[i].node_num

    # 找到所有入度为0的节点（起始节点）
    queue = deque([node for node in nodes if ou_degree[node] == 0])
    for node in queue:
        pos = calc_subtask_instance_position(Task, i,q,node)
        if processors[pos] < Task.CPU_number:
            latest_time[node] = ddl- Task.app_list[i].sw_exe[node]
        elif processors[pos] < Task.CPU_number + Task.NPU_number:
            latest_time[node] = ddl- Task.app_list[i].npu_exe[node]
        else:
            latest_time[node] = ddl- Task.app_list[i].hw_exe[node] - Task.app_list[i].tpr[node]

    # 拓扑排序
    while queue:
        u = queue.popleft()
        u_pos =calc_subtask_instance_position(Task, i,q,u)
        if u_pos == task:
            return latest_time[u]
        u_time = latest_time[u]
        if processors[u_pos] >= Task.CPU_number + Task.NPU_number:
            u_time -= Task.app_list[i].tpr[u]
        # 遍历u的所有邻接节点
        for v in Task.app_list[i].reverse_graph[u]:
            # 更新v的最早开始时间
            pos = calc_subtask_instance_position(Task, i, q, v)
            if processors[pos] < Task.CPU_number:
                latest_time[v] = min(latest_time[v],u_time - Task.app_list[i].sw_exe[v])
            elif processors[pos] < Task.CPU_number + Task.NPU_number:
                latest_time[v] = min(latest_time[v],u_time - Task.app_list[i].npu_exe[v])
            else:
                latest_time[v] = min(latest_time[v],u_time - Task.app_list[i].hw_exe[v])
            # 减少v的入度
            ou_degree[v] -= 1
            # 如果v的入度为0，加入队列
            if ou_degree[v] == 0:
                queue.append(v)
    # 如果目标节点不在图中（理论上不会，因为输入合法）
    raise ValueError(f"Task {task} not found in the graph.")

# （若任务后续节点时间不动）计算任务的实际最晚执行结束时间
def calc_ALFT(Task:TaskConfig, task, processor_completed_task_list, task_start_time, task_DPR_start_time=None, task_DPR=None, mode_DPR:bool = False):
    i,q,j  = calc_subtask_instance_from_position(Task, task)
    succ = list_successors(Task,task)
    # 后续节点最早开始时间
    if succ:
        min_succ = min([task_start_time[mm] for mm in succ])
    else:
        min_succ = Task.instance_dict_list[i][q]["ddl"]
    # 处理器上后续任务最早开始时间
    next_task = calc_next_task_on_processor(task, processor_completed_task_list)
    # print(processor_completed_task_list)
    # print(f"next task:{next_task}")
    if next_task != None:
        if mode_DPR:
            if task_DPR[next_task]:
                next_task_start_time = task_DPR_start_time[next_task]
            else:
                next_task_start_time = task_start_time[next_task]
            # print(f"next task start time:{next_task_start_time}")
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
        if b <= prev_end:
            gap_start = b
            gap_end = prev_end
            gap_length = gap_end - gap_start
            if gap_length >= e:
                start = gap_end - e
                return [start, gap_end]
        prev_end = min(prev_end, a)
        if c > prev_end:
            break
    if prev_end >= c:
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

def calc_CPU_dynamic_energy(Task:TaskConfig, task_start_time, task_end_time, task_list):
    sum = 0
    if task_list:
        for task in task_list:
            i,q,j = calc_subtask_instance_from_position(Task,task)
            sum += (task_end_time[task]-task_start_time[task]) * Task.app_list[i].P_sw[j]
    return sum

def calc_NPU_dynamic_energy(Task:TaskConfig,task_start_time, task_end_time, task_list):
    sum = 0
    if task_list:
        for task in task_list:
            i,q,j = calc_subtask_instance_from_position(Task,task)
            sum += (task_end_time[task]-task_start_time[task]) * Task.app_list[i].P_npu[j]
    return sum

def _overlap_length(start, end, cutoff_time):
    if start >= cutoff_time:
        return 0.0
    return max(0.0, min(end, cutoff_time) - start)


def calc_RP_dynamic_energy(
    Task:TaskConfig,
    task_start_time,
    task_end_time,
    task_list,
    task_DPR_start_time,
    task_DPR_end_time,
    task_DPR,
    interval_end=None,
):
    if interval_end is None:
        interval_end = Task.H
    interval_end = min(interval_end, Task.H)
    if interval_end <= 0:
        return 0.0

    energy = interval_end * Task.RP_P_idle
    if not task_list:
        return energy

    current_segment = None
    for task in task_list:
        i, q, j = calc_subtask_instance_from_position(Task, task)
        exec_overlap = _overlap_length(task_start_time[task], task_end_time[task], interval_end)

        if task_DPR[task]:
            if current_segment is not None:
                current_segment["end"] = min(current_segment["end"], task_DPR_start_time[task], interval_end)
                if current_segment["end"] > current_segment["start"]:
                    energy += (current_segment["static_power"] - Task.RP_P_idle) * (
                        current_segment["end"] - current_segment["start"]
                    )
                current_segment = None

            load_time = min(task_DPR_end_time[task], interval_end)
            if load_time < interval_end:
                current_segment = {
                    "start": load_time,
                    "end": interval_end,
                    "static_power": Task.app_list[i].P_static[j],
                }
        elif current_segment is None:
            current_segment = {
                "start": min(task_start_time[task], interval_end),
                "end": interval_end,
                "static_power": Task.app_list[i].P_static[j],
            }

        energy += exec_overlap * Task.app_list[i].P_dyn[j]

    if current_segment is not None:
        current_segment["end"] = min(current_segment["end"], interval_end)
        if current_segment["end"] > current_segment["start"]:
            energy += (current_segment["static_power"] - Task.RP_P_idle) * (
                current_segment["end"] - current_segment["start"]
            )

    return energy

# 返回某个任务实例的后续任务节点list
def list_successors(Task:TaskConfig,task):
    i,q,j=calc_subtask_instance_from_position(Task,task)
    successors = Task.app_list[i].successor_of_node(j)
    return [calc_subtask_instance_position(Task,i,q,successor) for successor in successors]

# 返回受某个任务实例开始执行时间的影响的后续所有任务节点list(包括该任务自身)
def list_affected_tasks(Task:TaskConfig,task, reverse=False):
    i,q,j = calc_subtask_instance_from_position(Task,task)
    successors = Task.app_list[i].find_affected_tasks(j,reverse=reverse)
    return [calc_subtask_instance_position(Task,i,q,successor) for successor in successors]

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
def judge_instance_ddl(Task:TaskConfig,app_index, instance_index,end_time_list):
    sink_nodes = Task.app_list[app_index].sink_nodes()
    sink_nodes = [calc_subtask_instance_position(Task,app_index,instance_index + 1,node) for node in sink_nodes]
    end_time = [end_time_list[node] for node in sink_nodes]
    # print(app_index, instance_index, max(end_time))
    return max(end_time)-Task.instance_dict_list[app_index][instance_index + 1]["ddl"] if max(end_time)>Task.instance_dict_list[app_index][instance_index + 1]["ddl"] else 0

def priority_list_generate(Task:TaskConfig)->list:
    priority_list = [0] * Task.task_number
    for i in range(len(Task.app_list)):
        for q in range(Task.instance_number_list[i]):
            ou_degree = Task.app_list[i].out_degree.copy()
            nodes = list(range(Task.app_list[i].node_num))

            # 初始化最早开始时间
            priority_base = (Task.instance_number_list[i] - q - 1) * Task.app_list[i].period

            # 找到所有入度为0的节点（起始节点）
            queue = deque([node for node in nodes if ou_degree[node] == 0])
            for node in queue:
                pos = calc_subtask_instance_position(Task, i, q + 1, node)
                avg_exe = average_eligible_execution_time_for_priority(Task.app_list[i], node)
                priority_list[pos] = avg_exe + priority_base

            # 拓扑排序
            while queue:
                u = queue.popleft()
                u_pos = calc_subtask_instance_position(Task, i, q + 1, u)
                u_prio = priority_list[u_pos]
                # 遍历u的所有邻接节点
                for v in Task.app_list[i].reverse_graph[u]:
                    pos = calc_subtask_instance_position(Task, i, q + 1, v)
                    avg_exe = average_eligible_execution_time_for_priority(Task.app_list[i], v)
                    priority_list[pos] = max(priority_list[pos], avg_exe + u_prio)
                    # 减少v的入度
                    ou_degree[v] -= 1
                    # 如果v的入度为0，加入队列
                    if ou_degree[v] == 0:
                        queue.append(v)
    return priority_list

class MappingSolution:
    def __init__(self, CPU_number, NPU_number, RP_number, task_number):
        self.processor_number = CPU_number + NPU_number + RP_number

        # 处理器相关决策变量
        self.processor_avail = [0] * self.processor_number                           # 处理器空闲时间
        self.processor_RP_config_time = 0                                       # 重构配置接口的空余时间
        self.processor_completed_task = [[] for _ in range(self.processor_number)]   # 处理器上已调度的任务
        self.processor_RP_task_list = []                                        # 记录重构的任务
        self.processor_RP_time_list = []                                        # 记录每次重构的[开始时间,截止时间]

        # 任务相关决策变量
        self.task_mapping = [0] * task_number                                   # 任务映射
        self.task_start_time = [0] * task_number                                # 任务开始执行时间
        self.task_end_time = [0] * task_number                                  # 任务结束执行时间
        self.task_DPR = [0] * task_number                                       # 任务是否重构
        self.task_DPR_start_time = [0] * task_number                            # 任务重构开始时间
        self.task_DPR_end_time = [0] * task_number                              # 任务重构结束时间
        self.task_not_scheduled = list(range(task_number))                      # 未被调度的任务
        self.task_into_degree = [0] * task_number                               # 任务入度

        # 可视化工具
        self.task_blocks = []
        self.plotter = None
        self.app_color_list = []

    def init_randomly(self,Task):
        position = 0
        for i in range(len(Task.app_list)):
            for q in range(Task.instance_number_list[i]):
                for j in range(Task.app_list[i].node_num):
                    self.task_mapping[position] = random.choice(eligible_processor_ids(Task, position))
                    position += 1
    def init_definitely(self,task_map):
        self.task_mapping = task_map.copy()

    def clear_scheduling(self, task_number):
        # 处理器相关决策变量
        self.processor_avail = [0] * self.processor_number  # 处理器空闲时间
        self.processor_RP_config_time = 0  # 重构配置接口的空余时间
        self.processor_completed_task = [[] for _ in range(self.processor_number)]  # 处理器上已调度的任务
        self.processor_RP_task_list = []  # 记录每次重构的任务
        self.processor_RP_time_list = []  # 记录每次重构的[开始时间,截止时间]

        # 任务相关决策变量
        self.task_start_time = [0] * task_number  # 任务开始执行时间
        self.task_end_time = [0] * task_number  # 任务结束执行时间
        self.task_DPR = [0] * task_number  # 任务是否重构
        self.task_DPR_start_time = [0] * task_number  # 任务重构开始时间
        self.task_DPR_end_time = [0] * task_number  # 任务重构结束时间
        self.task_not_scheduled = list(range(task_number))  # 未被调度的任务
        self.task_into_degree = [0] * task_number  # 任务入度

    def scheduling(self,Task:TaskConfig):
        # 清空原来的调度
        # 更新任务入度
        position = 0
        for i in range(len(Task.app_list)):
            for q in range(Task.instance_number_list[i]):
                for j in range(Task.app_list[i].node_num):
                    self.task_into_degree[position] = Task.app_list[i].into_degree[j]
                    position += 1

        task_available_list = []
        # 提取所有 [i, q+1, ddl] 三元组
        arr_entries = []
        for i in range(len(Task.instance_dict_list)):
            for q_plus_1 in Task.instance_dict_list[i].keys():
                arr = Task.instance_dict_list[i][q_plus_1]["arr"]
                arr_entries.append((i, q_plus_1, arr))
        # 按 ddl 从小到大排序
        sorted_entries = sorted(arr_entries, key=lambda x: x[2])
        # 分离结果（如果需要单独获取索引和ddl值）
        sorted_arrs = [entry[2] for entry in sorted_entries]
        idx = 0
        for idx,(i, q_plus_1, arr) in enumerate(sorted_entries):
            if arr == 0:
                for j in range(Task.app_list[i].node_num):
                    task_available_list.append(calc_subtask_instance_position(Task,i,q_plus_1,j))
            else:
                break
        del sorted_arrs[:idx]
        del sorted_entries[:idx]
        # print("start...")
        CT = 0
        while self.task_not_scheduled:
            flag = True
            for processor in range(Task.processor_number):
                if CT >= self.processor_avail[processor]:
                    flag = False
                    break
            if flag:
                CT +=1
                continue
            # print(f"current time:{CT}")
            # 获取当前可调度的任务集
            task_list = []
            processor_list = set([])
            # start = time.time()
            for task in task_available_list:
                if self.task_into_degree[task] == 0:  # 入度为0的任务才可能被调度
                    task_list.append(task)
                    processor_list.add(self.task_mapping[task])
            # print(f"time cost: {time.time() - start:.2f}s")
            # start = time.time()
            # print(processor_list)
            for processor in processor_list:
                if CT < self.processor_avail[processor]:
                    continue
                task_sublist = [task for task in task_list if self.task_mapping[task] == processor]
                if task_sublist:
                    task_prio = [0] * len(task_sublist)
                    for index, task in enumerate(task_sublist):
                        # 计算任务的最早开始执行时间
                        # EST, self.task_DPR[task] = calc_EST(task, self.task_end_time, self.task_mapping[task],
                        #                                self.processor_avail[self.task_mapping[task]],
                        #                                self.processor_completed_task[self.task_mapping[task]],
                        #                                self.processor_RP_config_time)
                        EST = CT
                        # 计算任务的最晚开始执行时间
                        i, q, j = calc_subtask_instance_from_position(Task,task)
                        LST = calc_LST(Task,task, self.task_mapping, Task.instance_dict_list[i][q]["ddl"])
                        # print(LST)
                        # print(task, EST, LST)
                        # 计算任务执行时间
                        if self.task_mapping[task] < Task.CPU_number:
                            exe = Task.app_list[i].sw_exe[j]
                        elif self.task_mapping[task] < Task.CPU_number + Task.NPU_number:
                            exe = Task.app_list[i].npu_exe[j]
                        else:
                            exe = Task.app_list[i].hw_exe[j]
                        # 计算任务优先级
                        if LST - EST <= 0:
                            task_prio[index] = 9999
                        else:
                            task_prio[index] = exe / (LST - EST)
                    task_prio_sorted, task_list_sorted = sort_lists(task_prio, task_sublist)
                    # 取优先级最高的任务开始调度
                    scheduling_task = task_list_sorted[0]
                    i, q, j = calc_subtask_instance_from_position(Task,scheduling_task)
                    AST, self.task_DPR[scheduling_task] = calc_EST(Task,scheduling_task, self.task_end_time,
                                                              self.task_mapping[scheduling_task],
                                                              self.processor_avail[self.task_mapping[scheduling_task]],
                                                              self.processor_completed_task[self.task_mapping[scheduling_task]],
                                                              self.processor_RP_config_time)
                    # 任务实际开始时间、任务是否重构
                    if self.task_mapping[scheduling_task] < Task.CPU_number:
                        exe = Task.app_list[i].sw_exe[j]
                    elif self.task_mapping[scheduling_task] < Task.CPU_number + Task.NPU_number:
                        exe = Task.app_list[i].npu_exe[j]
                    else:
                        exe = Task.app_list[i].hw_exe[j]
                    self.processor_avail[self.task_mapping[scheduling_task]] = AST + exe  # 处理器空闲时间更新
                    if self.task_DPR[scheduling_task]:
                        if self.processor_completed_task[self.task_mapping[scheduling_task]]:
                            processor_RP_config_time_start = max(
                                self.task_end_time[self.processor_completed_task[self.task_mapping[scheduling_task]][-1]],
                                self.processor_RP_config_time)
                        else:
                            processor_RP_config_time_start = self.processor_RP_config_time
                        self.task_DPR_start_time[scheduling_task] = processor_RP_config_time_start  # 任务开始重构时间更新
                        self.task_DPR_end_time[scheduling_task] = processor_RP_config_time_start + Task.app_list[i].tpr[j]
                        self.processor_RP_config_time = processor_RP_config_time_start + Task.app_list[i].tpr[j]  # 可重构配置接口空余时间更新
                        self.processor_RP_task_list.append(scheduling_task)  # 任务需要重构，将任务放入重构记录队列
                        self.processor_RP_time_list.append(
                            [processor_RP_config_time_start, self.processor_RP_config_time])  # 任务需要重构，记录任务的开始重构时间和结束重构时间
                    self.processor_completed_task[self.task_mapping[scheduling_task]].append(scheduling_task)  # 处理器已执行任务列表更新

                    self.task_start_time[scheduling_task] = AST  # 任务开始执行时间更新
                    self.task_end_time[scheduling_task] = AST + exe  # 任务结束执行时间更新
                    self.task_not_scheduled.remove(scheduling_task)  # 从未调度的任务列表中删去已调度的任务
                    task_available_list.remove(scheduling_task)
                    scheduling_task_successors = list_successors(Task,scheduling_task)
                    for successor in scheduling_task_successors:
                        self.task_into_degree[successor] -= 1  # 后续节点入度减1
                    # print(
                    #     f"任务{scheduling_task}，映射{self.task_mapping[scheduling_task]}，开始时间{self.task_start_time[scheduling_task]}，结束时间{self.task_end_time[scheduling_task]}")
                    # print(self.task_not_scheduled)

            index = bisect.bisect_right(sorted_arrs, CT)
            # 若任务队列为空但任务未被调度完，将触发时间距离max(self.processor_avail)最近的任务加入队列
            # if not task_available_list and self.task_not_scheduled:
            #     for j in range(TaskConfig.app_list[sorted_entries[0][0]].node_num):
            #         task_available_list.append(calc_subtask_instance_position(sorted_entries[0][0], sorted_entries[0][1], j))
            #     del sorted_arrs[:1]
            #     del sorted_entries[:1]
                # print(len(sorted_arrs))
            if index > 0 and sorted_arrs:
                for idx in range(index):
                    for j in range(Task.app_list[sorted_entries[idx][0]].node_num):
                        task_available_list.append(
                            calc_subtask_instance_position(Task,sorted_entries[idx][0], sorted_entries[idx][1], j))
                del sorted_arrs[:index]
                del sorted_entries[:index]

            # print(f"time cost: {time.time() - start:.2f}s")
            # start = time.time()
            CT += 1
        # print("over")

    def scheduling2(self,Task:TaskConfig):
        # 清空原来的调度
        # 更新任务入度
        position = 0
        for i in range(len(Task.app_list)):
            for q in range(Task.instance_number_list[i]):
                for j in range(Task.app_list[i].node_num):
                    self.task_into_degree[position] = Task.app_list[i].into_degree[j]
                    position += 1

        CT = 0  # 当前时间
        while self.task_not_scheduled:
            # print(f"current time:{CT}")
            # 获取当前可调度的任务集
            task_list = []
            start = time.time()
            for task in self.task_not_scheduled:
                if self.task_into_degree[task] == 0:  # 入度为0的任务才可能被调度
                    i, q, j = calc_subtask_instance_from_position(Task,task)
                    RP_flag = define_RP(Task,task, self.task_mapping[task], self.processor_completed_task[self.task_mapping[task]])
                    # 只有任务所在应用实例(包含重构时间后)已经被触发，任务才可调度
                    if RP_flag and CT >= Task.instance_dict_list[i][q]["arr"] - Task.app_list[i].tpr[j]:
                        task_list.append(task)
                    elif CT >= Task.instance_dict_list[i][q]["arr"]:
                        task_list.append(task)
            print(f"time cost: {time.time() - start:.2f}s")
            start = time.time()
            processor_list = list(range(self.processor_number))
            random.shuffle(processor_list)
            for processor in processor_list:
                task_sublist = [task for task in task_list if self.task_mapping[task] == processor]
                if CT >= self.processor_avail[processor] and task_sublist:  # 当前时刻处理器空闲
                    task_prio = [0] * len(task_sublist)
                    for index, task in enumerate(task_sublist):
                        # 计算任务的最早开始执行时间
                        EST, self.task_DPR[task] = calc_EST(Task,task, self.task_end_time, self.task_mapping[task],
                                                       self.processor_avail[self.task_mapping[task]],
                                                       self.processor_completed_task[self.task_mapping[task]],
                                                       self.processor_RP_config_time)
                        # EST = max(EST,CT)
                        EST = CT
                        # 计算任务的最晚开始执行时间
                        i, q, j = calc_subtask_instance_from_position(Task,task)
                        LST = calc_LST(Task,task, self.task_mapping, Task.instance_dict_list[i][q]["ddl"])
                        # print(LST)
                        # print(task, EST, LST)
                        # 计算任务执行时间
                        if self.task_mapping[task] < Task.CPU_number:
                            exe = Task.app_list[i].sw_exe[j]
                        elif self.task_mapping[task] < Task.CPU_number + Task.NPU_number:
                            exe = Task.app_list[i].npu_exe[j]
                        else:
                            exe = Task.app_list[i].hw_exe[j]
                        # 计算任务优先级
                        if LST - EST <= 0:
                            task_prio[index] = 9999
                        else:
                            task_prio[index] = exe / (LST - EST)
                    task_prio_sorted, task_list_sorted = sort_lists(task_prio, task_sublist)
                    # 取优先级最高的任务开始调度
                    scheduling_task = task_list_sorted[0]
                    i, q, j = calc_subtask_instance_from_position(Task,scheduling_task)
                    AST, self.task_DPR[scheduling_task] = calc_EST(Task,scheduling_task, self.task_end_time,
                                                              self.task_mapping[scheduling_task],
                                                              self.processor_avail[self.task_mapping[scheduling_task]],
                                                              self.processor_completed_task[self.task_mapping[scheduling_task]],
                                                              self.processor_RP_config_time)
                    # 任务实际开始时间、任务是否重构
                    AST = max(AST, CT)
                    if self.task_mapping[scheduling_task] < Task.CPU_number:
                        exe = Task.app_list[i].sw_exe[j]
                    elif self.task_mapping[scheduling_task] < Task.CPU_number + Task.NPU_number:
                        exe = Task.app_list[i].npu_exe[j]
                    else:
                        exe = Task.app_list[i].hw_exe[j]
                    self.processor_avail[self.task_mapping[scheduling_task]] = AST + exe  # 处理器空闲时间更新
                    if self.task_DPR[scheduling_task]:
                        if self.processor_completed_task[self.task_mapping[scheduling_task]]:
                            processor_RP_config_time_start = max(
                                self.task_end_time[self.processor_completed_task[self.task_mapping[scheduling_task]][-1]],
                                self.processor_RP_config_time)
                        else:
                            processor_RP_config_time_start = self.processor_RP_config_time
                        self.task_DPR_start_time[scheduling_task] = processor_RP_config_time_start  # 任务开始重构时间更新
                        self.task_DPR_end_time[scheduling_task] = processor_RP_config_time_start + Task.app_list[i].tpr[j]
                        self.processor_RP_config_time = processor_RP_config_time_start + Task.app_list[i].tpr[j]  # 可重构配置接口空余时间更新
                        self.processor_RP_task_list.append(scheduling_task)  # 任务需要重构，将任务放入重构记录队列
                        self.processor_RP_time_list.append(
                            [processor_RP_config_time_start, self.processor_RP_config_time])  # 任务需要重构，记录任务的开始重构时间和结束重构时间
                    self.processor_completed_task[self.task_mapping[scheduling_task]].append(scheduling_task)  # 处理器已执行任务列表更新

                    self.task_start_time[scheduling_task] = AST  # 任务开始执行时间更新
                    self.task_end_time[scheduling_task] = AST + exe  # 任务结束执行时间更新
                    self.task_not_scheduled.remove(scheduling_task)  # 从未调度的任务列表中删去已调度的任务
                    scheduling_task_successors = list_successors(Task,scheduling_task)
                    for successor in scheduling_task_successors:
                        self.task_into_degree[successor] -= 1  # 后续节点入度减1
                    # print(
                    #     f"任务{scheduling_task}，映射{self.task_mapping[scheduling_task]}，开始时间{self.task_start_time[scheduling_task]}，结束时间{self.task_end_time[scheduling_task]}")

            print(f"time cost: {time.time() - start:.2f}s")
            start = time.time()
            # CT = min(self.processor_avail)
            CT=CT+1
            print(f"CT:{CT}")

    def heft_scheduling(self,Task:TaskConfig):
        self.clear_scheduling(Task.task_number)
        # 生成任务优先级
        priority_list = priority_list_generate(Task)
        task_list = list(range(Task.task_number))
        # position = 0
        # for i in range(len(TaskConfig.app_list)):
        #     for q in range(TaskConfig.instance_number_list[i]):
        #         for j in range(TaskConfig.app_list[i].node_num):
        #             print(f"{i},{q},{j},{priority_list[position]}")
        #             position += 1
        priority_list_sorted, task_list_sorted = sort_lists(priority_list, task_list)
        for idx, task in enumerate(task_list_sorted):
            i,q,j = calc_subtask_instance_from_position(Task,task)
            EFT_list = []
            candidate_processors = eligible_processor_ids(Task, task)
            arr = Task.instance_dict_list[i][q]["arr"]
            predecessor_end_time = arr
            predecessor = Task.app_list[i].predecessor_of_node(j)
            if predecessor:
                predecessor = [calc_subtask_instance_position(Task,i,q,j2) for j2 in predecessor]
                predecessor_end_time_list = [self.task_end_time[pre_task] for pre_task in predecessor]
                predecessor_end_time = max(predecessor_end_time_list)
            for processor in candidate_processors:
                if processor < Task.CPU_number:
                    exe = Task.app_list[i].sw_exe[j]
                    EFT = max(predecessor_end_time, self.processor_avail[processor]) + exe
                elif processor <Task.CPU_number + Task.NPU_number:
                    exe = Task.app_list[i].npu_exe[j]
                    EFT = max(predecessor_end_time, self.processor_avail[processor]) + exe
                else:
                    DRP = True
                    if self.processor_completed_task[processor]:
                        task_last = self.processor_completed_task[processor][-1]
                        i2, q2, j2 = calc_subtask_instance_from_position(Task,task_last)
                        if i2==i and j2==j:
                            DRP = False
                    DRP_end_time = self.processor_avail[processor]
                    if DRP:
                        DRP_start_time = max(self.processor_avail[processor], self.processor_RP_config_time)
                        DRP_end_time = DRP_start_time + Task.app_list[i].tpr[j]
                    exe = Task.app_list[i].hw_exe[j]
                    EFT = max(predecessor_end_time, DRP_end_time) + exe
                EFT_list.append(EFT)
            processor = candidate_processors[int(np.argmin(EFT_list))]
            self.task_mapping[task] = processor
            if processor < Task.CPU_number + Task.NPU_number:
                if processor < Task.CPU_number:
                    exe = Task.app_list[i].sw_exe[j]
                else:
                    exe = Task.app_list[i].npu_exe[j]
                self.task_start_time[task] = max(predecessor_end_time, self.processor_avail[processor])
            else:
                DRP = True
                if self.processor_completed_task[processor]:
                    task_last = self.processor_completed_task[processor][-1]
                    i2, q2, j2 = calc_subtask_instance_from_position(Task,task_last)
                    if i2 == i and j2 == j:
                        DRP = False
                DRP_end_time = self.processor_avail[processor]
                if DRP:
                    DRP_start_time = max(self.processor_avail[processor], self.processor_RP_config_time)
                    DRP_end_time = DRP_start_time + Task.app_list[i].tpr[j]
                    self.task_DPR[task] = 1 # 任务是否重构
                    self.task_DPR_start_time[task] = DRP_start_time  # 任务重构开始时间
                    self.task_DPR_end_time[task] = DRP_end_time  # 任务重构结束时间
                    self.processor_RP_config_time = DRP_end_time  # 重构配置接口的空余时间
                    self.processor_RP_task_list.append(task)  # 记录重构的任务
                    self.processor_RP_time_list.append([DRP_start_time, DRP_end_time])  # 记录每次重构的[开始时间,截止时间]
                exe = Task.app_list[i].hw_exe[j]
                self.task_start_time[task] = max(predecessor_end_time, DRP_end_time)
            self.task_end_time[task] = self.task_start_time[task] + exe
            self.processor_completed_task[processor].append(task)
            self.processor_avail[processor] = self.task_end_time[task]  # 处理器空闲时间
        assert_mapping_eligible(Task, self.task_mapping, context="heft_scheduling")

    def heft_scheduling_for_certain_task_mapping(self,Task):
        self.clear_scheduling(Task.task_number)
        assert_mapping_eligible(Task, self.task_mapping, context="heft_scheduling_for_certain_task_mapping")
        # 生成任务优先级
        priority_list = priority_list_generate(Task)
        task_list = list(range(Task.task_number))
        priority_list_sorted, task_list_sorted = sort_lists(priority_list, task_list)
        for idx, task in enumerate(task_list_sorted):
            i, q, j = calc_subtask_instance_from_position(Task,task)
            arr = Task.instance_dict_list[i][q]["arr"]
            predecessor_end_time = arr
            predecessor = Task.app_list[i].predecessor_of_node(j)
            if predecessor:
                predecessor = [calc_subtask_instance_position(Task,i, q, j2) for j2 in predecessor]
                predecessor_end_time_list = [self.task_end_time[pre_task] for pre_task in predecessor]
                predecessor_end_time = max(predecessor_end_time_list)
            processor = self.task_mapping[task]
            if processor < Task.CPU_number + Task.NPU_number:
                if processor < Task.CPU_number:
                    exe = Task.app_list[i].sw_exe[j]
                else:
                    exe = Task.app_list[i].npu_exe[j]
                self.task_start_time[task] = max(predecessor_end_time, self.processor_avail[processor])
            else:
                DRP = True
                if self.processor_completed_task[processor]:
                    task_last = self.processor_completed_task[processor][-1]
                    i2, q2, j2 = calc_subtask_instance_from_position(Task,task_last)
                    if i2 == i and j2 == j:
                        DRP = False
                DRP_end_time = self.processor_avail[processor]
                if DRP:
                    DRP_start_time = max(self.processor_avail[processor], self.processor_RP_config_time)
                    DRP_end_time = DRP_start_time + Task.app_list[i].tpr[j]
                    self.task_DPR[task] = 1  # 任务是否重构
                    self.task_DPR_start_time[task] = DRP_start_time  # 任务重构开始时间
                    self.task_DPR_end_time[task] = DRP_end_time  # 任务重构结束时间
                    self.processor_RP_config_time = DRP_end_time  # 重构配置接口的空余时间
                    self.processor_RP_task_list.append(task)  # 记录重构的任务
                    self.processor_RP_time_list.append([DRP_start_time, DRP_end_time])  # 记录每次重构的[开始时间,截止时间]
                exe = Task.app_list[i].hw_exe[j]
                self.task_start_time[task] = max(predecessor_end_time, DRP_end_time)
            self.task_end_time[task] = self.task_start_time[task] + exe
            self.processor_completed_task[processor].append(task)
            self.processor_avail[processor] = self.task_end_time[task]  # 处理器空闲时间
        assert_mapping_eligible(Task, self.task_mapping, context="heft_decoder_result")


    def ecs_scheduling(self,Task:TaskConfig):
        self.clear_scheduling(Task.task_number)
        # 生成任务优先级
        priority_list = priority_list_generate(Task)
        task_list = list(range(Task.task_number))
        priority_list_sorted, task_list_sorted = sort_lists(priority_list, task_list)
        for idx, task in enumerate(task_list_sorted):
            i,q,j = calc_subtask_instance_from_position(Task,task)
            arr = Task.instance_dict_list[i][q]["arr"]
            predecessor_end_time = arr
            predecessor = Task.app_list[i].predecessor_of_node(j)
            if predecessor:
                predecessor = [calc_subtask_instance_position(Task,i,q,j2) for j2 in predecessor]
                predecessor_end_time_list = [self.task_end_time[pre_task] for pre_task in predecessor]
                predecessor_end_time = max(predecessor_end_time_list)
            candidate_processors = eligible_processor_ids(Task, task)
            best_processor = candidate_processors[0]
            if best_processor < Task.CPU_number:
                exe = Task.app_list[i].sw_exe[j]
            elif best_processor < Task.CPU_number + Task.NPU_number:
                exe = Task.app_list[i].npu_exe[j]
            else:
                exe = Task.app_list[i].hw_exe[j]
            best_EST = max(predecessor_end_time, self.processor_avail[best_processor])
            best_EFT = best_EST + exe
            best_E = Task.app_list[i].P_sw[j] * exe
            for processor in candidate_processors:
                if processor < Task.CPU_number:
                    exe = Task.app_list[i].sw_exe[j]
                    EST = max(predecessor_end_time, self.processor_avail[processor])
                    EFT = EST + exe
                    E = Task.app_list[i].P_sw[j] * exe
                elif processor <Task.CPU_number + Task.NPU_number:
                    exe = Task.app_list[i].npu_exe[j]
                    EST = max(predecessor_end_time, self.processor_avail[processor])
                    EFT = EST + exe
                    E = Task.app_list[i].P_npu[j] * exe
                else:
                    DRP = True
                    if self.processor_completed_task[processor]:
                        task_last = self.processor_completed_task[processor][-1]
                        i2, q2, j2 = calc_subtask_instance_from_position(Task,task_last)
                        if i2==i and j2==j:
                            DRP = False
                    DRP_end_time = self.processor_avail[processor]
                    if DRP:
                        DRP_start_time = max(self.processor_avail[processor], self.processor_RP_config_time)
                        DRP_end_time = DRP_start_time + Task.app_list[i].tpr[j]
                    exe = Task.app_list[i].hw_exe[j]
                    EST = max(predecessor_end_time, DRP_end_time)
                    EFT = EST + exe
                    E = (Task.app_list[i].P_dyn[j] + Task.app_list[i].P_static[j]) * exe
                RS1 = -((best_E-E)/(best_E+1e-6) +((best_EFT-EFT)/(best_EFT-min(best_EST,EST)+1e-6)))
                RS2 = -((E - best_E)/(E+1e-6) +((EFT - best_EFT)/(EFT-min(best_EST,EST)+1e-6)))
                # print(RS1,RS2)
                if RS1 < RS2:
                    best_processor = processor
                    best_EST = EST
                    best_EFT = EFT
                    best_E = E
            processor = best_processor
            # print(i,q,j,processor)
            self.task_mapping[task] = processor
            if processor < Task.CPU_number + Task.NPU_number:
                if processor < Task.CPU_number:
                    exe = Task.app_list[i].sw_exe[j]
                else:
                    exe = Task.app_list[i].npu_exe[j]
                self.task_start_time[task] = max(predecessor_end_time, self.processor_avail[processor])
            else:
                DRP = True
                if self.processor_completed_task[processor]:
                    task_last = self.processor_completed_task[processor][-1]
                    i2, q2, j2 = calc_subtask_instance_from_position(Task,task_last)
                    if i2 == i and j2 == j:
                        DRP = False
                DRP_end_time = self.processor_avail[processor]
                if DRP:
                    DRP_start_time = max(self.processor_avail[processor], self.processor_RP_config_time)
                    DRP_end_time = DRP_start_time + Task.app_list[i].tpr[j]
                    self.task_DPR[task] = 1  # 任务是否重构
                    self.task_DPR_start_time[task] = DRP_start_time  # 任务重构开始时间
                    self.task_DPR_end_time[task] = DRP_end_time  # 任务重构结束时间
                    self.processor_RP_config_time = DRP_end_time  # 重构配置接口的空余时间
                    self.processor_RP_task_list.append(task)  # 记录重构的任务
                    self.processor_RP_time_list.append([DRP_start_time, DRP_end_time])  # 记录每次重构的[开始时间,截止时间]
                exe = Task.app_list[i].hw_exe[j]
                self.task_start_time[task] = max(predecessor_end_time, DRP_end_time)
            self.task_end_time[task] = self.task_start_time[task] + exe
            self.processor_completed_task[processor].append(task)
            self.processor_avail[processor] = self.task_end_time[task]  # 处理器空闲时间
        # print(f"esc initial energy:{self.calc_energy(Task)}")
        # 优化排序
        for i3 in range(len(Task.app_list)):
            for q3 in range(Task.instance_number_list[i3]):
                for j3 in range(Task.app_list[i3].node_num):
                    pos =calc_subtask_instance_position(Task,i3,q3,j3)
                    for processor2 in eligible_processor_ids(Task, pos):
                        task_mapping = self.task_mapping.copy()
                        task_DPR = self.task_DPR.copy() # 任务是否重构
                        task_DPR_start_time = self.task_DPR_start_time.copy()  # 任务重构开始时间
                        task_DPR_end_time = self.task_DPR_end_time.copy()  # 任务重构结束时间
                        processor_RP_task_list = self.processor_RP_task_list.copy()  # 记录重构的任务
                        processor_RP_time_list = self.processor_RP_time_list.copy()  # 记录每次重构的[开始时间,截止时间]
                        task_start_time = self.task_start_time.copy()
                        task_end_time = self.task_end_time.copy()
                        processor_completed_task = self.processor_completed_task.copy()
                        energy = self.calc_energy(Task)
                        delay = self.calc_delay(Task)
                        self.clear_scheduling(Task.task_number)
                        self.task_mapping = task_mapping.copy()
                        self.task_mapping[pos] = processor2
                        assert_mapping_eligible(Task, self.task_mapping, context="ecs_candidate")
                        for idx, task in enumerate(task_list_sorted):
                            i, q, j = calc_subtask_instance_from_position(Task,task)
                            arr = Task.instance_dict_list[i][q]["arr"]
                            predecessor_end_time = arr
                            predecessor = Task.app_list[i].predecessor_of_node(j)
                            if predecessor:
                                predecessor = [calc_subtask_instance_position(Task,i, q, j2) for j2 in predecessor]
                                predecessor_end_time_list = [self.task_end_time[pre_task] for pre_task in predecessor]
                                predecessor_end_time = max(predecessor_end_time_list)
                            processor = self.task_mapping[task]
                            if processor < Task.CPU_number + Task.NPU_number:
                                if processor < Task.CPU_number:
                                    exe = Task.app_list[i].sw_exe[j]
                                else:
                                    exe = Task.app_list[i].npu_exe[j]
                                self.task_start_time[task] = max(predecessor_end_time,
                                                                 self.processor_avail[processor])
                            else:
                                DRP = True
                                if self.processor_completed_task[processor]:
                                    task_last = self.processor_completed_task[processor][-1]
                                    i2, q2, j2 = calc_subtask_instance_from_position(Task,task_last)
                                    if i2 == i and j2 == j:
                                        DRP = False
                                DRP_end_time = self.processor_avail[processor]
                                if DRP:
                                    DRP_start_time = max(self.processor_avail[processor],
                                                         self.processor_RP_config_time)
                                    DRP_end_time = DRP_start_time + Task.app_list[i].tpr[j]
                                    self.task_DPR[task] = 1  # 任务是否重构
                                    self.task_DPR_start_time[task] = DRP_start_time  # 任务重构开始时间
                                    self.task_DPR_end_time[task] = DRP_end_time  # 任务重构结束时间
                                    self.processor_RP_config_time = DRP_end_time  # 重构配置接口的空余时间
                                    self.processor_RP_task_list.append(task)  # 记录重构的任务
                                    self.processor_RP_time_list.append(
                                        [DRP_start_time, DRP_end_time])  # 记录每次重构的[开始时间,截止时间]
                                exe = Task.app_list[i].hw_exe[j]
                                self.task_start_time[task] = max(predecessor_end_time, DRP_end_time)
                            self.task_end_time[task] = self.task_start_time[task] + exe
                            self.processor_completed_task[processor].append(task)
                            self.processor_avail[processor] = self.task_end_time[task]  # 处理器空闲时间
                        if self.calc_delay(Task) > delay or (self.calc_delay(Task) == delay and self.calc_energy(Task) > energy):
                            self.task_mapping = task_mapping
                            self.task_DPR = task_DPR
                            self.task_DPR_start_time =task_DPR_start_time   # 任务重构开始时间
                            self.task_DPR_end_time = task_DPR_end_time  # 任务重构结束时间
                            self.processor_RP_task_list = processor_RP_task_list  # 记录重构的任务
                            self.processor_RP_time_list = processor_RP_time_list # 记录每次重构的[开始时间,截止时间]
                            self.task_start_time = task_start_time
                            self.task_end_time = task_end_time
                            self.processor_completed_task = processor_completed_task
        assert_mapping_eligible(Task, self.task_mapping, context="ecs_scheduling")


    def visualize_init(self, Task:TaskConfig, window_name="initial scheduler", is_block=False, is_save=False):
        ### 绘制调度结果
        # 随机生成每个应用的颜色
        self.app_color_list = [colors[i] for i in range(len(Task.app_list) + 1)]
        # print(f"application colors:{app_color_list}")
        subtask_colors = [0] * Task.task_number
        position = 0
        subtask_labels = [""] * Task.task_number
        subtask_RP_info = [[0, 0, 0] for _ in range(Task.task_number)]
        for i in range(len(Task.app_list)):
            for q in range(Task.instance_number_list[i]):
                for j in range(Task.app_list[i].node_num):
                    if self.task_DPR[position]:
                        subtask_RP_info[position][0] = 1
                        subtask_RP_info[position][1] = self.task_DPR_start_time[position]
                        subtask_RP_info[position][2] = self.task_DPR_start_time[position] + Task.app_list[i].tpr[j]
                    subtask_colors[position] = self.app_color_list[i]
                    subtask_labels[position] = f"({i},{q + 1},{j})"
                    position += 1
        self.plotter = ProcessorTimelinePlotter(Task.processor_number, Task.CPU_number, Task.NPU_number,
                                           Task.RP_number, time_axis_length=Task.H)
        self.task_blocks = [
            {'processor': self.task_mapping[i], 'start': self.task_start_time[i], 'end': self.task_end_time[i],
             'color': subtask_colors[i], 'label': subtask_labels[i], 'RP': subtask_RP_info[i]}
            for i in range(Task.task_number)
        ]
        # print(f"task blocks:{self.task_blocks}")
        self.plotter.set(window_name, task_blocks=self.task_blocks, is_block=is_block, is_save=is_save)

    def visualize(self, window_name="scheduler", is_block=False, is_save=False):
        self.plotter.set(window_name, task_blocks=self.task_blocks, is_block=is_block, is_save=is_save)

    # 计算延迟（不满足截止时间的）
    def calc_delay(self,Task:TaskConfig):
        time_list = []
        for i in range(len(Task.app_list)):
            for q in range(Task.instance_number_list[i]):
                time_list.append(judge_instance_ddl(Task,i, q, self.task_end_time))
        return sum(time_list)

    # 计算调度方案的能耗
    def calc_energy(self,Task:TaskConfig):
        sum = 0
        for processor in range(Task.processor_number):
            if processor < Task.CPU_number:
                # print(self.processor_completed_task[processor])
                sum += calc_CPU_dynamic_energy(Task,self.task_start_time, self.task_end_time, self.processor_completed_task[processor])
            elif processor < Task.CPU_number + Task.NPU_number:
                sum += calc_NPU_dynamic_energy(Task,self.task_start_time, self.task_end_time, self.processor_completed_task[processor])
            else:
                sum += calc_RP_dynamic_energy(
                    Task,
                    self.task_start_time,
                    self.task_end_time,
                    self.processor_completed_task[processor],
                    self.task_DPR_start_time,
                    self.task_DPR_end_time,
                    self.task_DPR,
                )
        return sum
    def energy_optimization(self,Task:TaskConfig):
        # self.visualize_init()
        ### 优化能耗
        RP_task_list = sum(self.processor_completed_task[-Task.RP_number:], [])  # 在fpga上调度的任务list
        RP_task_list = [task for task in RP_task_list if self.task_DPR[task]]
        # print(RP_task_list)
        # 获取任务的静态能耗比例list
        P_ratio_list = []
        for task in RP_task_list:
            task_index = self.processor_completed_task[self.task_mapping[task]].index(task)
            i, q, j = calc_subtask_instance_from_position(Task,task)
            curr_P = Task.app_list[i].P_static[j]
            if task_index == 0:
                prev_task = self.processor_completed_task[self.task_mapping[task]][-1]
                i_prev, q_prev, j_prev = calc_subtask_instance_from_position(Task,prev_task)
                prev_P = Task.app_list[i_prev].P_static[j_prev]
            else:
                prev_task = self.processor_completed_task[self.task_mapping[task]][task_index - 1]
                i_prev, q_prev, j_prev = calc_subtask_instance_from_position(Task,prev_task)
                prev_P = Task.app_list[i_prev].P_static[j_prev]
            P_ratio_list.append(curr_P / prev_P)
        # print(P_ratio_list)
        P_ratio_list, RP_task_list = sort_lists(P_ratio_list, RP_task_list)
        # print(P_ratio_list)
        # print(RP_task_list)
        flag = True
        while flag:
            flag1 = False
            for index, task in enumerate(RP_task_list):
                # print(f"任务{task},能耗比例{P_ratio_list[index]}")
                if P_ratio_list[index] > 1:
                    # 只是为了可视化
                    i, q, j = calc_subtask_instance_from_position(Task,task)
                    # print(i,q,j)
                    # for k in range(Task.app_list[i].node_num):
                        # if calc_subtask_instance_position(Task, i,q,j) in RP_task_list:
                        #     print(calc_subtask_instance_position(Task, i,q,j), P_ratio_list[RP_task_list.index(calc_subtask_instance_position(Task,i,q,j))])
                    #     self.task_blocks[calc_subtask_instance_position(Task,i, q, k)]['color'] = self.app_color_list[-1]
                    # self.plotter.set(f"initial scheduling {task}", task_blocks=self.task_blocks, is_block=True)
                    ## 调整执行时间，重构时间
                    # 调整任务图后续任务的执行时间（只调整不在fpga执行的任务）
                    affected_task_list = list_affected_tasks(Task,task, reverse=True)
                    affected_task_list = [affected_task for affected_task in affected_task_list if
                                          self.task_mapping[affected_task] < Task.CPU_number + Task.NPU_number]
                    for affected_task in affected_task_list:
                        # 计算每个受影响任务的最晚执行时间
                        affected_task_LFT = calc_ALFT(Task,affected_task,
                                                      self.processor_completed_task[self.task_mapping[affected_task]],
                                                      self.task_start_time)
                        # print(affected_task, affected_task_LFT)
                        if affected_task_LFT != self.task_end_time[affected_task]:
                            task_start_time_copy, task_end_time_copy = self.task_start_time[affected_task], self.task_end_time[
                                affected_task]
                            self.task_end_time[affected_task] = affected_task_LFT
                            task_exe = task_end_time_copy - task_start_time_copy
                            self.task_start_time[affected_task] = affected_task_LFT - task_exe
                            # 只是为了可视化
                            # print(
                            #     f"任务{affected_task}可调整时间，原执行区间[{task_start_time_copy},{task_end_time_copy}]，"
                            #     f"现执行区间可调整为[{self.task_start_time[affected_task]},{self.task_end_time[affected_task]}]")
                            # self.task_blocks[affected_task]["start"], self.task_blocks[affected_task]["end"] = self.task_start_time[
                            #     affected_task], self.task_end_time[affected_task]
                            # self.plotter.set(f"initial scheduling {task}-{affected_task}", task_blocks=self.task_blocks, is_block=True)
                    # 最晚结束时间是ddl，任务图后续节点开始时间，同一RP区下一任务开始时间的最小值
                    LFT = calc_ALFT(Task,task, self.processor_completed_task[self.task_mapping[task]], self.task_start_time,
                                    self.task_DPR_start_time, task_DPR=self.task_DPR, mode_DPR=True)
                    # 如果计算所得的任务最晚结束时间大于当前任务的实际结束时间，则考虑改变任务执行时间和任务重构时间
                    if self.task_end_time[task] <= LFT:
                        # print(f"task{task}({i},{q},{j})可以调整重构时间！{LFT}")
                        # print(self.processor_RP_task_list)
                        # print(self.processor_completed_task[self.task_mapping[task]])
                        LST = LFT - (self.task_end_time[task] - self.task_start_time[task])# 任务执行的最晚开始时间（任务重构的最晚结束时间）
                        # 计算前一任务的执行结束时间，缩小重构时间的可行区间
                        task_index = self.processor_completed_task[self.task_mapping[task]].index(task)
                        if task_index != 0:
                            prev_task = self.processor_completed_task[self.task_mapping[task]][task_index - 1]
                            # i_prev,q_prev,j_prev=calc_subtask_instance_from_position(Task,prev_task)
                            # print(prev_task,i_prev,q_prev,j_prev)
                            prev_task_end_time = self.task_end_time[prev_task]
                        else:
                            prev_task_end_time = 0
                        # 删除现有的重构时间，在可行的重构时间内寻找最靠后的时间
                        # print(self.processor_completed_task[self.task_mapping[task]])
                        # print(self.processor_RP_time_list[self.processor_RP_task_list.index(task)])
                        # print(prev_task_end_time)
                        # print(LST)
                        # print(self.processor_RP_time_list)
                        time_to_remove = self.processor_RP_time_list[self.processor_RP_task_list.index(task)]  # 获取对应的时间
                        self.processor_RP_time_list.remove(time_to_remove)  # 通过值删除
                        self.processor_RP_task_list.remove(task)
                        # print(self.processor_RP_time_list)
                        RP_interval = calc_last_RP_interval(self.processor_RP_time_list, prev_task_end_time, LST,
                                                            Task.app_list[i].tpr[j])
                        # if RP_interval == None:
                        #     RP_interval = time_to_remove
                        # print(self.processor_RP_time_list, prev_task_end_time, LST,
                        #                                     Task.app_list[i].tpr[j])
                        # print(RP_interval)
                        if RP_interval[0] != self.task_DPR_start_time[task]:
                            # print(
                            #     f"任务{task}原重构时间：[{self.task_DPR_start_time[task]},{self.task_DPR_start_time[task] + Task.app_list[i].tpr[j]}],新重构时间：{RP_interval}")
                            # print(
                            #     f"任务{task}原执行时间：[{self.task_start_time[task]},{self.task_end_time[task]}],新执行时间：[{LST},{LFT}]")
                            self.task_start_time[task], self.task_end_time[task] = LST, LFT
                            self.task_DPR_start_time[task] = RP_interval[0]
                            self.task_DPR_end_time[task] = RP_interval[1]
                            # self.processor_completed_task[self.task_mapping[task]].sort(key=lambda tt: self.task_start_time[tt])
                            # 只是为了可视化
                            # self.task_blocks[task]["start"], self.task_blocks[task]["end"] = self.task_start_time[task], self.task_end_time[
                            #     task]
                            # self.task_blocks[task]["RP"] = [1, self.task_DPR_start_time[task],
                            #                            self.task_DPR_start_time[task] + Task.app_list[i].tpr[j]]
                            
                            flag1 = True
                        # 重新插入重构区间
                        insert_RP_interval(self.processor_RP_time_list, RP_interval, self.processor_RP_task_list, task)
                        # print(self.processor_RP_time_list)
                    # 只是为了可视化
                    # self.plotter.set(f"initial scheduling {task}", task_blocks=self.task_blocks, is_block=True)
                    # for j in range(Task.app_list[i].node_num):
                    #     self.task_blocks[calc_subtask_instance_position(Task,i, q, j)]['color'] = self.app_color_list[i]
                    
                    if flag1:
                            break
                else:
                    # print(f"任务{task}能耗比小于1")
                    break
            if flag1:
                continue
            else:
                flag = False

        # self.plotter.set(f"initial scheduling final", task_blocks=self.task_blocks, is_save=True)
        energy = self.calc_energy(Task)
        # print(f"优化后能耗为{energy}")
