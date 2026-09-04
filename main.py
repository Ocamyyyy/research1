from DAGs import DAG
from time_slot_plot import ProcessorTimelinePlotter, colors
import math
from functools import reduce
import random

app_list = []
instance_number_list = []
instance_dict_list = {}

# 计算超周期
def calc_superperiod(application_list:list[DAG]):
    periods = [application.period for application in application_list]
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    return reduce(lcm, periods)


# 计算子任务实例在决策变量中的位置
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

def calc_max_predessor_AFT(app_index,instance_index,predecessor_list,CPU_number,NPU_number,mapping_list,t_s_list):
    predecessor_AFT_list = []
    max_AFT = 0
    if not predecessor_list:
        return max_AFT,predecessor_AFT_list
    else:
        for task in predecessor_list:
            pos = calc_subtask_instance_position(app_index,instance_index,task)
            AFT = 0
            if mapping_list[pos] < CPU_number:
                AFT = t_s_list[pos]+app_list[app_index].sw_exe[task]
            elif mapping_list[pos] < CPU_number + NPU_number:
                AFT = t_s_list[pos] + app_list[app_index].npu_exe[task]
            else:
                AFT = t_s_list[pos] + app_list[app_index].hw_exe[task]
            if AFT > max_AFT:
                max_AFT = AFT
                predecessor_AFT_list.clear()
                predecessor_AFT_list.append(pos)
            elif AFT == max_AFT:
                predecessor_AFT_list.append(pos)
        return max_AFT, predecessor_AFT_list

def check_same_subtask_mapping_successively(app_index, instance_index, subtask_index, processor_list, RPtime_list):
    flag = False
    pos = calc_subtask_instance_position(app_index, instance_index, subtask_index)
    max_prev_time = 0
    prev_subtask = 0
    prev_subtask_flag = False
    for subtask in range(len(processor_list)):
        if processor_list[subtask] == processor_list[pos] and RPtime_list[subtask] < RPtime_list[pos]:
            if RPtime_list[subtask] > max_prev_time:
                max_prev_time = RPtime_list[subtask]
                prev_subtask = subtask
                prev_subtask_flag = True
    if prev_subtask_flag:
        prev_i,_,prev_j = calc_subtask_instance_from_position(prev_subtask)
        if prev_i == app_index and prev_j==subtask_index:
            flag = True
    return flag, prev_subtask


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
    sorted_pairs = sorted(zipped_lists, key=lambda x: (x[0], x[1]))


    # 解包排序后的元组列表，得到排序后的 RPpr_list 和 subtask_list
    if sorted_pairs:
        RPpr_list_sorted, subtask_list_sorted = zip(*sorted_pairs)
        # 将结果转换回列表
        RPpr_list_sorted = list(RPpr_list_sorted)
        subtask_list_sorted = list(subtask_list_sorted)
    else:
        RPpr_list_sorted, subtask_list_sorted = [], []  # 或者根据你的需求处理空情况

    return RPpr_list_sorted, subtask_list_sorted

def RPpr_adjust(mapping_subtask_list_sorted, RPpr, check_same_subtask_mapping_successively_list):
    """
    检查是否存在同一任务的实例在前后执行，若存在则调整RPpr时间
    """
    for subtask in mapping_subtask_list_sorted:
        check_same_subtask_mapping_successively_list[subtask] = [0, -1]
    for m in range(1, len(mapping_subtask_list_sorted)):
        i_prev2, q_prev2, j_prev2 = calc_subtask_instance_from_position(mapping_subtask_list_sorted[m - 1])
        i2, q2, j2 = calc_subtask_instance_from_position(mapping_subtask_list_sorted[m])
        if i_prev2 == i2 and j_prev2 == j2:
            print(
                f"same mapping for same subtask! subtask pos is {mapping_subtask_list_sorted[m]}, "
                f"prev_pos is {mapping_subtask_list_sorted[m - 1]}")
            # 调整其 RPpr 为前一个实例的 RPpr
            RPpr[mapping_subtask_list_sorted[m]] = RPpr[mapping_subtask_list_sorted[m - 1]]
            if check_same_subtask_mapping_successively_list[subtask_list_sorted[m - 1]][0] == 2:
                check_same_subtask_mapping_successively_list[subtask_list_sorted[m]] = [2,check_same_subtask_mapping_successively_list[subtask_list_sorted[m -1]][1]]
            else:
                # 更新 check_same_subtask_mapping_successively_list
                check_same_subtask_mapping_successively_list[mapping_subtask_list_sorted[m]] = [2, mapping_subtask_list_sorted[m - 1]]
                check_same_subtask_mapping_successively_list[mapping_subtask_list_sorted[m - 1]] = [1, mapping_subtask_list_sorted[m]]

if __name__== "__main__" :
    ### 定义应用集
    generate_dict_list = []
    generate_dict_list.append({"node num": 7, "edges": [(0, 4), (1, 4), (2, 3), (2, 5), (3, 6), (4, 6)], "period": 20,
                "sw exe": [4] * 7, "npu exe": [5] * 7, "hw exe": [3] * 7, "tpr": [2] * 7,
                "P static":[30] * 7, "P dyn":[6] * 7})
    generate_dict_list.append({"node num": 7, "edges": [(0, 4), (1, 4), (2, 3), (2, 5), (3, 6), (4, 6)], "period": 30,
                               "sw exe": [3] * 7, "npu exe": [1,1,2,1,1,1,1], "hw exe": [2] * 7, "tpr": [1] * 7,
                               "P static": [30] * 7, "P dyn": [6] * 7})

    for generate_dict in generate_dict_list:
        app_list.append(DAG(generate_dict))

    ### 定义调度周期
    H = calc_superperiod(app_list)  # 计算超周期
    instance_number_list = [int(H / app.period) for app in app_list]  # 计算每个应用的实例个数
    # 生成实例的到达时间、截止时间和所有子任务的EST和LST
    for i in range(len(app_list)):
        instance_dict_list[i] = {}
        for q in range(instance_number_list[i]):
            instance_dict_list[i][q + 1] = {}
            instance_dict_list[i][q + 1]["arr"] = q * app_list[i].period
            instance_dict_list[i][q + 1]["ddl"] = (q + 1) * app_list[i].period
            instance_dict_list[i][q + 1]["EST"] = list(map(lambda x: x + q * app_list[i].period, app_list[i].EST))
            instance_dict_list[i][q + 1]["LST"] = list(map(lambda x: x + (q + 1) * app_list[i].period, app_list[i].LST))
    print(instance_dict_list)

    ### 定义处理器
    CPU_number = 4
    NPU_number = 1
    RP_number = 4
    processor_number = CPU_number + NPU_number + RP_number

    ### 定义决策变量
    subtask_number = 0                          # 子任务实例总数
    for i in range(len(app_list)):
        subtask_number += instance_number_list[i] * app_list[i].node_num
    processor_mapping = [0] * subtask_number    # 处理器映射
    t_s = [0] * subtask_number                  # 开始执行时间
    RPpr = [0] * subtask_number                 # 开始重构时间

    ### 生成随机解
    position = 0
    for i in range(len(app_list)):
        for q in range(instance_number_list[i]):
            for j in range(app_list[i].node_num):
                processor_mapping[position] = random.randrange(processor_number)
                t_s[position] = random.randint(instance_dict_list[i][q + 1]["EST"][j],
                                               instance_dict_list[i][q + 1]["LST"][j])
                if processor_mapping[position] < CPU_number + NPU_number:   # 没有映射到FPGA的子任务的重构开始时间统一为0
                    RPpr[position] = 0
                else:
                    RPpr[position] = t_s[position] - app_list[i].tpr[j]
                    flag = False # 判断重构时间合理性
                    while not flag:
                        count1 = 0
                        for k in range(position):               # 重构开始时间与其他子任务同一时刻，重新生成子任务开始执行时间和重构开始时间
                            if processor_mapping[k] == processor_mapping[position] and RPpr[k] == RPpr[position]:
                                print(f"repeat RPpr time! subtask pos is {position}")
                                count1 += 1
                        if count1 == 0 and RPpr[position] >= 0:  # 重构不合理，重新生成子任务开始执行时间和重构开始时间
                            flag = True
                        else:
                            print(f"RPpr and start time resigned!subtask pos is {position}, RPpr is {RPpr[position]}")
                            t_s[position] = random.randint(instance_dict_list[i][q + 1]["EST"][j],
                                                           instance_dict_list[i][q + 1]["LST"][j])
                            RPpr[position] = t_s[position] - app_list[i].tpr[j]
                position += 1
    print(processor_mapping)
    print(t_s)
    print(RPpr)
    # 对于同一子任务在同一RP区先后进行的2个实例，调整其RPpr为前一个实例的RPpr
    check_same_subtask_mapping_successively_list = [[0, -1] for _ in range(subtask_number)]  # 0表示正常重构，1表示正常重构且其后重构任务为同一子任务实例，2表示无需重构且记录前一子任务实例
    for RP_index in range(CPU_number + NPU_number, processor_number):       # 逐RP区进行判断
        subtask_list = [subtask for subtask in range(len(processor_mapping)) if processor_mapping[subtask] == RP_index]
        RPpr_list = [RPpr[subtask] for subtask in subtask_list]
        RPpr_list_sorted, subtask_list_sorted = sort_lists(RPpr_list, subtask_list)
        print(RPpr_list_sorted)
        print(subtask_list_sorted)
        RPpr_adjust(subtask_list_sorted,RPpr,check_same_subtask_mapping_successively_list)
    print(f"initial mapping:{processor_mapping}")
    print(f"initial start time:{t_s}")
    print(f"initial reconfiguring start time:{RPpr}")
    print(check_same_subtask_mapping_successively_list)
    ### 判断解对约束条件的满足情况
    flag = False
    rectify_count = -1
    while not flag:
        rectify_count += 1
        print(f"修正次数：{rectify_count}")
        # 判断条件一：是否满足FPGA只能进行一个子任务的DRP
        # 若不符合条件，则随机选择修改前一任务的RPpr或后一任务的RPpr或2个任务的映射硬件类型(只能映射为CPU或NPU)
        condition1_flag = True  # 是否一次性满足判断条件1
        RP_index_list = list(range(CPU_number + NPU_number, processor_number))
        subtask_list = [subtask for subtask in range(len(processor_mapping)) if processor_mapping[subtask] in RP_index_list]
        RPpr_list = [RPpr[subtask] for subtask in subtask_list]
        RPpr_list_sorted, subtask_list_sorted = sort_lists(RPpr_list, subtask_list)
        print(subtask_list)
        # print(f"condition1 flag: {condition1_flag}")
        for k in range(1,len(subtask_list_sorted)):
            # 如果该子任务执行前需要重构，则需要满足重构开始时间大于前一重构任务的重构结束时间
            if check_same_subtask_mapping_successively_list[subtask_list_sorted[k]][0] < 2:
                i_prev,q_prev,j_prev = calc_subtask_instance_from_position(subtask_list_sorted[k - 1])
                i, q, j = calc_subtask_instance_from_position(subtask_list_sorted[k])
                # 不满足重构开始时间大于前一重构任务的重构结束时间
                if RPpr[subtask_list_sorted[k-1]] + app_list[i_prev].tpr[j_prev] > RPpr[subtask_list_sorted[k]]:
                    condition1_flag = False
                    print(f"DPR condition not met! subtasks are {subtask_list_sorted[k-1]} and {subtask_list_sorted[k]}")
                    change_choose_list = list(range(4))  # 记录随机修改的方案是否已经尝试
                    change_flag = False                  # 记录随机修改的方案是否成功
                    while not change_flag:
                        change_choose = random.choice(change_choose_list)
                        print(change_choose)
                        '''change_choose = 0：选择修改前一任务（后续条件称为当前任务）的RPpr，修改到DPR条件刚好成立，然后注意：
                        （1）修改当前任务的开始执行时间为子任务的EST和重构结束时间的较大值,
                        对比改变时间前后当前任务所在RP区的重构先后顺序是否发生变化，如果不变化，执行（2），如果变化，执行（3）
                        （2）如果当前任务所在RP区的下一个子任务实例是同一子任务，则递归修改下一子任务实例的重构时间，change_flag置为True
                        （3）重构顺序发生变化，对当前的RP区重新执行一轮RPpr
                        时间调整，change_flag置为True'''
                        if change_choose == 0:
                            change_choose_list.remove(change_choose)
                            RPpr_reserve = RPpr[subtask_list_sorted[k-1]]   # 保存变化前的值
                            pre_mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                            processor_mapping[subtask]==processor_mapping[subtask_list_sorted[k-1]]]
                            pre_mapping_RPpr_list = [RPpr[subtask] for subtask in pre_mapping_subtask_list]
                            pre_mapping_RPpr_list_sorted, pre_mapping_subtask_list_sorted \
                                = sort_lists(pre_mapping_RPpr_list, pre_mapping_subtask_list)
                            # 保存原先RPpr和所在区重构顺序后，修改RPpr
                            RPpr[subtask_list_sorted[k - 1]] = RPpr[subtask_list_sorted[k]] - app_list[i_prev].tpr[j_prev]
                            if RPpr[subtask_list_sorted[k - 1]] < 0:
                                RPpr[subtask_list_sorted[k - 1]] = RPpr_reserve
                                continue
                            mapping_RPpr_list = [RPpr[subtask] for subtask in pre_mapping_subtask_list]
                            mapping_RPpr_list_sorted, mapping_subtask_list_sorted \
                                = sort_lists(mapping_RPpr_list, pre_mapping_subtask_list)
                            t_s[subtask_list_sorted[k - 1]] \
                                = max([instance_dict_list[i_prev][q_prev]["EST"][j_prev],
                                       RPpr[subtask_list_sorted[k - 1]] + app_list[i_prev].tpr[j_prev]])
                            # 顺序不变，执行（2）
                            if mapping_subtask_list_sorted == pre_mapping_subtask_list_sorted:
                                print("rectify success!")
                                print(f"子任务所在RP区执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"子任务所在RP区的各子任务重构开始时间：{mapping_RPpr_list_sorted}")
                                change_flag = True
                                subtask_changed_index = mapping_subtask_list_sorted.index(subtask_list_sorted[k-1])
                                while check_same_subtask_mapping_successively_list[mapping_subtask_list_sorted[subtask_changed_index]][0] == 1:
                                    RPpr[mapping_RPpr_list_sorted[subtask_changed_index + 1]] = RPpr[subtask_list_sorted[k-1]]
                                    print("rectify successive subtask too!")
                                    print(f"子任务所在RP区的各子任务重构开始时间：{mapping_RPpr_list_sorted}")
                                    subtask_changed_index +=1
                            # 顺序改变，执行（3）
                            else:
                                print("RP sequence changed!")
                                print(f"原子任务所在RP区执行的任务为：{pre_mapping_subtask_list_sorted}")
                                print(f"现子任务所在RP区执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"现子任务所在RP区的各子任务重构开始时间：{mapping_RPpr_list_sorted}")
                                RPpr_adjust(mapping_subtask_list_sorted,RPpr,check_same_subtask_mapping_successively_list)
                                mapping_RPpr_list_sorted = [RPpr[subtask] for subtask in mapping_subtask_list_sorted]
                                print(f"现子任务所在RP区的各子任务重构开始时间：{mapping_RPpr_list_sorted}")
                                change_flag = True
                            '''change_choose = 1：选择修改后一任务（后续条件称为当前任务）的RPpr，修改到DPR条件刚好成立，然后注意：
                            （1）判断当前子任务的重构结束时间是否大于任务的LST，如果是，认为本变化不成立，选择其他修改方式；如果重构结束时间合理，执行（2）
                            （2）修改当前任务的开始执行时间为重构结束时间，对比改变时间前后当前任务所在RP区的重构先后顺序是否发生变化，如果不变化，执行（3），如果变化，执行（4）
                            （3）如果当前任务所在RP区的下一个子任务实例是同一子任务，则递归修改下一子任务实例的重构时间，change_flag置为True
                            （4）重构顺序发生变化，对当前的RP区重新执行一轮RPpr时间调整，change_flag置为True'''
                        elif change_choose == 1:
                            change_choose_list.remove(change_choose)
                            RPpr_reserve = RPpr[subtask_list_sorted[k]]  # 保存变化前的值
                            pre_mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == processor_mapping[
                                                            subtask_list_sorted[k]]]
                            pre_mapping_RPpr_list = [RPpr[subtask] for subtask in pre_mapping_subtask_list]
                            pre_mapping_RPpr_list_sorted, pre_mapping_subtask_list_sorted \
                                = sort_lists(pre_mapping_RPpr_list, pre_mapping_subtask_list)
                            # 保存原先RPpr和所在区重构顺序后，修改RPpr
                            RPpr[subtask_list_sorted[k]] = RPpr[subtask_list_sorted[k - 1]] + app_list[i_prev].tpr[
                                j_prev]
                            if RPpr[subtask_list_sorted[k]] + app_list[i].tpr[j] > instance_dict_list[i][q]["LST"][j]:
                                print("rectify fail!")
                                RPpr[subtask_list_sorted[k]] = RPpr_reserve
                                continue    # 当前子任务的重构结束时间是否大于任务的LST,认为本变化不成立，选择其他修改方式
                            mapping_RPpr_list = [RPpr[subtask] for subtask in pre_mapping_subtask_list]
                            mapping_RPpr_list_sorted, mapping_subtask_list_sorted \
                                = sort_lists(mapping_RPpr_list, pre_mapping_subtask_list)
                            t_s[subtask_list_sorted[k]] = RPpr[subtask_list_sorted[k]] + app_list[i].tpr[j]
                            # 顺序不变，执行（3）
                            if mapping_subtask_list_sorted == pre_mapping_subtask_list_sorted:
                                print("rectify success!")
                                print(f"子任务所在RP区执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"子任务所在RP区的各子任务重构开始时间：{mapping_RPpr_list_sorted}")
                                change_flag = True
                                subtask_changed_index = mapping_subtask_list_sorted.index(subtask_list_sorted[k])
                                while check_same_subtask_mapping_successively_list[
                                    mapping_subtask_list_sorted[subtask_changed_index]][0] == 1:
                                    RPpr[mapping_RPpr_list_sorted[subtask_changed_index + 1]] = RPpr[
                                        subtask_list_sorted[k]]
                                    print("rectify successive subtask too!")
                                    print(f"子任务所在RP区的各子任务重构开始时间：{mapping_RPpr_list_sorted}")
                                    subtask_changed_index += 1
                            # 顺序改变，执行（4）
                            else:
                                print("RP sequence changed!")
                                print(f"原子任务所在RP区执行的任务为：{pre_mapping_subtask_list_sorted}")
                                print(f"现子任务所在RP区执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"现子任务所在RP区的各子任务重构开始时间：{mapping_RPpr_list_sorted}")
                                RPpr_adjust(mapping_subtask_list_sorted,RPpr,check_same_subtask_mapping_successively_list)
                                mapping_RPpr_list_sorted = [RPpr[subtask] for subtask in mapping_subtask_list_sorted]
                                print(f"现子任务所在RP区的各子任务重构开始时间：{mapping_RPpr_list_sorted}")
                                change_flag = True
                            '''change_choose = 2：选择修改前一任务的映射（只能为CPU或NPU），然后注意：
                            （1）修改任务的重构开始时间为0，如果任务重构性质为1，注意修改下一子任务实例的重构开始时间
                            （2）将当前任务的性质修改为0，并重新调整原RP区，change_flag置为True'''
                        elif change_choose == 2:
                            change_choose_list.remove(change_choose)
                            mapping_index = processor_mapping[subtask_list_sorted[k - 1]]
                            RPpr[subtask_list_sorted[k - 1]] = 0
                            processor_mapping[subtask_list_sorted[k-1]] = random.randrange(CPU_number + NPU_number)
                            print(f"change processor to: {processor_mapping[subtask_list_sorted[k-1]]}")
                            if check_same_subtask_mapping_successively_list[subtask_list_sorted[k-1]][0] == 1:
                                next_subtask = check_same_subtask_mapping_successively_list[subtask_list_sorted[k-1]][1]
                                i_next,q_next,j_next = calc_subtask_instance_from_position(next_subtask)
                                RPpr[next_subtask] = t_s[next_subtask] - app_list[i_next].tpr[j_next]
                            pre_mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == mapping_index]
                            pre_mapping_RPpr_list = [RPpr[subtask] for subtask in pre_mapping_subtask_list]
                            pre_mapping_RPpr_list_sorted, pre_mapping_subtask_list_sorted \
                                = sort_lists(pre_mapping_RPpr_list, pre_mapping_subtask_list)
                            RPpr_adjust(pre_mapping_subtask_list_sorted, RPpr,
                                        check_same_subtask_mapping_successively_list)
                            print(f"子任务原先所在RP区执行的任务为：{pre_mapping_subtask_list_sorted}")
                            print(f"子任务原先所在RP区的各子任务重构开始时间：{pre_mapping_RPpr_list_sorted}")
                            check_same_subtask_mapping_successively_list[subtask_list_sorted[k - 1]] = [0, -1]
                            change_flag = True
                            '''change_choose = 3：选择修改后一任务的映射（只能为CPU或NPU），然后注意：
                            （1）修改任务的重构开始时间为0，如果任务重构性质为1，注意修改下一子任务实例的重构开始时间
                            （2）将当前任务的性质修改为0，并重新调整原RP区，change_flag置为True'''
                        else:
                            mapping_index = processor_mapping[subtask_list_sorted[k]]
                            RPpr[subtask_list_sorted[k]] = 0
                            processor_mapping[subtask_list_sorted[k]] = random.randrange(CPU_number + NPU_number)
                            print(f"change processor to: {processor_mapping[subtask_list_sorted[k]]}")
                            if check_same_subtask_mapping_successively_list[subtask_list_sorted[k]][0] == 1:
                                next_subtask = check_same_subtask_mapping_successively_list[subtask_list_sorted[k]][1]
                                i_next, q_next, j_next = calc_subtask_instance_from_position(next_subtask)
                                RPpr[next_subtask] = t_s[next_subtask] - app_list[i_next].tpr[j_next]
                            pre_mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == mapping_index]
                            pre_mapping_RPpr_list = [RPpr[subtask] for subtask in pre_mapping_subtask_list]
                            pre_mapping_RPpr_list_sorted, pre_mapping_subtask_list_sorted \
                                = sort_lists(pre_mapping_RPpr_list, pre_mapping_subtask_list)
                            RPpr_adjust(pre_mapping_subtask_list_sorted, RPpr,
                                        check_same_subtask_mapping_successively_list)
                            print(f"子任务原先所在RP区执行的任务为：{pre_mapping_subtask_list_sorted}")
                            print(f"子任务原先所在RP区的各子任务重构开始时间：{pre_mapping_RPpr_list_sorted}")
                            check_same_subtask_mapping_successively_list[subtask_list_sorted[k]] = [0, -1]
                            change_flag = True
                    break   # 有一个任务不满足，在修改后就要重新check
        if not condition1_flag:
            continue
        # 判断条件2：是否满足不同任务映射到同一可重构区，后执行的任务开始重构的时间大于先执行任务的结束时间(FPGA资源)
        # 若不符合条件，则随机选择修改前一任务的开始时间或后一任务的开始时间或2个任务的映射硬件类型
        condition2_flag = True  # 是否一次性满足判断条件2
        subtask_wrong_pos = 0
        for RP_index in range(CPU_number + NPU_number, processor_number):  # 逐RP区进行判断
            subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                            processor_mapping[subtask]==RP_index]
            RPpr_list = [RPpr[subtask] for subtask in subtask_list]
            if subtask_list:
                RPpr_list_sorted, subtask_list_sorted = sort_lists(RPpr_list, subtask_list)
                for k in range(1,len(subtask_list_sorted)):
                    i_prev, q_prev, j_prev = calc_subtask_instance_from_position(subtask_list_sorted[k - 1])
                    i, q, j = calc_subtask_instance_from_position(subtask_list_sorted[k])
                    if check_same_subtask_mapping_successively_list[subtask_list_sorted[k]][0] < 2:
                        if RPpr_list_sorted[k] < t_s[subtask_list_sorted[k - 1]] + app_list[i_prev].hw_exe[j_prev]:
                            condition2_flag = False
                            subtask_wrong_pos = k
                            print(f"FPGA资源要求不满足，子任务为{subtask_list_sorted[subtask_wrong_pos]}，前一个任务为{subtask_list_sorted[subtask_wrong_pos-1]}")
                            break
                    else:
                        if t_s[subtask_list_sorted[k]] < t_s[subtask_list_sorted[k - 1]] + app_list[i_prev].hw_exe[j_prev]:
                            condition2_flag = False
                            subtask_wrong_pos = k
                            print(f"FPGA资源要求不满足，子任务为{subtask_list_sorted[subtask_wrong_pos]}，前一个任务为{subtask_list_sorted[subtask_wrong_pos-1]}")
                            break
                if not condition2_flag:
                    change_choose_list = list(range(4))  # 记录随机修改的方案是否已经尝试
                    change_flag = False  # 记录随机修改的方案是否成功
                    while not change_flag:
                        change_choose = random.choice(change_choose_list)
                        print(change_choose)
                        i_prev, q_prev, j_prev = calc_subtask_instance_from_position(subtask_list_sorted[subtask_wrong_pos - 1])
                        i, q, j = calc_subtask_instance_from_position(subtask_list_sorted[subtask_wrong_pos])
                        '''change_choose = 0：选择修改前一任务的开始时间，修改到条件刚好成立，然后注意：
                        （1）修改当前任务的重构开始时间，若重构时间小于0，或者任务开始执行时间小于EST，则认为不合理,否则执行（2）
                        （2）对当前的RP区重新执行一轮RPpr时间调整，change_flag置为True'''
                        if change_choose == 0:
                            change_choose_list.remove(change_choose)
                            print(f"原子任务所在RP区执行的任务为：{subtask_list_sorted}")
                            print(f"原子任务所在RP区执行的各子任务重构开始时间：{RPpr_list_sorted}")
                            t_s_reserve = t_s[subtask_list_sorted[subtask_wrong_pos - 1]]
                            RPpr_reserve = RPpr[subtask_list_sorted[subtask_wrong_pos - 1]]
                            if check_same_subtask_mapping_successively_list[subtask_list_sorted[subtask_wrong_pos]][0] < 2:
                                t_s[subtask_list_sorted[subtask_wrong_pos - 1]] = RPpr[subtask_list_sorted[subtask_wrong_pos]]-app_list[i_prev].hw_exe[j_prev]
                                RPpr[subtask_list_sorted[subtask_wrong_pos - 1]]=t_s[subtask_list_sorted[subtask_wrong_pos - 1]]-app_list[i_prev].tpr[j_prev]
                            else:
                                t_s[subtask_list_sorted[subtask_wrong_pos - 1]] = t_s[subtask_list_sorted[
                                    subtask_wrong_pos]] - app_list[i_prev].hw_exe[j_prev]
                                RPpr[subtask_list_sorted[subtask_wrong_pos - 1]] = t_s[subtask_list_sorted[
                                    subtask_wrong_pos - 1]] - app_list[i_prev].tpr[j_prev]
                            if RPpr[subtask_list_sorted[subtask_wrong_pos - 1]] < 0 or t_s[subtask_list_sorted[subtask_wrong_pos - 1]]<instance_dict_list[i_prev][q_prev]['EST'][j_prev]:
                                print("任务重构开始时间不合理，重新选择修改方式！")
                                t_s[subtask_list_sorted[subtask_wrong_pos - 1]] = t_s_reserve
                                RPpr[subtask_list_sorted[subtask_wrong_pos - 1]] = RPpr_reserve
                                continue
                            RPpr_list = [RPpr[subtask] for subtask in subtask_list]
                            RPpr_list_sorted, subtask_list_sorted = sort_lists(RPpr_list, subtask_list)
                            print(f"现子任务所在RP区执行的任务为：{subtask_list_sorted}")

                            RPpr_adjust(subtask_list_sorted, RPpr, check_same_subtask_mapping_successively_list)
                            t_s_list_sorted= [t_s[subtask] for subtask in subtask_list_sorted]
                            print(f"现子任务所在RP区的各子任务重构开始时间：{RPpr_list_sorted}")
                            print(f"现子任务所在RP区的各子任务执行开始时间：{t_s_list_sorted}")
                            change_flag = True
                            '''change_choose = 1：选择修改后一任务的重构开始时间，修改到条件刚好成立，然后注意：
                            （1）修改当前任务的开始执行时间，若超出当前任务的LST，则认为不合理,否则执行change_flag置为True'''
                        elif change_choose == 1:
                            change_choose_list.remove(change_choose)
                            print(f"原子任务所在RP区执行的任务为：{subtask_list_sorted}")
                            print(f"原子任务所在RP区执行的各子任务重构开始时间：{RPpr_list_sorted}")
                            t_s_reserve = t_s[subtask_list_sorted[subtask_wrong_pos]]
                            RPpr_reserve = RPpr[subtask_list_sorted[subtask_wrong_pos]]
                            if check_same_subtask_mapping_successively_list[subtask_list_sorted[subtask_wrong_pos]][
                                0] < 2:
                                RPpr[subtask_list_sorted[subtask_wrong_pos]] = t_s[subtask_list_sorted[subtask_wrong_pos - 1]]+ app_list[i_prev].hw_exe[j_prev]
                                t_s[subtask_list_sorted[subtask_wrong_pos]] = RPpr[subtask_list_sorted[subtask_wrong_pos]] + app_list[i].tpr[j]
                            else:
                                t_s[subtask_list_sorted[subtask_wrong_pos]] = t_s[subtask_list_sorted[
                                    subtask_wrong_pos]] + app_list[i_prev].hw_exe[j_prev]
                            if t_s[subtask_list_sorted[subtask_wrong_pos]]>instance_dict_list[i][q]["LST"][j]:
                                print("任务重构开始时间不合理，重新选择修改方式！")
                                t_s[subtask_list_sorted[subtask_wrong_pos - 1]] = t_s_reserve
                                RPpr[subtask_list_sorted[subtask_wrong_pos - 1]] = RPpr_reserve
                                continue
                            else:
                                change_flag = True
                            '''change_choose = 2或3：选择修改一任务的映射，然后注意：
                            （1）如果是CPU或NPU，执行（2），否则执行（3）
                            （2）将当前任务的性质修改为0，修改任务的重构开始时间为0，如果任务重构性质为1，注意修改下一子任务实例的重构开始时间，执行（4）
                            （3）调整映射后的RP区
                            （4）并重新调整原RP区，change_flag置为True
                            '''
                        else:
                            if change_choose == 2:
                                subtask_wrong_pos = subtask_wrong_pos - 1
                            change_choose_list.remove(change_choose)
                            print(
                                f"原来映射所在处理器为：{processor_mapping[subtask_list_sorted[subtask_wrong_pos]]}")
                            mapping_reserve = processor_mapping[subtask_list_sorted[subtask_wrong_pos]]
                            check_same_subtask_mapping_successively_reserve = \
                            check_same_subtask_mapping_successively_list[subtask_list_sorted[subtask_wrong_pos]]
                            mapping_choice = list(range(processor_number))
                            mapping_choice.remove(processor_mapping[subtask_list_sorted[subtask_wrong_pos]])
                            processor_mapping[subtask_list_sorted[subtask_wrong_pos]] = random.choice(
                                mapping_choice)
                            print(
                                f"现在映射所在处理器为：{processor_mapping[subtask_list_sorted[subtask_wrong_pos]]}")
                            if processor_mapping[subtask_list_sorted[subtask_wrong_pos]] < CPU_number + NPU_number:
                                check_same_subtask_mapping_successively_list[
                                    subtask_list_sorted[subtask_wrong_pos]] = [0, -1]
                            else:
                                mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == processor_mapping[
                                                            subtask_list_sorted[subtask_wrong_pos]]]
                                mapping_t_s_list = [t_s[subtask] for subtask in mapping_subtask_list]
                                mapping_t_s_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_t_s_list, mapping_subtask_list)
                                print(f"子任务现在所在RP区执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"子任务现在所在RP区的各子任务开始执行时间：{mapping_t_s_list_sorted}")
                                # 根据执行时间进行重构时间调整
                                mapping_pos = mapping_subtask_list_sorted.index(
                                    subtask_list_sorted[subtask_wrong_pos])
                                if mapping_pos != len(mapping_subtask_list_sorted) - 1:
                                    next_subtask = mapping_subtask_list_sorted[mapping_pos + 1]
                                    if check_same_subtask_mapping_successively_list[next_subtask][0] == 2:
                                        i_next, q_next, j_next = calc_subtask_instance_from_position(next_subtask)
                                        RPpr[next_subtask] = t_s[next_subtask] - app_list[i_next].tpr[j_next]
                                i_now, q_now, j_now = calc_subtask_instance_from_position(
                                    subtask_list_sorted[subtask_wrong_pos])
                                RPpr_reserve = RPpr[subtask_list_sorted[subtask_wrong_pos]]
                                RPpr[subtask_list_sorted[subtask_wrong_pos]] = t_s[subtask_list_sorted[
                                    subtask_wrong_pos]] - app_list[i_now].tpr[j_now]
                                if RPpr[subtask_list_sorted[subtask_wrong_pos]] < 0:
                                    RPpr[subtask_list_sorted[subtask_wrong_pos]] = RPpr_reserve
                                    processor_mapping[subtask_list_sorted[subtask_wrong_pos]] = mapping_reserve
                                    continue
                                mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == processor_mapping[
                                                            subtask_list_sorted[subtask_wrong_pos]]]
                                mapping_RPpr_list = [RPpr[subtask] for subtask in mapping_subtask_list]
                                mapping_RPpr_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_RPpr_list, mapping_subtask_list)
                                RPpr_adjust(mapping_subtask_list_sorted, RPpr,
                                            check_same_subtask_mapping_successively_list)
                                mapping_t_s_list = [t_s[subtask] for subtask in mapping_subtask_list]
                                mapping_t_s_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_t_s_list, mapping_subtask_list)
                                print(f"子任务现在所在RP区执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"子任务现在所在RP区的各子任务开始执行时间：{mapping_t_s_list_sorted}")

                            if check_same_subtask_mapping_successively_reserve[0] == 1:
                                next_subtask = check_same_subtask_mapping_successively_reserve[1]
                                i_next, q_next, j_next = calc_subtask_instance_from_position(next_subtask)
                                RPpr[next_subtask] = t_s[next_subtask] - app_list[i_next].tpr[j_next]
                            pre_mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == mapping_reserve]
                            pre_mapping_RPpr_list = [RPpr[subtask] for subtask in pre_mapping_subtask_list]
                            pre_mapping_RPpr_list_sorted, pre_mapping_subtask_list_sorted \
                                = sort_lists(pre_mapping_RPpr_list, pre_mapping_subtask_list)
                            RPpr_adjust(pre_mapping_subtask_list_sorted, RPpr,
                                        check_same_subtask_mapping_successively_list)
                            print(f"子任务原先所在RP区执行的任务为：{pre_mapping_subtask_list_sorted}")
                            print(f"子任务原先所在RP区的各子任务重构开始时间：{pre_mapping_RPpr_list_sorted}")
                            change_flag = True
                    break
        if not condition2_flag:
            continue

        # 判断条件3：是否满足CPU资源约束
        # 若不符合条件，则随机选择修改前一任务的开始时间或后一任务的开始时间或2个任务的映射硬件类型
        condition3_flag = True  # 是否一次性满足判断条件2
        subtask_wrong_pos = 0
        for CPU_index in range(CPU_number):  # 逐CPU进行判断
            subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                            processor_mapping[subtask] == CPU_index]
            t_s_list = [t_s[subtask] for subtask in subtask_list]
            if subtask_list:
                t_s_list_sorted, subtask_list_sorted = sort_lists(t_s_list, subtask_list)
                for k in range(1, len(subtask_list_sorted)):
                    i_prev, q_prev, j_prev = calc_subtask_instance_from_position(subtask_list_sorted[k - 1])
                    i, q, j = calc_subtask_instance_from_position(subtask_list_sorted[k])
                    if t_s[subtask_list_sorted[k]] < t_s[subtask_list_sorted[k - 1]] + app_list[i_prev].sw_exe[j_prev]:
                        condition3_flag = False
                        subtask_wrong_pos = k
                        print(
                            f"CPU资源要求不满足，子任务为{subtask_list_sorted[subtask_wrong_pos]}，前一个任务为{subtask_list_sorted[subtask_wrong_pos - 1]}")
                        break
                if not condition3_flag:
                    change_choose_list = list(range(4))  # 记录随机修改的方案是否已经尝试
                    change_flag = False  # 记录随机修改的方案是否成功
                    while not change_flag:
                        change_choose = random.choice(change_choose_list)
                        print(change_choose)
                        i_prev, q_prev, j_prev = calc_subtask_instance_from_position(
                            subtask_list_sorted[subtask_wrong_pos - 1])
                        i, q, j = calc_subtask_instance_from_position(subtask_list_sorted[subtask_wrong_pos])
                        '''change_choose = 0：选择修改前一任务的开始时间，修改到条件刚好成立，然后注意：
                        （1）如果任务开始执行时间小于EST，则认为不合理,否则change_flag置为True'''
                        if change_choose == 0:
                            change_choose_list.remove(change_choose)
                            print(f"原子任务所在CPU执行的任务为：{subtask_list_sorted}")
                            print(f"原子任务所在CPU执行的各子任务开始执行时间：{t_s_list_sorted}")
                            t_s_reserve = t_s[subtask_list_sorted[subtask_wrong_pos - 1]]
                            t_s[subtask_list_sorted[subtask_wrong_pos - 1]] = t_s[subtask_list_sorted[
                                    subtask_wrong_pos]] - app_list[i_prev].sw_exe[j_prev]
                            if t_s[subtask_list_sorted[subtask_wrong_pos - 1]] < instance_dict_list[i_prev][q_prev]['EST'][
                                j_prev]:
                                print("任务重构开始时间不合理，重新选择修改方式！")
                                t_s[subtask_list_sorted[subtask_wrong_pos - 1]] = t_s_reserve
                                continue
                            t_s_list = [t_s[subtask] for subtask in subtask_list]
                            t_s_list_sorted, subtask_list_sorted = sort_lists(t_s_list, subtask_list)
                            print(f"原子任务所在CPU执行的各子任务开始执行时间：{t_s_list_sorted}")
                            change_flag = True
                            '''change_choose = 1：选择修改后一任务的开始时间，修改到条件刚好成立，然后注意：
                            （1）如果任务开始执行时间大于LST，则认为不合理,否则change_flag置为True'''
                        elif change_choose == 1:
                            change_choose_list.remove(change_choose)
                            print(f"原子任务所在CPU执行的任务为：{subtask_list_sorted}")
                            print(f"原子任务所在CPU执行的各子任务开始执行时间：{t_s_list_sorted}")
                            t_s_reserve = t_s[subtask_list_sorted[subtask_wrong_pos]]
                            t_s[subtask_list_sorted[subtask_wrong_pos]] = t_s[subtask_list_sorted[
                                subtask_wrong_pos-1]] + app_list[i_prev].sw_exe[j_prev]
                            if t_s[subtask_list_sorted[subtask_wrong_pos]] > instance_dict_list[i][q]['LST'][j]:
                                print("任务重构开始时间不合理，重新选择修改方式！")
                                t_s[subtask_list_sorted[subtask_wrong_pos]] = t_s_reserve
                                continue
                            t_s_list = [t_s[subtask] for subtask in subtask_list]
                            t_s_list_sorted, subtask_list_sorted = sort_lists(t_s_list, subtask_list)
                            print(f"原子任务所在CPU执行的各子任务开始执行时间：{t_s_list_sorted}")
                            change_flag = True
                            '''change_choose = 2或3：选择修改一任务的映射，然后注意：
                            （1）如果不是CPU或NPU，执行（2），否则执行（3）
                            （2）调整映射后的RP区，执行（3）
                            （3）change_flag置为True
                            '''
                        else:
                            if change_choose == 2:
                                subtask_wrong_pos = subtask_wrong_pos - 1
                            change_choose_list.remove(change_choose)
                            print(
                                f"原来映射所在处理器为：{processor_mapping[subtask_list_sorted[subtask_wrong_pos]]}")
                            mapping_reserve = processor_mapping[subtask_list_sorted[subtask_wrong_pos]]
                            mapping_choice = list(range(processor_number))
                            mapping_choice.remove(processor_mapping[subtask_list_sorted[subtask_wrong_pos]])
                            processor_mapping[subtask_list_sorted[subtask_wrong_pos]] = random.choice(
                                mapping_choice)
                            print(
                                f"现在映射所在处理器为：{processor_mapping[subtask_list_sorted[subtask_wrong_pos]]}")
                            if processor_mapping[subtask_list_sorted[subtask_wrong_pos]] >= CPU_number + NPU_number:
                                mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == processor_mapping[
                                                            subtask_list_sorted[subtask_wrong_pos]]]
                                mapping_t_s_list = [t_s[subtask] for subtask in mapping_subtask_list]
                                mapping_t_s_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_t_s_list, mapping_subtask_list)
                                print(f"子任务现在所在RP区执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"子任务现在所在RP区的各子任务开始执行时间：{mapping_t_s_list_sorted}")
                                # 根据执行时间进行重构时间调整
                                mapping_pos = mapping_subtask_list_sorted.index(subtask_list_sorted[subtask_wrong_pos])
                                if mapping_pos != len(mapping_subtask_list_sorted) - 1:
                                    next_subtask = mapping_subtask_list_sorted[mapping_pos + 1]
                                    if check_same_subtask_mapping_successively_list[next_subtask][0] == 2:
                                        i_next, q_next, j_next = calc_subtask_instance_from_position(next_subtask)
                                        RPpr[next_subtask] = t_s[next_subtask] - app_list[i_next].tpr[j_next]
                                i_now, q_now, j_now = calc_subtask_instance_from_position(
                                    subtask_list_sorted[subtask_wrong_pos])
                                RPpr_reserve = RPpr[subtask_list_sorted[subtask_wrong_pos]]
                                RPpr[subtask_list_sorted[subtask_wrong_pos]] = t_s[subtask_list_sorted[
                                    subtask_wrong_pos]] - app_list[i_now].tpr[j_now]
                                if RPpr[subtask_list_sorted[subtask_wrong_pos]] < 0:
                                    RPpr[subtask_list_sorted[subtask_wrong_pos]] = RPpr_reserve
                                    processor_mapping[subtask_list_sorted[subtask_wrong_pos]] = mapping_reserve
                                    continue
                                mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == processor_mapping[
                                                            subtask_list_sorted[subtask_wrong_pos]]]
                                mapping_RPpr_list = [RPpr[subtask] for subtask in mapping_subtask_list]
                                mapping_RPpr_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_RPpr_list, mapping_subtask_list)
                                RPpr_adjust(mapping_subtask_list_sorted, RPpr,
                                            check_same_subtask_mapping_successively_list)
                                mapping_t_s_list = [t_s[subtask] for subtask in mapping_subtask_list]
                                mapping_t_s_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_t_s_list, mapping_subtask_list)
                                print(f"子任务现在所在RP区执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"子任务现在所在RP区的各子任务开始执行时间：{mapping_t_s_list_sorted}")
                            else:
                                mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == processor_mapping[
                                                            subtask_list_sorted[subtask_wrong_pos]]]
                                mapping_t_s_list = [t_s[subtask] for subtask in mapping_subtask_list]
                                mapping_t_s_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_t_s_list, mapping_subtask_list)
                                print(f"子任务现在所在处理器执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"子任务现在所在处理器各子任务开始执行时间：{mapping_t_s_list_sorted}")
                            subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == mapping_reserve]
                            t_s_list = [t_s[subtask] for subtask in subtask_list]
                            t_s_list_sorted, subtask_list_sorted = sort_lists(t_s_list, subtask_list)
                            print(f"子任务原先所在CPU执行的任务为：{subtask_list_sorted}")
                            print(f"子任务原先所在CPU执行的各子任务开始执行时间：{t_s_list_sorted}")
                            change_flag = True
                    break
        if not condition3_flag:
            continue

        # 判断条件4：是否满足NPU资源约束
        # 若不符合条件，则随机选择修改前一任务的开始时间或后一任务的开始时间或2个任务的映射硬件类型
        condition4_flag = True  # 是否一次性满足判断条件4
        subtask_wrong_pos = 0
        for NPU_index in range(CPU_number, CPU_number + NPU_number):  # 逐CPU进行判断
            subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                            processor_mapping[subtask] == NPU_index]
            t_s_list = [t_s[subtask] for subtask in subtask_list]
            if subtask_list:
                t_s_list_sorted, subtask_list_sorted = sort_lists(t_s_list, subtask_list)
                for k in range(1, len(subtask_list_sorted)):
                    i_prev, q_prev, j_prev = calc_subtask_instance_from_position(subtask_list_sorted[k - 1])
                    i, q, j = calc_subtask_instance_from_position(subtask_list_sorted[k])
                    if t_s[subtask_list_sorted[k]] < t_s[subtask_list_sorted[k - 1]] + app_list[i_prev].npu_exe[
                        j_prev]:
                        condition4_flag = False
                        subtask_wrong_pos = k
                        print(
                            f"NPU资源要求不满足，子任务为{subtask_list_sorted[subtask_wrong_pos]}，前一个任务为{subtask_list_sorted[subtask_wrong_pos - 1]}")
                        break
                if not condition4_flag:
                    change_choose_list = list(range(4))  # 记录随机修改的方案是否已经尝试
                    change_flag = False  # 记录随机修改的方案是否成功
                    while not change_flag:
                        change_choose = random.choice(change_choose_list)
                        print(change_choose)
                        i_prev, q_prev, j_prev = calc_subtask_instance_from_position(
                            subtask_list_sorted[subtask_wrong_pos - 1])
                        i, q, j = calc_subtask_instance_from_position(subtask_list_sorted[subtask_wrong_pos])
                        '''change_choose = 0：选择修改前一任务的开始时间，修改到条件刚好成立，然后注意：
                        （1）如果任务开始执行时间小于EST，则认为不合理,否则change_flag置为True'''
                        if change_choose == 0:
                            change_choose_list.remove(change_choose)
                            print(f"原子任务所在NPU执行的任务为：{subtask_list_sorted}")
                            print(f"原子任务所在NPU执行的各子任务开始执行时间：{t_s_list_sorted}")
                            t_s_reserve = t_s[subtask_list_sorted[subtask_wrong_pos - 1]]
                            t_s[subtask_list_sorted[subtask_wrong_pos - 1]] = t_s[subtask_list_sorted[
                                subtask_wrong_pos]] - app_list[i_prev].npu_exe[j_prev]
                            if t_s[subtask_list_sorted[subtask_wrong_pos - 1]] < \
                                    instance_dict_list[i_prev][q_prev]['EST'][
                                        j_prev]:
                                print("任务重构开始时间不合理，重新选择修改方式！")
                                t_s[subtask_list_sorted[subtask_wrong_pos - 1]] = t_s_reserve
                                continue
                            t_s_list = [t_s[subtask] for subtask in subtask_list]
                            t_s_list_sorted, subtask_list_sorted = sort_lists(t_s_list, subtask_list)
                            print(f"原子任务所在NPU执行的各子任务开始执行时间：{t_s_list_sorted}")
                            change_flag = True
                            '''change_choose = 1：选择修改后一任务的开始时间，修改到条件刚好成立，然后注意：
                            （1）如果任务开始执行时间大于LST，则认为不合理,否则change_flag置为True'''
                        elif change_choose == 1:
                            change_choose_list.remove(change_choose)
                            print(f"原子任务所在NPU执行的任务为：{subtask_list_sorted}")
                            print(f"原子任务所在NPU执行的各子任务开始执行时间：{t_s_list_sorted}")
                            t_s_reserve = t_s[subtask_list_sorted[subtask_wrong_pos]]
                            t_s[subtask_list_sorted[subtask_wrong_pos]] = t_s[subtask_list_sorted[
                                subtask_wrong_pos - 1]] + app_list[i_prev].npu_exe[j_prev]
                            if t_s[subtask_list_sorted[subtask_wrong_pos]] > instance_dict_list[i][q]['LST'][j]:
                                print("任务重构开始时间不合理，重新选择修改方式！")
                                t_s[subtask_list_sorted[subtask_wrong_pos]] = t_s_reserve
                                continue
                            t_s_list = [t_s[subtask] for subtask in subtask_list]
                            t_s_list_sorted, subtask_list_sorted = sort_lists(t_s_list, subtask_list)
                            print(f"原子任务所在NPU执行的各子任务开始执行时间：{t_s_list_sorted}")
                            change_flag = True
                            '''change_choose = 2或3：选择修改一任务的映射，然后注意：
                            （1）如果不是CPU或NPU，执行（2），否则执行（3）
                            （2）调整映射后的RP区，执行（3）
                            （3）change_flag置为True
                            '''
                        else:
                            if change_choose == 2:
                                subtask_wrong_pos = subtask_wrong_pos - 1
                            change_choose_list.remove(change_choose)
                            print(
                                f"原来映射所在处理器为：{processor_mapping[subtask_list_sorted[subtask_wrong_pos]]}")
                            mapping_reserve = processor_mapping[subtask_list_sorted[subtask_wrong_pos]]
                            mapping_choice = list(range(processor_number))
                            mapping_choice.remove(processor_mapping[subtask_list_sorted[subtask_wrong_pos]])
                            processor_mapping[subtask_list_sorted[subtask_wrong_pos]] = random.choice(
                                mapping_choice)
                            print(
                                f"现在映射所在处理器为：{processor_mapping[subtask_list_sorted[subtask_wrong_pos]]}")
                            if processor_mapping[subtask_list_sorted[subtask_wrong_pos]] >= CPU_number + NPU_number:
                                mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == processor_mapping[
                                                            subtask_list_sorted[subtask_wrong_pos]]]
                                mapping_t_s_list = [t_s[subtask] for subtask in mapping_subtask_list]
                                mapping_t_s_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_t_s_list, mapping_subtask_list)
                                print(f"子任务现在所在RP区执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"子任务现在所在RP区的各子任务开始执行时间：{mapping_t_s_list_sorted}")
                                # 根据执行时间进行重构时间调整
                                mapping_pos = mapping_subtask_list_sorted.index(
                                    subtask_list_sorted[subtask_wrong_pos])
                                if mapping_pos != len(mapping_subtask_list_sorted) - 1:
                                    next_subtask = mapping_subtask_list_sorted[mapping_pos + 1]
                                    if check_same_subtask_mapping_successively_list[next_subtask][0] == 2:
                                        i_next, q_next, j_next = calc_subtask_instance_from_position(next_subtask)
                                        RPpr[next_subtask] = t_s[next_subtask] - app_list[i_next].tpr[j_next]
                                i_now, q_now, j_now = calc_subtask_instance_from_position(
                                    subtask_list_sorted[subtask_wrong_pos])
                                RPpr_reserve = RPpr[subtask_list_sorted[subtask_wrong_pos]]
                                RPpr[subtask_list_sorted[subtask_wrong_pos]] = t_s[subtask_list_sorted[
                                    subtask_wrong_pos]] - app_list[i_now].tpr[j_now]
                                if RPpr[subtask_list_sorted[subtask_wrong_pos]] < 0:
                                    RPpr[subtask_list_sorted[subtask_wrong_pos]] = RPpr_reserve
                                    processor_mapping[subtask_list_sorted[subtask_wrong_pos]] = mapping_reserve
                                    continue
                                mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == processor_mapping[
                                                            subtask_list_sorted[subtask_wrong_pos]]]
                                mapping_RPpr_list = [RPpr[subtask] for subtask in mapping_subtask_list]
                                mapping_RPpr_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_RPpr_list, mapping_subtask_list)
                                RPpr_adjust(mapping_subtask_list_sorted, RPpr,
                                            check_same_subtask_mapping_successively_list)
                                mapping_t_s_list = [t_s[subtask] for subtask in mapping_subtask_list]
                                mapping_t_s_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_t_s_list, mapping_subtask_list)
                                print(f"子任务现在所在RP区执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"子任务现在所在RP区的各子任务开始执行时间：{mapping_t_s_list_sorted}")
                            else:
                                mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                        processor_mapping[subtask] == processor_mapping[
                                                            subtask_list_sorted[subtask_wrong_pos]]]
                                mapping_t_s_list = [t_s[subtask] for subtask in mapping_subtask_list]
                                mapping_t_s_list_sorted, mapping_subtask_list_sorted \
                                    = sort_lists(mapping_t_s_list, mapping_subtask_list)
                                print(f"子任务现在所在处理器执行的任务为：{mapping_subtask_list_sorted}")
                                print(f"子任务现在所在处理器各子任务开始执行时间：{mapping_t_s_list_sorted}")
                            subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                            processor_mapping[subtask] == mapping_reserve]
                            t_s_list = [t_s[subtask] for subtask in subtask_list]
                            t_s_list_sorted, subtask_list_sorted = sort_lists(t_s_list, subtask_list)
                            print(f"子任务原先所在NPU执行的任务为：{subtask_list_sorted}")
                            print(f"子任务原先所在NPU执行的各子任务开始执行时间：{t_s_list_sorted}")
                            change_flag = True
                    break
        if not condition4_flag:
            continue
        # 判断条5：是否满足依赖约束
        # 若不符合条件，则随机选择修改前一任务的开始时间或后一任务的开始时间,如果2种修改方式不可行，随机选一个映射到用时最短的执行器上
        condition5_flag = True  # 是否一次性满足判断条件5
        max_AFT = 0
        predecessors_max = []
        subtask_wrong_pos = 0
        for subtask in range(len(processor_mapping)):
            i,q,j = calc_subtask_instance_from_position(subtask)
            predecessors = app_list[i].predecessor_of_node(j)
            max_AFT,predecessors_max = calc_max_predessor_AFT(i,q,predecessors,CPU_number,NPU_number,processor_mapping,t_s)
            if t_s[subtask]< max_AFT:
                print(f"依赖约束不满足，子任务为{subtask}，前序节点为{predecessors_max}")
                condition5_flag = False
                subtask_wrong_pos = subtask
                break
        if not condition5_flag:
            change_choose_list = list(range(2))  # 记录随机修改的方案是否已经尝试
            change_flag = False  # 记录随机修改的方案是否成功
            while not change_flag:
                if change_choose_list:
                    change_choose = random.choice(change_choose_list)
                    print(change_choose)
                    '''change_choose = 0：选择修改前一任务的开始时间，修改到条件刚好成立，然后注意：
                    （1）如果任务开始执行时间小于EST，则认为不合理,否则执行（2）
                    （2）如果任务位于RP区，需要修改RPpr，需要调整RPpr'''
                    if change_choose == 0:
                        change_choose_list.remove(change_choose)
                        change_flag2 = True
                        for subtask_info in predecessors_max:
                            i_prev,q_prev,j_prev = calc_subtask_instance_from_position(subtask_info)
                            if processor_mapping[subtask_info]<CPU_number:
                                t_s_change = t_s[subtask_wrong_pos] - app_list[i_prev].sw_exe[j_prev]
                            elif processor_mapping[subtask_info]<CPU_number + NPU_number:
                                t_s_change = t_s[subtask_wrong_pos] - app_list[i_prev].npu_exe[j_prev]
                            else:
                                t_s_change = t_s[subtask_wrong_pos] - app_list[i_prev].hw_exe[j_prev]
                            if t_s_change<instance_dict_list[i_prev][q_prev]["EST"][j_prev]:
                                print("修改后的t_s不满足！")
                                change_flag2 = False
                                break
                            else:
                                if processor_mapping[subtask_info]>=CPU_number + NPU_number:    # 前序节点映射到RP区
                                    pre_mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                                processor_mapping[subtask] == processor_mapping[subtask_info]]
                                    pre_mapping_RPpr_list = [RPpr[subtask] for subtask in pre_mapping_subtask_list]
                                    if RPpr[subtask_info] > t_s_change - app_list[i_prev].tpr[j_prev]: # 如果当前任务执行前要重构，则要满足重构-执行约束
                                        t_s_reserve = t_s[subtask_info]
                                        RPpr_reserve = RPpr[subtask_info]
                                        t_s[subtask_info] = t_s_change
                                        RPpr[subtask_info] = t_s[subtask_info] - app_list[i_prev].tpr[j_prev]
                                        if RPpr[subtask_info] < 0:
                                            print("修改后的RPpr不满足！")
                                            RPpr[subtask_info] = RPpr_reserve
                                            t_s[subtask_info] = t_s_reserve
                                            change_flag2 = False
                                            break
                                        mapping_subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                                                                processor_mapping[subtask] == processor_mapping[
                                                                    subtask_info]]
                                        mapping_RPpr_list = [RPpr[subtask] for subtask in mapping_subtask_list]
                                        mapping_RPpr_list_sorted, mapping_subtask_list_sorted \
                                            = sort_lists(mapping_RPpr_list, mapping_subtask_list)
                                        # 可能因为提前不小心穿插到了2个同一任务的实例之间，需要调整
                                        mapping_pos = mapping_subtask_list_sorted.index(subtask_info)
                                        if mapping_pos != len(mapping_subtask_list_sorted) - 1:
                                            next_subtask = mapping_subtask_list_sorted[mapping_pos + 1]
                                            if check_same_subtask_mapping_successively_list[next_subtask][0] == 2:
                                                i_next, q_next, j_next = calc_subtask_instance_from_position(
                                                    next_subtask)
                                                RPpr[next_subtask] = t_s[next_subtask] - app_list[i_next].tpr[j_next]
                                        mapping_RPpr_list = [RPpr[subtask] for subtask in mapping_subtask_list]
                                        mapping_RPpr_list_sorted, mapping_subtask_list_sorted \
                                            = sort_lists(mapping_RPpr_list, mapping_subtask_list)
                                        RPpr_adjust(mapping_subtask_list_sorted, RPpr,
                                                    check_same_subtask_mapping_successively_list)
                                        mapping_RPpr_list = [RPpr[subtask] for subtask in mapping_subtask_list]
                                        mapping_RPpr_list_sorted, mapping_subtask_list_sorted \
                                            = sort_lists(mapping_RPpr_list, mapping_subtask_list)
                                        print(f"子任务现在所在处理器执行的任务为：{mapping_subtask_list_sorted}")
                                        print(f"子任务现在所在处理器各子任务开始执行时间：{mapping_RPpr_list_sorted}")
                                    else:
                                        t_s[subtask_info] = t_s_change
                                        print("RP 开始执行时间已修改")
                                else:
                                    t_s[subtask_info] = t_s_change
                                    print("CPU/NPU 开始执行时间已修改")
                        if change_flag2:
                            change_flag = True
                    if change_choose == 1:
                        pass
                else:
                    print("需要修改映射")


        if condition1_flag:
            print("DRP条件满足！")
            subtask_list = [subtask for subtask in range(len(processor_mapping)) if
                            processor_mapping[subtask] in RP_index_list]
            RPpr_list = [RPpr[subtask] for subtask in subtask_list]
            RPpr_list_sorted, subtask_list_sorted = sort_lists(RPpr_list, subtask_list)
            print(f"映射到FPGA的子任务：{subtask_list_sorted}")
            print(f"子任务重构开始时间：{RPpr_list_sorted}")
        if condition2_flag:
            print("fpga各区资源满足！")
        if condition3_flag:
            print("CPU各区资源满足！")
        if condition4_flag:
            print("NPU各区资源满足！")
        if condition1_flag and condition2_flag and condition3_flag and condition4_flag:
            flag = True

    print(f"修正次数：{rectify_count}")
    ### 绘制调度结果
    # 随机生成每个应用的颜色
    app_color_list = random.sample(colors, len(app_list))
    print(f"application colors:{app_color_list}")
    t_l = [0] * subtask_number
    subtask_colors = [0] *subtask_number
    position = 0
    subtask_labels = [""] * subtask_number
    subtask_RP_info = [[0, 0, 0] for _ in range(subtask_number)]
    for i in range(len(app_list)):
        for q in range(instance_number_list[i]):
            for j in range(app_list[i].node_num):
                if processor_mapping[position] < CPU_number:
                    t_l[position]=t_s[position]+app_list[i].sw_exe[j]
                elif processor_mapping[position] < CPU_number + NPU_number:
                    t_l[position] = t_s[position] + app_list[i].npu_exe[j]
                else:
                    if check_same_subtask_mapping_successively_list[position][0] < 2:
                        subtask_RP_info[position][0] = 1
                        subtask_RP_info[position][1] = RPpr[position]
                        subtask_RP_info[position][2] = RPpr[position] + app_list[i].tpr[j]
                    t_l[position] = t_s[position] + app_list[i].hw_exe[j]
                subtask_colors[position] = app_color_list[i]
                subtask_labels[position] = f"({i},{q+1},{j})"
                position+=1
    plotter = ProcessorTimelinePlotter(processor_number, CPU_number, NPU_number, RP_number, time_axis_length=H)
    task_blocks = [
        {'processor': processor_mapping[i], 'start': t_s[i], 'end': t_l[i], 'color': subtask_colors[i],
         'label':subtask_labels[i],'RP':subtask_RP_info[i]} for i in range(subtask_number)
    ]
    print(f"task blocks:{task_blocks}")
    print(len(task_blocks))
    plotter.set(task_blocks)

