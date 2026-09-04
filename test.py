import numpy as np
from math import gamma, sin, pi


def levy_flight(lambda_param=1.5, size=1):
    """
    生成服从 Lévy 分布的步长。
    参数:
        lambda_param (float): 稳定性指数 (1 < λ ≤ 3)
        size (int): 返回的步长数量
    返回:
        numpy.ndarray: Lévy 步长数组
    """
    beta = lambda_param - 1  # β = λ - 1
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    s = u / (np.abs(v) ** (1 / beta))

    return s


# 示例：生成 10 个 Lévy 步长
steps = levy_flight(lambda_param=1.5, size=10)
print("Lévy steps:", steps)

def cuckoo_search_levy(nests, best_nest, alpha=1.0, lambda_param=1.5):
    """
    使用 Lévy flight 更新解的位置
    参数:
        nests (np.array): 当前解的集合
        best_nest (np.array): 当前最优解
        alpha (float): 步长缩放因子
        lambda_param (float): Lévy 分布的稳定性指数
    返回:
        np.array: 更新后的解
    """
    steps = alpha * levy_flight(lambda_param, size=nests.shape[0])
    new_nests = nests + steps.reshape(-1, 1) * (nests - best_nest)
    return new_nests

import matplotlib.pyplot as plt

# 生成 1000 个 Lévy 步长
steps = levy_flight(lambda_param=1.5, size=1000)

# 绘制直方图
plt.hist(steps, bins=50, density=True)
plt.title("Lévy Flight Step Distribution (λ=1.5)")
plt.xlabel("Step Size")
plt.ylabel("Probability Density")
plt.show()

# if (i, q, j) == (i2, q2, j2):  # 跳过同一任务
                                #     continue
                                # # 只有当X和X'都为真时约束才生效
                                # condition = And(X[(i, q, j, k)], X[(i2, q2, j2, k)])
                                # cpu_time1 = TaskConfig.app_list[i].sw_exe[j]
                                # cpu_time2 = TaskConfig.app_list[i2].sw_exe[j2]
                                # # 第一部分约束
                                # part1 = t[(i, q, j)] + M4 * (1 - If(A1[(i, q, j, i2, q2, j2)], 1, 0)) - t[
                                #     (i2, q2, j2)] - cpu_time2
                                # solver.add(Implies(condition, part1 >= 0)) # 在Implies(condition, constraint)中，当condition为真时，constraint必须成立（即被激活），而当condition为假时，constraint不会被强制执行
                                # # 第二部分约束
                                # part2 = t[(i2, q2, j2)] + M4 * If(A1[(i, q, j, i2, q2, j2)], 1, 0) - t[(i, q, j)] - cpu_time1
                                # solver.add(Implies(condition, part2 >= 0))

# if (i, q, j) == (i2, q2, j2):  # 跳过同一任务
                                #     continue
                                # # 只有当Y和Y'都为真时约束才生效
                                # condition = And(Y[(i, q, j, r)], Y[(i2, q2, j2, r)])
                                # npu_time1 = TaskConfig.app_list[i].npu_exe[j]
                                # npu_time2 = TaskConfig.app_list[i2].npu_exe[j2]
                                # # 第一部分约束
                                # part1 = t[(i, q, j)] + M4 * (1 - If(A1[(i, q, j, i2, q2, j2)], 1, 0)) - t[(i2, q2, j2)] - npu_time2
                                # solver.add(Implies(condition, part1 >= 0)) # 在Implies(condition, constraint)中，当condition为真时，constraint必须成立（即被激活），而当condition为假时，constraint不会被强制执行
                                # # 第二部分约束
                                # part2 = t[(i2, q2, j2)] + M4 * If(A1[(i, q, j, i2, q2, j2)], 1, 0) - t[(i, q, j)] - npu_time1
                                # solver.add(Implies(condition, part2 >= 0))
def compute_indicator(i, q, q2, j, eta):
    """计算复杂的指示器函数部分"""
    total = 0
    for i3 in range(len(TaskConfig.app_list)):
        for q3 in range(TaskConfig.instance_number_list[i3]):
            for j3 in range(TaskConfig.app_list[i3].node_num):
                if i3 == i and j3 == j and (q3 == q or q3 == q2):
                    continue
                rp = RPMap.get((i, q, j, i3, q3, j3, eta), False)
                as1 = A3.get((i, q, j, i3, q3, j3), False)
                as2 = A3.get((i3, q3, j3, i, q2, j), False)
                total += If(And(rp, as1, as2), 1, 0)
    return If(total > 0, True, False)  # 转换为指示器函数(0或1)


for i in range(len(TaskConfig.app_list)):
    for j in range(TaskConfig.app_list[i].node_num):
        for q in range(TaskConfig.instance_number_list[i]):
            for q2 in range(TaskConfig.instance_number_list[i]):
                if q == q2:  # 跳过q=q'的情况
                    continue
                for eta in range(TaskConfig.RP_number):
                    # 获取相关变量
                    simap = SIMap[(i, j, q, q2, eta)]
                    rpmap = RPMap[(i, q, j, i, q2, j, eta)]  # 注意i'=i, j'=j
                    a3_var = A3[(i, q, j, i, q2, j)]

                    # 计算Indicator部分
                    indicator = compute_indicator(i, q, q2, j, eta)  # 判断有没有别的任务实例出现在这两个任务之间

                    # 构建完整约束
                    solver.add(
                        simap == And(
                            rpmap,
                            a3_var,
                            Not(indicator)
                        )
                    )