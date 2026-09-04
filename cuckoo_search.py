import numpy as np
from scipy.special import gamma
import random


def cuckoo_search(n=25, pa=0.25, nd=15, tol=1.0e-5, max_iter=10000):
    """
    布谷鸟搜索算法Python实现

    参数:
    n -- 巢穴数量(默认25)
    pa -- 发现概率(默认0.25)
    nd -- 问题维度(默认15)
    tol -- 容差(默认1e-5)
    max_iter -- 最大迭代次数(默认10000)

    返回:
    bestnest -- 最优解
    fmin -- 最优值
    """
    # 搜索空间边界
    Lb = -5 * np.ones(nd)
    Ub = 5 * np.ones(nd)

    # 初始化巢穴
    nest = np.zeros((n, nd))
    print(len(nest))
    for i in range(n):
        nest[i, :] = Lb + (Ub - Lb) * np.random.rand(nd)


    # 初始评估
    fitness = np.ones(n) * 1e10
    fmin, bestnest, nest, fitness = get_best_nest(nest, nest, fitness)
    print(fmin)

    N_iter = 0
    while fmin > tol and N_iter < max_iter:
        # 通过Levy飞行生成新解
        new_nest = get_cuckoos(nest, bestnest, Lb, Ub)
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
        N_iter += n

        # 发现并丢弃较差解
        new_nest = empty_nests(nest, Lb, Ub, pa)
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
        N_iter += n

        # 更新全局最优
        if fnew < fmin:
            fmin = fnew
            bestnest = best

    print(f"Total number of iterations: {N_iter}")
    print(f"Minimum value found: {fmin}")
    print(f"Best solution: {bestnest}")

    return bestnest, fmin


def get_cuckoos(nest, best, Lb, Ub):
    """
    通过Levy飞行生成新解
    """
    n, d = nest.shape
    beta = 3 / 2
    # 正确计算sigma的公式
    numerator = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    denominator = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (numerator / denominator) ** (1 / beta)

    new_nest = np.zeros((n, d))
    for j in range(n):
        s = nest[j, :]
        # Mantegna算法实现Levy飞行
        u = np.random.randn(d) * sigma
        v = np.random.randn(d)
        step = u / np.abs(v) ** (1 / beta)

        # 步长缩放
        stepsize = 0.01 * step * (s - best)
        print(s,best)
        s_new = s + stepsize * np.random.randn(d)

        # 应用边界约束
        new_nest[j, :] = simplebounds(s_new, Lb, Ub)

    return new_nest


def get_best_nest(nest, new_nest, fitness):
    """
    评估并获取最优解
    """
    n = nest.shape[0]
    updated_fitness = fitness.copy()
    updated_nest = nest.copy()

    # 对于每个个体，若新个体适应度更强，则选择新个体
    for j in range(n):
        fnew = fobj(new_nest[j, :])
        if fnew <= fitness[j]:
            updated_fitness[j] = fnew
            updated_nest[j, :] = new_nest[j, :]

    # 找到当前最优
    fmin = np.min(updated_fitness)
    best_idx = np.argmin(updated_fitness)
    best = updated_nest[best_idx, :]

    return fmin, best, updated_nest, updated_fitness


def empty_nests(nest, Lb, Ub, pa):
    """
    发现并替换较差解
    """
    n, d = nest.shape
    # 随机决定哪些巢穴会被发现
    K = np.random.rand(n, d) > pa

    # 生成随机步长
    perm1 = np.random.permutation(n)
    perm2 = np.random.permutation(n)
    stepsize = np.random.rand() * (nest[perm1, :] - nest[perm2, :])

    # 生成新解
    new_nest = nest + stepsize * K
    for j in range(n):
        new_nest[j, :] = simplebounds(new_nest[j, :], Lb, Ub)

    return new_nest


def simplebounds(s, Lb, Ub):
    """
    应用简单边界约束
    """
    s_new = s.copy()
    s_new[s_new < Lb] = Lb[s_new < Lb]
    s_new[s_new > Ub] = Ub[s_new > Ub]
    return s_new


def fobj(u):
    """
    目标函数示例: d维球面函数
    最小值在(1,1,...,1)处，值为0
    """
    return np.sum((u - 1) ** 2)


# 示例使用
if __name__ == "__main__":
    best_solution, best_value = cuckoo_search()