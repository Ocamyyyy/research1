import re
import matplotlib.pyplot as plt
from collections import defaultdict

# 数据结构：{algorithm: {tot_value: [time_values]}}
data = defaultdict(lambda: defaultdict(list))

# 正则表达式匹配模式 - 保持原来的模式
pattern = re.compile(r'^(\d+) (\d+) (\d+) iter:(\d+) (HEFT|GAA|QC-HHO|GACSM|QMES):[\d.]+ delay:[\d.]+ time:([\d.]+)s$')

# 读取文件并解析数据
with open('.\\test1-formal, 3\\my_document.txt', 'r') as f:
    for line in f:
        line = line.strip()
        match = pattern.match(line)
        if match:
            _, _, tot_value, iter_num, algorithm, time_value = match.groups()
            tot_value = int(tot_value)
            time_value = float(time_value)
            # 将GACSM和QMES映射为绘图时使用的名称
            if algorithm == 'GACSM':
                algorithm = 'MoDCSA'
            elif algorithm == 'QMES':
                algorithm = 'AEHS'
            data[algorithm][tot_value].append(time_value)

# 计算每个算法和每个tot_value的平均时间
averages = defaultdict(dict)
for algorithm in data:
    for tot_value in sorted(data[algorithm]):
        times = data[algorithm][tot_value]
        averages[algorithm][tot_value] = sum(times) / len(times)

# 准备绘图数据
algorithms = ['GAA', 'QC-HHO', 'MoDCSA', 'AEHS']
original_x_values = sorted({tot for alg in averages for tot in averages[alg]})  # 原始x值（如40,70,100,...）

# 创建等距的x轴位置（0,1,2,...）
x_positions = range(len(original_x_values))  # 等距分布
x_labels = [str(x) for x in original_x_values]  # 保持原始标签

# 创建图表
plt.figure(figsize=(12, 10))

# 为每种算法绘制折线图
colors = ["#F56E68",
        "#FEB5AF",
        "#FFC249",
        "#1597A5",
        "#0E606C"]
markers = ['o', 's', '^', 'D', 'v']

for idx, algorithm in enumerate(algorithms):
    if algorithm in averages:
        # 按等距x_positions绘制，但标签仍显示原始tot值
        y_values = [averages[algorithm].get(x, 0) for x in original_x_values]
        plt.plot(x_positions, y_values,
                 label=algorithm,
                 color=colors[idx],
                 marker=markers[idx],
                 linewidth=2,
                 markersize=8)
plt.rcParams['font.family'] = 'Arial'
# 设置图表属性
plt.xlabel('Task Number in Hyperperiod', fontsize=28)
plt.ylabel('Average Complete Computing Time (s)', fontsize=28)
plt.xticks(x_positions, x_labels, fontsize=28)  # 关键修改：等距x轴，但标签用原始值
plt.yticks(fontsize=28)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=20)

plt.tight_layout()
plt.savefig("painter-time.png")
plt.show()  # 添加show()以便在开发环境中查看图表