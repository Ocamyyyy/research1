import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

save_dic_list = ["test1-formal, 3", "test2-formal, 1", "test3-formal, 1"]
# save_dic_list = ["test1-formal, 3", "test2-formal, 1"]
fig_title_list=["(a)","(b)","(c)"]
plt.figure(figsize=(16, 10))
for idx, save_dic in enumerate(save_dic_list):
    plt.close()
    plt.figure(figsize=(16, 10))
    # plt.title(
    #     fig_title_list[idx],
    #     y=-0.2,  # 控制标题位置（负值表示在子图下方）
    #     verticalalignment='top',  # 文本顶部对齐（因为 y 是负值）
    #     fontsize=10
    # )
    # Read the Excel file (replace 'your_file.xlsx' with your actual file path)
    df = pd.read_excel(f'.\\{save_dic}\\experiment_results.xlsx')

    # Prepare the data structure
    all_results = []
    for _, row in df.iterrows():
        if row['tot']!=70 and row['tot'] !=40:
            key = (row['N'], row['V'], row['tot'], row['iterr'])
            results = {
                'heft_energy': row['HEFT energy'],
                'heft_delay': row['HEFT delay'],
                'ga_energy': row['GAA energy'],
                'ga_delay': row['GAA delay'],
                'hho_modified_energy': row['QC-HHO energy'],
                'hho_modified_delay': row['QC-HHO delay'],
                'cuckoo_energy': row['GACSM energy'],
                'cuckoo_delay': row['GACSM delay'],
                'qha_energy': row['QHA energy'],
                'qha_delay': row['QHA delay']
            }
            all_results.append((key, results))

    # Process the data as per your provided code
    NE_data = defaultdict(lambda: defaultdict(list))
    delay_data = defaultdict(lambda: defaultdict(list))
    for (N, V, tot, iterr), results in all_results:
        NE_data[(N, V, tot)]['heft'].append(results['heft_energy'])
        NE_data[(N, V, tot)]['ga'].append(results['ga_energy'])
        NE_data[(N, V, tot)]['hho_modified'].append(results['hho_modified_energy'])
        NE_data[(N, V, tot)]['cuckoo'].append(results['cuckoo_energy'])
        NE_data[(N, V, tot)]['qha'].append(results['qha_energy'])
        delay_data[(N, V, tot)]['heft'].append(results['heft_delay'])
        delay_data[(N, V, tot)]['ga'].append(results['ga_delay'])
        delay_data[(N, V, tot)]['hho_modified'].append(results['hho_modified_delay'])
        delay_data[(N, V, tot)]['cuckoo'].append(results['cuckoo_delay'])
        delay_data[(N, V, tot)]['qha'].append(results['qha_delay'])

    algorithms = ["HEFT", "GAA", "QC-HHO", "GACSM", "QMES"]
    # Calculate averages and populate NE_list
    NE_list = [[] for _ in range(len(algorithms))]
    delay_list = [[] for _ in range(len(algorithms))]
    test_case_order = sorted(NE_data.keys(), key=lambda x: x[2])  # Sort by tot

    for N, V, tot in test_case_order:
        avg_heft = np.mean(NE_data[(N, V, tot)]['heft'])
        avg_cuckoo = np.mean(NE_data[(N, V, tot)]['cuckoo'])
        avg_ga = np.mean(NE_data[(N, V, tot)]['ga'])
        avg_hho_modified = np.mean(NE_data[(N, V, tot)]['hho_modified'])
        avg_qha = np.mean(NE_data[(N, V, tot)]['qha'])
        avg_heft_delay = np.mean(delay_data[(N, V, tot)]['heft'])
        avg_cuckoo_delay = np.mean(delay_data[(N, V, tot)]['cuckoo'])
        avg_ga_delay = np.mean(delay_data[(N, V, tot)]['ga'])
        avg_hho_modified_delay = np.mean(delay_data[(N, V, tot)]['hho_modified'])
        avg_qha_delay = np.mean(delay_data[(N, V, tot)]['qha'])

        NE_list[0].append(avg_heft)
        NE_list[1].append(avg_ga)
        NE_list[2].append(avg_hho_modified)
        NE_list[3].append(avg_cuckoo)
        NE_list[4].append(avg_qha)

        delay_list[0].append(avg_heft_delay)
        delay_list[1].append(avg_ga_delay)
        delay_list[2].append(avg_hho_modified_delay)
        delay_list[3].append(avg_cuckoo_delay)
        delay_list[4].append(avg_qha_delay)

    # Prepare x-axis labels
    x_values = [str(int(tot)) for _, _, tot in test_case_order]
    x_positions = np.arange(len(x_values))  # 使用简单的数值位置作为x轴

    colors = [
        "#7F8A9B",
        "#B7CBD5",
        "#C1DDDB",
        "#CFE0D8",
        "#D1DED7"
    ]

    markers = ['o', 's', '^', 'D', 'v']  # 为每条线定义不同的标记
    line_styles = ['-', '--', '-.', ':', '-']  # 为每条线定义不同的线型

    # 绘制折线图
    for i in range(len(algorithms)):
        plt.plot(
            x_positions,
            NE_list[i],
            label=algorithms[i],
            color=colors[i],
            marker=markers[i],
            linestyle=line_styles[i],
            markersize=10,  # 标记大小
            linewidth=2.5   # 线宽
        )

    # Customize the plot
    plt.xlabel('Task Number in Hyperperiod', fontsize=28)
    plt.ylabel('Normalized Energy-consumption (%)', fontsize=28)
    plt.xticks(
        x_positions,
        x_values,
        fontsize=28
    )
    plt.yticks(fontsize=28)
    plt.ylim(15, max(max(sublist) for sublist in NE_list) * 1.1)
    # plt.grid(axis='y', linestyle='--', alpha=0.6)  # 横向网格线
    # 如果需要添加网格线，可以取消下面的注释
    # plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=28)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"painter-2-{idx}.png")