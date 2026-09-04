import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
print(plt.rcParams['font.family'])
plt.rcParams['font.family'] = 'Arial'
print(plt.rcParams['font.family'])
save_dic_list = ["test1-formal, 3", "test2-formal, 1", "test3-formal, 1"]
fig_title_list = ["(a)", "(b)", "(c)"]

for idx, save_dic in enumerate(save_dic_list):
    plt.figure(figsize=(16, 10))

    # Read the Excel file
    df = pd.read_excel(f'.\\{save_dic}\\experiment_results.xlsx')

    # Prepare the data structure
    all_results = []
    for _, row in df.iterrows():
        if row['tot'] != 70 and row['tot'] != 40:
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

    # Process the data
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

    algorithms = ["HEFT", "GAA", "QC-HHO", "MoDCSA", "AEHS"]

    # Calculate averages
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

    # 调整参数：减小柱子宽度并添加组内间距
    bar_width = 0.15  # 柱子宽度
    intra_group_gap = 0.05  # 组内柱子间距
    group_gap = 0.3  # 组间间距

    # 计算每个组的总宽度
    total_group_width = len(algorithms) * bar_width + (len(algorithms) - 1) * intra_group_gap
    index = np.arange(len(x_values)) * (total_group_width + group_gap)

    colors = [
        # "#7F8A9B",
        # "#B7CBD5",
        # "#C1DDDB",
        # "#CFE0D8",
        # "#D1DED7"
        "#F56E68",
        "#FEB5AF",
        "#FFF4F2",
        "#1597A5",
        "#0E606C"
    ]

    hatches = ['///', '\\\\\\', '++', 'xx','']

    # Plot the bar chart - 在组内柱子间添加间距
    for i in range(len(algorithms)):
        # 计算每个柱子的位置：基础位置 + 柱子宽度*i + 组内间距*i
        bar_positions = index + i * (bar_width + intra_group_gap)
        plt.bar(
            bar_positions,
            NE_list[i],
            width=bar_width,
            label=algorithms[i],
            color=colors[i],
            edgecolor='black',
            alpha=0.8,
            hatch=hatches[i]
        )

    # Customize the plot
    plt.xlabel('Task Number in Hyperperiod', fontsize=32)
    plt.ylabel('Normalized Energy-consumption (%)', fontsize=32)

    # 设置x轴标签在每组中间
    plt.xticks(
        index + total_group_width / 2,
        x_values,
        fontsize=28
    )

    plt.yticks(fontsize=28)
    plt.ylim(15, max(max(sublist) for sublist in NE_list) * 1.1)
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=28)

    plt.tight_layout()
    plt.savefig(f"painter-1-{idx}.png")
    plt.close()