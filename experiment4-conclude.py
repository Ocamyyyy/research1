import os
import pandas as pd
from glob import glob


save_dict="test4-formal, 2"
task_list = [(2,5,30),(2, 5, 40), (2, 5, 50), (2, 5, 60), (2, 5, 70), (2, 5, 80), (2, 5, 90), (2, 5, 100),
                     (2, 10, 40), (2, 10, 70), (2, 10, 100), (2, 10, 200), (2, 10, 300), (2, 10, 400), (2, 10, 500),
                     (2, 20, 100), (2, 20, 200), (2, 20, 300), (2, 20, 400), (2, 20, 500)]
NWR_list_tot = [0.5, 1.0, 2.0, 3.0, 4.0]

# 设置文件夹路径和输出文件名
# 创建一个空的DataFrame用于存储合并后的数据
merged_data = pd.DataFrame()
output_file = 'total.xlsx'  # 输出文件名

for task in task_list:
    for NWR in NWR_list_tot:
        folder_path=f".\\{save_dict}\\{task}\\{NWR}"
        # 获取文件夹中所有.xlsx文件
        excel_files = glob(os.path.join(folder_path, '*.xlsx'))
        for file in excel_files:
            try:
                # 读取Excel文件，跳过第一行
                df = pd.read_excel(file, skiprows=0)

                if not df.empty:
                    # 添加到合并数据中
                    merged_data = pd.concat([merged_data, df], ignore_index=True)
                    print(f"已处理文件: {os.path.basename(file)} - 添加了{len(df)}行数据")
                else:
                    print(f"文件 {os.path.basename(file)} 没有数据(只有标题行)")
            except Exception as e:
                print(f"处理文件 {os.path.basename(file)} 时出错: {str(e)}")

if not merged_data.empty:
    # 保存合并后的数据到新文件
    merged_data.to_excel(output_file, index=False)
    print(f"\n合并完成! 共合并了{len(merged_data)}行数据到 {output_file}")
else:
    print("没有数据被合并")