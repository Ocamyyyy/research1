import matplotlib.pyplot as plt
import matplotlib.patches as patches


colors = [
    "#F5D9E6",
    "#DFE1E2",
    "#CEE2F5",
    "#CBE0BB",
    "#F7E5CA",
    "#FFB6C1",  # LightPink 浅粉红
    "#FFC0CB",  # Pink 粉红
    "#DC143C",  # Crimson 深红/猩红
    "#FFF0F5",  # LavenderBlush 淡紫红
    "#DB7093",  # PaleVioletRed 弱紫罗兰红
    "#FF69B4",  # HotPink 热情的粉红
    "#FF1493",  # DeepPink 深粉红
    "#C71585",  # MediumVioletRed 中紫罗兰红
    "#DA70D6",  # Orchid 暗紫色/兰花紫
    "#D8BFD8",  # Thistle 蓟色
    "#DDA0DD",  # Plum 洋李色/李子紫
    "#EE82EE",  # Violet 紫罗兰
    "#FF00FF",  # Magenta 洋红/玫瑰红
    "#FF00FF",  # Fuchsia 紫红/灯笼海棠
    "#8B008B",  # DarkMagenta 深洋红
    "#800080",  # Purple 紫色
    "#BA55D3",  # MediumOrchid 中兰花紫
    "#9400D3",  # DarkViolet 暗紫罗兰
    "#9932CC",  # DarkOrchid 暗兰花紫
    "#4B0082",  # Indigo 靛青/紫兰色
    "#8A2BE2",  # BlueViolet 蓝紫罗兰
    "#9370DB",  # MediumPurple 中紫色
    "#7B68EE",  # MediumSlateBlue 中暗蓝色/中板岩蓝
    "#6A5ACD",  # SlateBlue 石蓝色/板岩蓝
    "#483D8B",  # DarkSlateBlue 暗灰蓝色/暗板岩蓝
    "#E6E6FA",  # Lavender 淡紫色/熏衣草淡紫
    "#F8F8FF",  # GhostWhite 幽灵白
    "#0000FF",  # Blue 纯蓝
    "#0000CD",  # MediumBlue 中蓝色
    "#191970",  # MidnightBlue 午夜蓝
    "#00008B",  # DarkBlue 暗蓝色
    "#000080",  # Navy 海军蓝
    "#4169E1",  # RoyalBlue 皇家蓝/宝蓝
    "#6495ED",  # CornflowerBlue 矢车菊蓝
    "#B0C4DE",  # LightSteelBlue 亮钢蓝
    "#778899",  # LightSlateGray 亮蓝灰/亮石板灰
    "#708090",  # SlateGray 灰石色/石板灰
    "#1E90FF",  # DodgerBlue 闪兰色/道奇蓝
    "#F0F8FF",  # AliceBlue 爱丽丝蓝
    "#4682B4",  # SteelBlue 钢蓝/铁青
    "#87CEFA",  # LightSkyBlue 亮天蓝色
    "#87CEEB",  # SkyBlue 天蓝色
    "#00BFFF",  # DeepSkyBlue 深天蓝
    "#ADD8E6",  # LightBlue 亮蓝
    "#B0E0E6",  # PowderBlue 粉蓝色/火药青
    "#5F9EA0",  # CadetBlue 军兰色/军服蓝
    "#F0FFFF",  # Azure 蔚蓝色
    "#E0FFFF",  # LightCyan 淡青色
    "#AFEEEE",  # PaleTurquoise 弱绿宝石
    "#00FFFF",  # Cyan 青色
    "#00FFFF",  # Aqua 浅绿色/水色
    "#00CED1",  # DarkTurquoise 暗绿宝石
    "#2F4F4F",  # DarkSlateGray 暗瓦灰色/暗石板灰
    "#008B8B",  # DarkCyan 暗青色
    "#008080",  # Teal 水鸭色
    "#48D1CC",  # MediumTurquoise 中绿宝石
    "#20B2AA",  # LightSeaGreen 浅海洋绿
    "#40E0D0",  # Turquoise 绿宝石
    "#7FFFD4",  # Aquamarine 宝石碧绿
    "#66CDAA",  # MediumAquamarine 中宝石碧绿
    "#00FA9A",  # MediumSpringGreen 中春绿色
    "#F5FFFA",  # MintCream 薄荷奶油
    "#00FF7F",  # SpringGreen 春绿色
    "#3CB371",  # MediumSeaGreen 中海洋绿
    "#2E8B57",  # SeaGreen 海洋绿
    "#F0FFF0",  # Honeydew 蜜色/蜜瓜色
    "#90EE90",  # LightGreen 淡绿色
    "#98FB98",  # PaleGreen 弱绿色
    "#8FBC8F",  # DarkSeaGreen 暗海洋绿
    "#32CD32",  # LimeGreen 闪光深绿
    "#00FF00",  # Lime 闪光绿
    "#228B22",  # ForestGreen 森林绿
    "#008000",  # Green 纯绿
    "#006400",  # DarkGreen 暗绿色
    "#7FFF00",  # Chartreuse 黄绿色/查特酒绿
    "#7CFC00",  # LawnGreen 草绿色/草坪绿
    "#ADFF2F",  # GreenYellow 绿黄色
    "#556B2F",  # DarkOliveGreen 暗橄榄绿
    "#9ACD32",  # YellowGreen 黄绿色
    "#6B8E23",  # OliveDrab 橄榄褐色
    "#F5F5DC",  # Beige 米色/灰棕色
    "#FAFAD2",  # LightGoldenrodYellow 亮菊黄
    "#FFFFF0",  # Ivory 象牙色
    "#FFFFE0",  # LightYellow 浅黄色
    "#FFFF00",  # Yellow 纯黄
    "#808000",  # Olive 橄榄
    "#BDB76B",  # DarkKhaki 暗黄褐色/深卡叽布
    "#FFFACD",  # LemonChiffon 柠檬绸
    "#EEE8AA",  # PaleGoldenrod 灰菊黄/苍麒麟色
    "#F0E68C",  # Khaki 黄褐色/卡叽布
    "#FFD700",  # Gold 金色
    "#FFF8DC",  # Cornsilk 玉米丝色
    "#DAA520",  # Goldenrod 金菊黄
    "#B8860B",  # DarkGoldenrod 暗金菊黄
    "#FFFAF0",  # FloralWhite 花的白色
    "#FDF5E6",  # OldLace 老花色/旧蕾丝
    "#F5DEB3",  # Wheat 浅黄色/小麦色
    "#FFE4B5",  # Moccasin 鹿皮色/鹿皮靴
    "#FFA500",  # Orange 橙色
    "#FFEFD5",  # PapayaWhip 番木色/番木瓜
    "#FFEBCD",  # BlanchedAlmond 白杏色
    "#FFDEAD",  # NavajoWhite 纳瓦白/土著白
    "#FAEBD7",  # AntiqueWhite 古董白
    "#D2B48C",  # Tan 茶色
    "#DEB887",  # BurlyWood 硬木色
    "#FFE4C4",  # Bisque 陶坯黄
    "#FF8C00",  # DarkOrange 深橙色
    "#FAF0E6",  # Linen 亚麻布
    "#CD853F",  # Peru 秘鲁色
    "#FFDAB9",  # PeachPuff 桃肉色
    "#F4A460",  # SandyBrown 沙棕色
    "#D2691E",  # Chocolate 巧克力色
    "#8B4513",  # SaddleBrown 重褐色/马鞍棕色
    "#FFF5EE",  # Seashell 海贝壳
    "#A0522D",  # Sienna 黄土赭色
    "#FFA07A",  # LightSalmon 浅鲑鱼肉色
    "#FF7F50",  # Coral 珊瑚
    "#FF4500",  # OrangeRed 橙红色
    "#E9967A",  # DarkSalmon 深鲜肉/鲑鱼色
    "#FF6347",  # Tomato 番茄红
    "#FFE4E1",  # MistyRose 浅玫瑰色/薄雾玫瑰
    "#FA8072",  # Salmon 鲜肉/鲑鱼色
    "#FFFAFA",  # Snow 雪白色
    "#F08080",  # LightCoral 淡珊瑚色
    "#BC8F8F",  # RosyBrown 玫瑰棕色
    "#CD5C5C",  # IndianRed 印度红
    "#FF0000",  # Red 纯红
    "#A52A2A",  # Brown 棕色
    "#B22222",  # FireBrick 火砖色/耐火砖
    "#8B0000",  # DarkRed 深红色
    "#800000",  # Maroon 栗色
    "#FFFFFF",  # White 纯白
    "#F5F5F5",  # WhiteSmoke 白烟
    "#DCDCDC",  # Gainsboro 淡灰色
    "#D3D3D3",  # LightGrey 浅灰色
    "#C0C0C0",  # Silver 银灰色
    "#A9A9A9",  # DarkGray 深灰色
    "#808080",  # Gray 灰色
    "#696969",  # DimGray 暗淡灰
]


class ProcessorTimelinePlotter:
    def __init__(self, num_processors, num_CPU, num_NPU, num_RP, time_axis_length):
        """
        初始化处理器时间轴绘制器

        :param num_processors: 处理器数量
        :param time_axis_length: 时间轴长度
        """
        self.num_processors = num_processors
        self.num_CPU = num_CPU
        self.num_NPU = num_NPU
        self.num_RP = num_RP
        self.time_axis_length = time_axis_length

    def set(self, window_name,task_blocks=None,is_block = False,is_save = False):
        """
        绘制处理器时间轴
        """
        # 创建一个图形
        plt.close()
        fig, ax = plt.subplots(num=window_name)
        # plt.ion()

        time_list = [block['end'] for block in task_blocks]
        self.time_axis_length = max(max(time_list),self.time_axis_length)

        # 为每个处理器绘制时间轴
        for i in range(self.num_processors):
            # 绘制水平线
            ax.hlines(y=i, xmin=0, xmax=self.time_axis_length, colors='grey', lw=2)

            # 标记时间轴的起点和终点
            ax.text(0, i, f'P{i}', va='center', ha='right', color='red')
            # ax.text(self.time_axis_length, i, f'{self.time_axis_length}', va='center', ha='left', color='green')

        # 如果有任务块，绘制小方块
        if task_blocks:
            for block in task_blocks:
                # print(block)
                processor = block['processor']
                start = block['start']
                end = block['end']
                RP = block['RP']
                color = block.get('color')  # 默认颜色为橙色
                label = block.get('label', '')  # 默认标注文字为空
                hatch = block.get('hatch', '')  # 默认填充样式为斜线

                # 方块的高度（设置为 0.66，确保方块之间有空隙）
                height = 0.66

                # 方块的 y 坐标（下边缘与时间轴齐平）
                y = processor

                # 绘制小方块
                rect = patches.Rectangle(
                    (start, y),  # (x, y) 位置
                    end - start,  # 宽度
                    height,  # 高度
                    linewidth=1, edgecolor='black', facecolor=color, hatch=hatch
                )
                ax.add_patch(rect)

                if RP[0]==1:
                    rect = patches.Rectangle(
                        (RP[1], y),  # (x, y) 位置
                        RP[2] - RP[1],  # 宽度
                        height,  # 高度
                        linewidth=1, edgecolor='black', facecolor=color, hatch="///"
                    )
                    ax.add_patch(rect)

                # 在小方块中心添加标注文字
                if label:
                    text_x = start + (end - start) / 2  # 方块中心的 x 坐标
                    text_y = y + height / 2  # 方块中心的 y 坐标
                    ax.text(text_x, text_y, label, va='center', ha='center', color='black', fontsize=10)

        # 设置图形属性
        ax.set_xlabel('Time')
        ax.set_ylabel('Processors')
        ax.set_yticks(range(self.num_processors))
        ax.set_yticklabels([f'CPU {i}' for i in range(self.num_CPU)]+[f'NPU {i}' for i in range(self.num_NPU)]+[f'RP {i}' for i in range(self.num_RP)])
        ax.set_title(f'Time Axis for {self.num_processors} Processors (0 to {self.time_axis_length})')

        # 设置 X 轴和 Y 轴的范围
        ax.set_xlim(-1, self.time_axis_length + 1)
        ax.set_ylim(0, self.num_processors)

        # 显示图形
        if is_save:
            plt.savefig(f"{window_name}.png")
            plt.close(fig)  # 显式关闭图形
        else:
            plt.show(block=is_block)
            plt.pause(0.1)  # 保证窗口渲染完成



# 示例用法
if __name__ == "__main__":
    # 初始化：5 个处理器，时间轴长度为 10
    CPU_number = 4
    NPU_number = 1
    RP_number = 4
    processor_number = CPU_number + NPU_number + RP_number
    plotter = ProcessorTimelinePlotter(processor_number, CPU_number, NPU_number, RP_number, time_axis_length=10)
    # 定义任务块：在 P0 的时间轴上添加一个从时间 0 开始，延续到时间片 5 结束的小方块
    task_blocks = [
        {'processor': 0, 'start': 0, 'end': 5, 'color': "#FFC0CB", 'RP':[0,-1]}
    ]

    # 调用绘制函数
    plotter.set(task_blocks)