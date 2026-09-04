from collections import defaultdict, deque
import math
import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm


class DAG:
    def __init__(self, generate_dict: dict, mode="certain"):
        self.node_num = 0
        self.edges = []
        self.period = 0
        self.sw_exe = []
        self.npu_exe = []
        self.hw_exe = []
        self.tpr = []
        self.P_static = []
        self.P_dyn = []
        self.P_sw = []
        self.P_npu = []

        self.cpu_eligible = []
        self.npu_eligible = []
        self.hw_eligible = []

        self.EST = []
        self.LST = []

        self.into_degree = []
        self.out_degree = []
        self.position = {}
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)

        if mode == "certain":
            self.generate_certain(generate_dict)
        else:
            self.generate_random(generate_dict)

    def _reset_graph_state(self):
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)
        self.into_degree = [0] * self.node_num
        self.out_degree = [0] * self.node_num

    def _rebuild_graph_indices(self):
        self._reset_graph_state()
        for start_node, end_node in self.edges:
            self.out_degree[start_node] += 1
            self.into_degree[end_node] += 1
            self.graph[start_node].append(end_node)
            self.reverse_graph[end_node].append(start_node)

    def _rebuild_positions(self):
        _depth, level_nodes = self.bfs_depth_and_levels()
        self.position = {}
        for level, nodes in level_nodes.items():
            x = 3 * level
            for idx, node in enumerate(nodes):
                self.position[node] = (x, idx * 5 + 1)

    def generate_certain(self, generate_dict):
        self.node_num = generate_dict["node num"]
        self.edges = list(generate_dict["edges"])
        self.period = generate_dict["period"]
        self.sw_exe = list(generate_dict["sw exe"])
        self.npu_exe = list(generate_dict["npu exe"])
        self.hw_exe = list(generate_dict["hw exe"])
        self.tpr = list(generate_dict["tpr"])
        self.P_static = list(generate_dict["P static"])
        self.P_dyn = list(generate_dict["P dyn"])
        self.P_sw = list(generate_dict["P sw"])
        self.P_npu = list(generate_dict["P npu"])
        self.cpu_eligible = list(generate_dict.get("cpu eligible", [True] * self.node_num))
        self.npu_eligible = list(generate_dict.get("npu eligible", [True] * self.node_num))
        self.hw_eligible = list(generate_dict.get("hw eligible", [True] * self.node_num))

        self._rebuild_graph_indices()
        self._rebuild_positions()

    def generate_random(self, generate_dict):
        self.node_num = generate_dict["V"]
        self.period = 0

        length = max(1, math.floor(math.sqrt(self.node_num) / generate_dict["alpha"]))
        mean_value = self.node_num / length
        random_num = np.random.normal(loc=mean_value, scale=generate_dict["beta"], size=length)

        dag_layers = []
        generated_nodes = 0
        for layer_index in range(length):
            layer_size = max(1, math.ceil(float(random_num[layer_index])))
            dag_layers.append(list(range(layer_size)))
            generated_nodes += layer_size

        if generated_nodes < self.node_num:
            for _ in range(self.node_num - generated_nodes):
                layer_index = random.randrange(length)
                dag_layers[layer_index].append(len(dag_layers[layer_index]))
        elif generated_nodes > self.node_num:
            removed = 0
            while removed < generated_nodes - self.node_num:
                layer_index = random.randrange(length)
                if len(dag_layers[layer_index]) <= 1:
                    continue
                del dag_layers[layer_index][-1]
                removed += 1

        relabeled_layers = []
        current_id = 0
        for layer in dag_layers:
            layer_nodes = list(range(current_id, current_id + len(layer)))
            relabeled_layers.append(layer_nodes)
            current_id += len(layer_nodes)

        self.edges = []
        for layer_index in range(length - 1):
            next_layer_candidates = list(range(len(relabeled_layers[layer_index + 1])))
            for node_index in range(len(relabeled_layers[layer_index])):
                out_degree = random.randrange(1, generate_dict["max_out"] + 1)
                out_degree = min(out_degree, len(relabeled_layers[layer_index + 1]))
                for chosen in random.sample(next_layer_candidates, out_degree):
                    self.edges.append(
                        (
                            relabeled_layers[layer_index][node_index],
                            relabeled_layers[layer_index + 1][chosen],
                        )
                    )

        self.cpu_eligible = [True] * self.node_num
        self.npu_eligible = [
            random.random() < generate_dict["npu_eligibility_prob"]
            for _ in range(self.node_num)
        ]
        self.hw_eligible = [
            random.random() < generate_dict["fpga_eligibility_prob"]
            for _ in range(self.node_num)
        ]

        self.tpr = [
            max(1, value)
            for value in self._generate_non_negative_integers(
                self.node_num,
                generate_dict["Tpr"],
                generate_dict["Tpr_beta"],
            )
        ]

        cpu_upper = max(1, int(round(2 * generate_dict["wDAG"])))
        self.sw_exe = [random.randint(1, cpu_upper) for _ in range(self.node_num)]

        self.npu_exe = []
        for node, sw_time in enumerate(self.sw_exe):
            if self.npu_eligible[node]:
                lower = max(1, int(math.floor(sw_time * (1 - generate_dict["gamma_GPU"] / 2))))
                self.npu_exe.append(random.randint(lower, sw_time))
            else:
                self.npu_exe.append(math.inf)

        self.hw_exe = []
        for node, sw_time in enumerate(self.sw_exe):
            if self.hw_eligible[node]:
                lower = max(1, int(math.floor(sw_time * (1 - generate_dict["gamma_hw"] / 2))))
                self.hw_exe.append(random.randint(lower, sw_time))
            else:
                self.hw_exe.append(math.inf)

        self.P_sw = [generate_dict["P_sw"]] * self.node_num
        self.P_npu = self._generate_positive_normal_floats(
            self.node_num,
            generate_dict["P_npu"],
            generate_dict["P_npu_beta"],
        )
        self.P_static = [
            random.uniform(generate_dict["P_static_min"], generate_dict["P_static_max"])
            for _ in range(self.node_num)
        ]
        self.P_dyn = [
            random.uniform(generate_dict["P_dyn_min"], generate_dict["P_dyn_max"])
            for _ in range(self.node_num)
        ]

        self._rebuild_graph_indices()
        self._rebuild_positions()

    @staticmethod
    def _generate_non_negative_integers(n, mean, std):
        lower = 0
        upper = max(mean * 2, 1)
        a = (lower - mean) / std
        b = (upper - mean) / std
        samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=n)
        integers = np.round(samples).astype(int)
        integers = np.clip(integers, 0, None)
        return integers.tolist()

    @staticmethod
    def _generate_positive_normal_floats(n, mean, std, lower=1e-6):
        values = []
        while len(values) < n:
            sample = random.gauss(mean, std)
            if sample >= lower:
                values.append(sample)
        return values

    def earliest_start_time(self):
        in_degree = self.into_degree.copy()
        nodes = list(range(self.node_num))
        earliest_time = [0] * self.node_num
        queue = deque([node for node in nodes if in_degree[node] == 0])

        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                earliest_time[v] = max(
                    earliest_time[v],
                    earliest_time[u] + min(self.sw_exe[u], self.npu_exe[u], self.hw_exe[u]),
                )
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return earliest_time

    def latest_start_time(self):
        out_degree = self.out_degree.copy()
        nodes = list(range(self.node_num))
        latest_time = [0] * self.node_num
        queue = deque([node for node in nodes if out_degree[node] == 0])
        for node in queue:
            latest_time[node] = -min(self.sw_exe[node], self.npu_exe[node], self.hw_exe[node])

        while queue:
            u = queue.popleft()
            for v in self.reverse_graph[u]:
                latest_time[v] = min(
                    latest_time[v],
                    latest_time[u] - min(self.sw_exe[v], self.npu_exe[v], self.hw_exe[v]),
                )
                out_degree[v] -= 1
                if out_degree[v] == 0:
                    queue.append(v)
        return latest_time

    def bfs_depth_and_levels(self):
        in_degree = self.into_degree.copy()
        queue = deque([i for i in range(self.node_num) if in_degree[i] == 0])
        level_nodes = defaultdict(list)
        node_levels = [-1] * self.node_num
        for node in queue:
            node_levels[node] = 1
            level_nodes[1].append(node)

        while queue:
            node = queue.popleft()
            for neighbor in self.graph[node]:
                max_predecessor_level = max(
                    node_levels[predecessor] for predecessor in self.reverse_graph[neighbor]
                )
                if node_levels[neighbor] == -1 or node_levels[neighbor] < max_predecessor_level + 1:
                    node_levels[neighbor] = max_predecessor_level + 1
                    level_nodes[node_levels[neighbor]].append(neighbor)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        max_depth = max(node_levels) if node_levels else 0
        for level in level_nodes:
            level_nodes[level].sort()
        return max_depth, dict(level_nodes)

    def predecessor_of_node(self, num):
        return [edge[0] for edge in self.edges if edge[1] == num]

    def successor_of_node(self, num):
        return [edge[1] for edge in self.edges if edge[0] == num]

    def sink_nodes(self):
        return [i for i, degree in enumerate(self.out_degree) if degree == 0]

    def find_affected_tasks(self, start_node, reverse=False):
        if start_node not in self.graph:
            present_nodes = {n for edges in self.graph.values() for n in edges}
            return [start_node] if start_node in present_nodes else []

        queue = deque([start_node])
        visited = {start_node}
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                else:
                    if neighbor in queue:
                        queue.remove(neighbor)
                        queue.append(neighbor)
                    elif neighbor in result:
                        result.remove(neighbor)
                        queue.append(neighbor)
        return result if not reverse else result[::-1]

    def plot_DAG(self, fig_name="DAG.png"):
        plt.clf()
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.node_num))
        graph.add_edges_from(self.edges)
        nx.draw_networkx(graph, arrows=True, pos=self.position)
        plt.savefig(fig_name, format="PNG")
        return plt.clf


if __name__ == "__main__":
    DAG_dict = {
        "node num": 7,
        "edges": [(0, 4), (1, 4), (2, 3), (2, 5), (3, 6), (4, 6)],
        "period": 30,
        "sw exe": [3] * 7,
        "npu exe": [1, 1, 2, 1, 1, 1, 1],
        "hw exe": [2] * 7,
        "tpr": [1] * 7,
        "P static": [1.0] * 7,
        "P dyn": [2.0] * 7,
        "P sw": [0.5] * 7,
        "P npu": [5.0] * 7,
        "cpu eligible": [True] * 7,
        "npu eligible": [True] * 7,
        "hw eligible": [True] * 7,
    }
    DAG1 = DAG(DAG_dict)
    DAG1.plot_DAG()
