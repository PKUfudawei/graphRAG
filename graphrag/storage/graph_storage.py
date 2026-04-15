"""
Graph storage utilities.
图的保存和加载工具。
"""
import json
import networkx as nx
from networkx.readwrite import json_graph
from pathlib import Path


class GraphStorage:
    """
    图存储类。
    提供图的保存和加载功能。
    """

    @staticmethod
    def save_graph(graph: nx.Graph, path: str) -> str:
        """
        保存图为 JSON 格式（node-link 格式）。

        Args:
            graph: NetworkX 图
            path: 保存路径

        Returns:
            保存的路径
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = json_graph.node_link_data(graph, edges="edges")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\tGraph saved: {path}")
        return path

    @staticmethod
    def load_graph(path: str) -> nx.MultiDiGraph:
        """
        从 JSON 文件加载图。

        Args:
            path: 文件路径

        Returns:
            NetworkX MultiDiGraph
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = json_graph.node_link_graph(data, edges="edges")
        print(f"Graph loaded: {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph

    @staticmethod
    def save_community_metadata(
        community_summaries: list,
        path: str
    ) -> str:
        """
        保存社区元数据。

        Args:
            community_summaries: 社区总结列表（字典或 CommunitySummary 对象）
            path: 保存路径

        Returns:
            保存的路径
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # 转换为字典列表（排除 embedding）
        communities_data = []
        for comm in community_summaries:
            if hasattr(comm, 'to_dict'):
                data = comm.to_dict()
            else:
                data = {k: v for k, v in comm.items() if k != 'embedding'}
            communities_data.append(data)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(communities_data, f, ensure_ascii=False, indent=2)

        print(f"\tCommunity metadata saved: {path}")
        return path

    @staticmethod
    def load_community_metadata(path: str) -> list:
        """
        加载社区元数据。

        Args:
            path: 文件路径

        Returns:
            社区数据列表
        """
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
