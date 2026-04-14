"""
Community detection using Louvain algorithm.
使用 Louvain 算法进行社区发现。
"""
import networkx as nx
import community as community_louvain
from typing import Dict, List, Tuple


class CommunityDetector:
    """
    社区检测器。
    使用 Louvain 算法进行层次化社区发现。

    Args:
        max_community_size: 社区最大大小，超过则继续细分
    """

    def __init__(self, max_community_size: int = 50):
        self.max_community_size = max_community_size

    def detect_communities(self, graph: nx.Graph) -> Tuple[nx.Graph, Dict[int, List[str]], int]:
        """
        检测图中的社区。

        Args:
            graph: NetworkX 图

        Returns:
            (更新后的图，社区字典 {community_id: [节点列表]}, 下一个 community_id)
        """
        communities = {}
        next_comm_id = 0

        # 从所有节点开始
        graph, communities, next_comm_id = self._build_communities_recursive(
            graph, list(graph.nodes()), communities, next_comm_id
        )

        return graph, communities, next_comm_id

    def _build_communities_recursive(
        self,
        graph: nx.Graph,
        comm_nodes: List[str],
        communities: Dict[int, List[str]],
        next_comm_id: int
    ) -> Tuple[nx.Graph, Dict[int, List[str]], int]:
        """
        递归构建社区。

        Args:
            graph: NetworkX 图
            comm_nodes: 当前社区候选节点列表
            communities: 社区字典
            next_comm_id: 下一个社区 ID

        Returns:
            (更新后的图，社区字典，下一个 community_id)
        """
        # 如果节点数小于等于最大大小，直接作为一个社区
        if len(comm_nodes) <= self.max_community_size:
            for node in comm_nodes:
                graph.nodes[node]['community_id'] = next_comm_id

            communities[next_comm_id] = comm_nodes
            return graph, communities, next_comm_id + 1

        # 构建子图并运行 Louvain 算法
        subgraph = graph.subgraph(comm_nodes).to_undirected()
        partition = community_louvain.best_partition(subgraph)

        # 按社区分组
        partition_groups = {}
        for node, comm_id in partition.items():
            partition_groups.setdefault(comm_id, []).append(node)

        # 如果只有一个社区，不再细分
        if len(partition_groups) == 1:
            for node in comm_nodes:
                graph.nodes[node]['community_id'] = next_comm_id
            communities[next_comm_id] = comm_nodes
            return graph, communities, next_comm_id + 1

        # 递归处理每个子社区
        for group_nodes in partition_groups.values():
            graph, sub_comms, next_comm_id = self._build_communities_recursive(
                graph, group_nodes, communities, next_comm_id
            )
            communities.update(sub_comms)

        return graph, communities, next_comm_id

