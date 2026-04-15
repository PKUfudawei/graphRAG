"""
Graph builder using NetworkX.
使用 NetworkX 构建知识图谱。
"""
import networkx as nx
from typing import List, Dict, Optional
from tqdm import tqdm

from ..models.graph_models import GraphDocumentWrapper


class GraphBuilder:
    """
    知识图谱构建器。
    将 GraphDocumentWrapper 转换为 NetworkX 图。

    Args:
        graph: 可选的初始图，默认为 MultiDiGraph
    """

    def __init__(self, graph: Optional[nx.MultiDiGraph] = None):
        self.graph = graph if graph is not None else nx.MultiDiGraph()

    def add_document(self, doc: GraphDocumentWrapper) -> "GraphBuilder":
        """
        添加单个 GraphDocument 到图。

        Args:
            doc: GraphDocumentWrapper 实例

        Returns:
            self
        """
        # 添加节点
        for node in doc.nodes:
            if node.id not in self.graph:
                self.graph.add_node(
                    node.id,
                    type=node.type,
                    weight=1,
                    **node.properties
                )
            else:
                self.graph.nodes[node.id]['weight'] += 1

        # 添加边
        for edge in doc.edges:
            source, target = edge.source, edge.target

            # 跳过自环
            if source == target:
                continue

            # 确保节点存在
            if source not in self.graph:
                self.graph.add_node(source, type="unknown", weight=1)
            if target not in self.graph:
                self.graph.add_node(target, type="unknown", weight=1)

            # 添加或更新边
            if not self.graph.has_edge(source, target):
                self.graph.add_edge(
                    source, target,
                    key=edge.relation,
                    weight=1,
                    **edge.properties
                )
            else:
                # 检查是否已有相同关系类型的边
                edge_found = False
                for key, data in self.graph[source][target].items():
                    if key == edge.relation:
                        data['weight'] += 1
                        edge_found = True
                        break

                if not edge_found:
                    self.graph.add_edge(
                        source, target,
                        key=edge.relation,
                        weight=1,
                        **edge.properties
                    )

        return self

    def add_documents(self, documents: List[GraphDocumentWrapper]) -> "GraphBuilder":
        """
        批量添加 GraphDocument 到图。

        Args:
            documents: GraphDocumentWrapper 列表

        Returns:
            self
        """
        for doc in tqdm(documents, desc="Adding documents to graph"):
            self.add_document(doc)
        return self

    def build_from_documents(self, documents: List[GraphDocumentWrapper]) -> nx.MultiDiGraph:
        """
        从文档列表构建图。

        Args:
            documents: GraphDocumentWrapper 列表

        Returns:
            NetworkX 图
        """
        self.add_documents(documents)
        return self.graph

    def remove_self_loops(self) -> "GraphBuilder":
        """移除自环"""
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        return self

    def get_graph(self) -> nx.MultiDiGraph:
        """获取当前图"""
        return self.graph

    def stats(self) -> Dict[str, int]:
        """获取图统计信息"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges()
        }

