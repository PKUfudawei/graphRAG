"""
Graph RAG Retriever - 基于知识图谱的检索器
"""

import faiss
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_core.documents import Document

from ..models import CommunitySummary


class GraphRetriever:
    """
    知识图谱检索器

    基于社区嵌入向量进行检索，返回相关社区的上下文信息。
    """

    def __init__(
        self,
        community_index: faiss.Index,
        community_ids: np.ndarray,
        community_metadata: List[Dict[str, Any]],
        graph=None,
    ):
        """
        初始化检索器

        Args:
            community_index: FAISS 社区嵌入索引
            community_ids: 社区 ID 数组
            community_metadata: 社区元数据列表
            graph: 可选的 NetworkX 图（用于图遍历扩展）
        """
        self.community_index = community_index
        self.community_ids = community_ids
        self.community_metadata = {
            m["community_id"]: m for m in community_metadata
        }
        self.graph = graph

    def retrieve(
        self,
        query: str,
        embed_model,
        top_k: int = 3,
    ) -> List[Document]:
        """
        检索相关社区上下文

        Args:
            query: 查询文本
            embed_model: 嵌入模型（需有 encode 方法）
            top_k: 返回的社区数量

        Returns:
            Document 列表，包含社区上下文
        """
        # 生成查询嵌入
        query_embedding = embed_model.encode(query)

        # FAISS 搜索
        result_ids, scores = self.community_index.search(
            query_embedding.reshape(1, -1), min(top_k, len(self.community_ids))
        )

        documents = []
        for comm_id, score in zip(result_ids[0], scores[0]):
            metadata = self.community_metadata.get(int(comm_id))
            if metadata is None:
                continue

            # 组装上下文文本
            context_text = self._build_context_text(metadata)

            documents.append(
                Document(
                    page_content=context_text,
                    metadata={
                        "community_id": int(comm_id),
                        "score": float(score),
                        "nodes": metadata.get("nodes", []),
                        "source": "graphrag_community",
                    },
                )
            )

        return documents

    def _build_context_text(self, metadata: Dict[str, Any]) -> str:
        """
        构建社区上下文文本

        Args:
            metadata: 社区元数据

        Returns:
            格式化后的上下文文本
        """
        lines = [
            f"Community {metadata.get('community_id', 'unknown')}: ",
            metadata.get("summary", ""),
            "",
            "Key Relations:",
        ]

        relations = metadata.get("relations", [])
        if relations:
            lines.extend(relations[:10])  # 限制关系数量

        return "\n".join(lines)

    def retrieve_with_graph_walk(
        self,
        query: str,
        embed_model,
        top_k: int = 3,
        walk_depth: int = 1,
    ) -> List[Document]:
        """
        检索 + 图遍历扩展

        Args:
            query: 查询文本
            embed_model: 嵌入模型
            top_k: 返回的社区数量
            walk_depth: 图遍历深度

        Returns:
            Document 列表
        """
        if self.graph is None:
            return self.retrieve(query, embed_model, top_k)

        # 先获取基础检索结果
        docs = self.retrieve(query, embed_model, top_k)

        # TODO: 图遍历扩展（可选增强）
        # 可以从社区节点出发，遍历邻居节点获取更多信息

        return docs


def get_graph_retriever(
    index_path: str,
    metadata_path: str,
    graph_path: Optional[str] = None,
) -> GraphRetriever:
    """
    获取图检索器实例

    Args:
        index_path: 社区嵌入索引路径
        metadata_path: 社区元数据路径
        graph_path: 可选的图文件路径

    Returns:
        GraphRetriever 实例
    """
    import faiss
    import json

    # 加载索引
    index, community_ids = faiss.read_index(index_path), np.load(index_path + ".ids.npy")

    # 加载元数据
    with open(metadata_path, "r", encoding="utf-8") as f:
        community_metadata = json.load(f)

    # 可选：加载图
    graph = None
    if graph_path and Path(graph_path).exists():
        import networkx as nx
        from networkx.readwrite import json_graph

        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = json_graph.node_link_graph(data)

    return GraphRetriever(
        community_index=index,
        community_ids=community_ids,
        community_metadata=community_metadata,
        graph=graph,
    )
