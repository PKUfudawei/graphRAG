"""
Node deduplication using FAISS.
使用 FAISS 进行节点去重和别名合并。
"""
import faiss
import numpy as np
from typing import List, Dict
from collections import Counter


class NodeDeduplicator:
    """
    节点去重器。
    使用嵌入向量和 FAISS HNSW 索引查找相似节点并合并。

    Args:
        embed_model: 嵌入模型，需有 encode 方法
        threshold: 相似度阈值，默认 0.9
    """

    def __init__(self, embed_model, threshold: float = 0.9):
        self.embed_model = embed_model
        self.threshold = threshold

    def find_aliases(self, node_names: List[str]) -> Dict[str, str]:
        """
        查找节点别名。

        Args:
            node_names: 节点名称列表

        Returns:
            别名映射字典 {别名：规范名称}
        """
        if not node_names:
            return {}

        # 计算名称频率
        freq = Counter(node_names)
        unique_names = list(freq.keys())

        # 生成嵌入
        embeddings = self.embed_model.encode(unique_names)
        faiss.normalize_L2(embeddings)

        # 构建 HNSW 索引
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        index.add(embeddings)

        # 使用 Union-Find 聚类相似节点
        alias_map = self._cluster_similar_nodes(index, embeddings, unique_names, freq)

        return alias_map

    def _cluster_similar_nodes(
        self,
        index: faiss.Index,
        embeddings: np.ndarray,
        names: List[str],
        freq: Counter
    ) -> Dict[str, str]:
        """
        使用 Union-Find 聚类相似节点。

        Args:
            index: FAISS 索引
            embeddings: 嵌入向量
            names: 节点名称列表
            freq: 名称频率计数

        Returns:
            别名映射字典
        """
        n = len(names)
        parent = list(range(n))

        def find(x: int) -> int:
            """带路径压缩的查找"""
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            """合并两个集合"""
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pb] = pa

        # 搜索相似节点
        for i, name in enumerate(names):
            D, I = index.search(embeddings[i:i+1], 50)
            for score, idx in zip(D[0], I[0]):
                if score >= self.threshold and idx != i:
                    union(i, idx)

        # 构建聚类
        clusters = {}
        for i, name in enumerate(names):
            root = find(i)
            clusters.setdefault(root, []).append(name)

        # 为每个聚类选择规范名称（频率最高，相同则选字典序）
        alias_map = {}
        for cluster in clusters.values():
            if not cluster:
                continue

            # 选择规范名称：频率最高，相同则选字典序最小的
            canonical = max(cluster, key=lambda x: (freq[x], x))
            alias_map.update({n: canonical for n in cluster})

        return alias_map

