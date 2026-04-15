"""
Embedding index utilities using FAISS.
使用 FAISS 的嵌入索引管理工具。
"""
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class EmbeddingIndex:
    """
    嵌入索引类。
    使用 FAISS 管理节点和社区嵌入索引。
    """

    @staticmethod
    def build_node_index(
        node_ids: List[str],
        embed_model,
        batch_size: int = 1000
    ) -> Tuple[faiss.Index, np.ndarray]:
        """
        构建节点嵌入索引。

        Args:
            node_ids: 节点 ID 列表
            embed_model: 嵌入模型
            batch_size: 批处理大小

        Returns:
            (FAISS 索引，节点 ID 数组)
        """
        from tqdm import tqdm

        index = None
        all_embeddings = []

        for i in tqdm(range(0, len(node_ids), batch_size), desc="Building node index"):
            batch_nodes = node_ids[i:i+batch_size]
            embeddings = embed_model.encode(batch_nodes)
            faiss.normalize_L2(embeddings)
            all_embeddings.append(embeddings)

            if index is None:
                index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        return index, np.array(node_ids)

    @staticmethod
    def build_community_index(
        community_summaries: List
    ) -> Tuple[faiss.Index, np.ndarray]:
        """
        构建社区嵌入索引。

        Args:
            community_summaries: 社区总结列表（包含 embedding 字段）

        Returns:
            (FAISS 索引，社区 ID 数组)
        """
        embeddings = np.vstack([c["embedding"] if isinstance(c, dict) else c.embedding
                               for c in community_summaries])
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        community_ids = np.array([c["community_id"] if isinstance(c, dict) else c.community_id
                                 for c in community_summaries])

        return index, community_ids

    @staticmethod
    def save_index(
        index: faiss.Index,
        path: str,
        ids: np.ndarray,
        id_type: str = "nodes"
    ) -> str:
        """
        保存 FAISS 索引。

        Args:
            index: FAISS 索引
            path: 保存路径
            ids: ID 数组（节点 ID 或社区 ID）
            id_type: ID 类型（"nodes" 或 "communities"）

        Returns:
            保存的路径
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, path)
        np.save(path + ".ids.npy", ids)

        print(f"\t{id_type.capitalize()} embeddings saved: {path}")
        return path

    @staticmethod
    def load_index(path: str) -> Tuple[faiss.Index, np.ndarray]:
        """
        加载 FAISS 索引。

        Args:
            path: 索引文件路径

        Returns:
            (FAISS 索引，ID 数组)
        """
        index = faiss.read_index(path)
        ids = np.load(path + ".ids.npy")

        print(f"Embedding index loaded: dimension={index.d}, number of vectors={index.ntotal}")
        return index, ids

    @staticmethod
    def search(
        index: faiss.Index,
        query_embedding: np.ndarray,
        ids: np.ndarray,
        k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        在索引中搜索最相似的项。

        Args:
            index: FAISS 索引
            query_embedding: 查询嵌入
            ids: ID 数组
            k: 返回结果数量

        Returns:
            (ID 列表，相似度分数列表)
        """
        faiss.normalize_L2(query_embedding)

        distances, indices = index.search(query_embedding.reshape(1, -1), k)

        result_ids = [ids[idx] for idx in indices[0]]
        scores = distances[0].tolist()

        return result_ids, scores
