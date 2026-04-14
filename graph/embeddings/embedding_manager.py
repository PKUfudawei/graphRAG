"""
Embedding manager for nodes and communities.
节点和社区的嵌入管理器。
"""
import faiss
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class EmbeddingManager:
    """
    嵌入管理器。
    统一管理节点和社区的嵌入生成、索引和搜索。

    Args:
        embed_model: 嵌入模型，需有 encode 方法
    """

    def __init__(self, embed_model):
        self.embed_model = embed_model

    def encode_texts(self, texts: List[str], batch_size: int = 1000) -> np.ndarray:
        """
        编码文本为嵌入向量。

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            嵌入向量数组 (n, dimension)
        """
        from tqdm import tqdm

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i:i+batch_size]
            embeddings = self.embed_model.encode(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def encode_nodes(self, node_ids: List[str]) -> np.ndarray:
        """
        编码节点 ID 为嵌入向量。

        Args:
            node_ids: 节点 ID 列表

        Returns:
            嵌入向量数组
        """
        return self.encode_texts(node_ids)

    def encode_community_summary(
        self,
        summary: str,
        nodes: List[str],
        relations: List[str]
    ) -> np.ndarray:
        """
        编码社区总结为嵌入向量。

        Args:
            summary: 社区总结文本
            nodes: 节点列表
            relations: 关系列表

        Returns:
            嵌入向量
        """
        embedding_text = f"""
Summary:
{summary}

Entities:
{", ".join(nodes)}

Relations:
{'\n'.join(relations)}
"""
        return self.embed_model.encode(embedding_text)

    def build_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "flat"
    ) -> faiss.Index:
        """
        构建 FAISS 索引。

        Args:
            embeddings: 嵌入向量数组
            index_type: 索引类型 ("flat" 或 "hnsw")

        Returns:
            FAISS 索引
        """
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 50
        else:
            index = faiss.IndexFlatIP(dimension)

        index.add(embeddings)
        return index

    def save_index(
        self,
        index: faiss.Index,
        path: str,
        ids: np.ndarray
    ) -> str:
        """
        保存索引和 ID 映射。

        Args:
            index: FAISS 索引
            path: 保存路径
            ids: ID 数组

        Returns:
            保存的路径
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, path)
        np.save(path + ".ids.npy", ids)

        print(f"\tEmbedding index saved: {path}")
        return path

    @staticmethod
    def load_index(path: str) -> Tuple[faiss.Index, np.ndarray]:
        """
        加载索引和 ID 映射。

        Args:
            path: 索引文件路径

        Returns:
            (FAISS 索引，ID 数组)
        """
        index = faiss.read_index(path)
        ids = np.load(path + ".ids.npy")

        print(f"Embedding index loaded: dimension={index.d}, number of vectors={index.ntotal}")
        return index, ids

    def search(
        self,
        index: faiss.Index,
        query: str,
        ids: np.ndarray,
        k: int = 10
    ) -> Tuple[List, List[float]]:
        """
        搜索最相似的项。

        Args:
            index: FAISS 索引
            query: 查询文本
            ids: ID 数组
            k: 返回结果数量

        Returns:
            (ID 列表，相似度分数列表)
        """
        query_embedding = self.embed_model.encode([query])
        faiss.normalize_L2(query_embedding)

        distances, indices = index.search(query_embedding, k)

        result_ids = [ids[idx] for idx in indices[0]]
        scores = distances[0].tolist()

        return result_ids, scores
