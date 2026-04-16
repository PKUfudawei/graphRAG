"""
GraphRAG Indexer - 整合向量索引和知识图谱索引
"""
from typing import List, Optional
import numpy as np
import faiss

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument

from graphrag.graph.builder import get_graph_builder
from rag.indexer import Indexer


class GraphRAGIndexer(Indexer):
    """GraphRAG 索引器 - 同时索引向量数据库和知识图谱

    工作流程:
    1. index_documents(): 分块文档（继承自 Indexer）
    2. build_graph_from_chunks(): 从 chunks 提取实体 -> 对齐 -> 建图
    3. index_entities(): 为实体生成 embedding 并建立向量索引
    """

    def __init__(
        self,
        chunker=None,
        embedding=None,
    ):
        super().__init__(chunker=chunker, embedding=embedding)
        self._graph_builder = None
        self._entity_index = None  # FAISS index for entities
        self._entity_metadata = []  # List of (entity_id, node_type, chunk_ids)

    @property
    def graph_builder(self):
        if self._graph_builder is None:
            self._graph_builder = get_graph_builder()
        return self._graph_builder

    def build_graph_from_chunks(self, chunks: List[Document]) -> dict:
        """从 chunks 构建知识图谱（提取 -> 对齐 -> 建图）。

        Args:
            chunks: 分块后的 Document 列表。

        Returns:
            统计信息字典。
        """
        return self.graph_builder.build_from_documents(chunks)

    def index_entities(self, reset: bool = False) -> int:
        """为实体生成 embedding 并建立 FAISS 向量索引。

        参考 rag.indexer.index_documents 的设计。

        Args:
            reset: 是否重置索引。默认 False。

        Returns:
            索引的实体数量。
        """
        if reset or self._entity_index is None:
            self._entity_index = None
            self._entity_metadata = []

        # 从 builder 获取所有实体
        entity_names = list(self.graph_builder.graph.nodes())

        if not entity_names:
            print("No entities to index.")
            return 0

        # 收集实体元数据
        for entity in entity_names:
            node_data = self.graph_builder.graph.nodes[entity]
            self._entity_metadata.append({
                "entity_id": entity,
                "node_type": node_data.get("node_type", "Entity"),
                "chunk_ids": node_data.get("chunk_ids", [])
            })

        # 生成 embeddings
        print(f"Generating embeddings for {len(entity_names)} entities...")
        embeddings = self.embedding.encode(entity_names)

        # 创建 FAISS 索引
        dim = embeddings.shape[1]
        if self._entity_index is None:
            self._entity_index = faiss.IndexFlatIP(dim)  # 内积相似度
        self._entity_index.add(embeddings)

        print(f"Indexed {len(entity_names)} entities with {dim}D embeddings")
        return len(entity_names)

    def clear_graph(self):
        """清空图谱和实体索引。"""
        self.graph_builder.clear_graph()
        self._entity_index = None
        self._entity_metadata = []

    def save(self, storage_path: str):
        """保存索引到磁盘。

        Args:
            storage_path: 存储目录路径。
        """
        import os
        import pickle

        os.makedirs(storage_path, exist_ok=True)

        # 保存图谱
        graph_path = os.path.join(storage_path, "graph.pkl")
        self.graph_builder._save_graph()
        # 复制文件到 storage_path
        import shutil
        shutil.copy(self.graph_builder.storage_path, graph_path)

        # 保存实体索引
        if self._entity_index is not None:
            entity_index_path = os.path.join(storage_path, "entity_index.pkl")
            with open(entity_index_path, "wb") as f:
                pickle.dump({
                    "index": self._entity_index,
                    "metadata": self._entity_metadata
                }, f)

        print(f"Saved index to {storage_path}")

    def load(self, storage_path: str):
        """从磁盘加载索引。

        Args:
            storage_path: 存储目录路径。
        """
        import os
        import pickle

        # 加载图谱
        graph_path = os.path.join(storage_path, "graph.pkl")
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                self.graph_builder._graph = pickle.load(f)

        # 加载实体索引
        entity_index_path = os.path.join(storage_path, "entity_index.pkl")
        if os.path.exists(entity_index_path):
            with open(entity_index_path, "rb") as f:
                data = pickle.load(f)
                self._entity_index = data["index"]
                self._entity_metadata = data["metadata"]

        print(f"Loaded index from {storage_path}")


def get_graphrag_indexer(
    chunker=None,
    embedding=None,
) -> GraphRAGIndexer:
    """获取 GraphRAG 索引器实例

    Args:
        chunker: 预创建的 chunker 实例
        embedding: 预创建的 Embeddings 实例

    Returns:
        GraphRAGIndexer 实例
    """
    return GraphRAGIndexer(
        chunker=chunker,
        embedding=embedding,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("GraphRAGIndexer 测试")
    print("=" * 60)

    # 测试 1: 创建索引器
    print("\n[Test 1] Create GraphRAGIndexer...")
    indexer = get_graphrag_indexer()
    print("  ✓ Passed")

    # 测试 2: 分块测试
    print("\n[Test 2] Index documents (chunking + chunk_id)...")
    texts = [
        "北京是中国的首都，位于华北平原。北京拥有丰富的历史文化遗产。",
        "上海是中国最大的城市，位于长江入海口。上海是国际金融中心。",
    ]
    documents = [Document(page_content=t, metadata={"source": f"text{i+1}"}) for i, t in enumerate(texts)]
    chunks = indexer.index_documents(documents)
    print(f"  Generated {len(chunks)} chunks")
    print("  ✓ Passed")

    # 测试 3: 图谱构建测试（提取 -> 对齐 -> 建图）
    print("\n[Test 3] Build graph from chunks...")
    indexer.clear_graph()
    result = indexer.build_graph_from_chunks(chunks)
    print(f"  Graph built: {result['entities']} entities, {result['relationships']} relationships")
    print(f"  Alias groups: {result['alias_groups']}")
    print("  ✓ Passed")

    # 测试 4: 实体向量索引
    print("\n[Test 4] Index entities...")
    indexed_count = indexer.index_entities()
    print(f"  Indexed {indexed_count} entities")
    print("  ✓ Passed")

    # 测试 5: 保存和加载
    print("\n[Test 5] Save and load index...")
    indexer.save("/tmp/test_graphrag_index")
    new_indexer = get_graphrag_indexer()
    new_indexer.load("/tmp/test_graphrag_index")
    print(f"  Loaded graph: {new_indexer.graph_builder.stats()}")
    print(f"  Loaded entity index: {new_indexer._entity_index.ntotal if new_indexer._entity_index else 0} entities")
    print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
