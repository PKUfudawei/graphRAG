"""
GraphRAG Indexer - 整合向量索引和知识图谱索引
"""
import os
import sys
from typing import List, Optional
from tqdm import tqdm
import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_text_splitters import TokenTextSplitter

# 支持直接运行和模块导入
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入依赖
from graphrag.graph.builder import get_graph_builder
from models.embedding import get_embedding
from models.chunker import get_chunker
from rag.indexer import Indexer


class GraphRAGIndexer(Indexer):
    """GraphRAG 索引器 - 同时索引向量数据库和知识图谱

    Args:
        chunk_size: 每个 chunk 的 token 数量
        overlap: 相邻 chunk 之间的重叠 token 数量
        dedup_threshold: 实体去重阈值
        embedding_model_name: 嵌入模型名称
        vector_dim: 向量维度
    """

    def __init__(
        self,
        chunker=None,
        embedding=None,
    ):
        super().__init__(chunker=chunker, embedding=embedding)
        self._graph_builder = None

    @property
    def graph_builder(self):
        if self._graph_builder is None:
            self._graph_builder = get_graph_builder()
        return self._graph_builder

    def build_graph(self, chunk: Document) -> GraphDocument:
        """为单个 chunk 构建图谱"""
        return self.graph_builder.build(chunk, chunk.metadata.get("chunk_id", 0))

    def build_graph_batch(self, chunks: List[Document]) -> List[GraphDocument]:
        """为多个 chunk 构建图谱"""
        return self.graph_builder.build_batch(chunks)

    def extract_and_save_batch(self, chunks: List[Document]) -> List[GraphDocument]:
        """提取图谱并保存到 NetworkX（不去重）"""
        return self.graph_builder.extract_and_save_batch(chunks)

    def align_entities(self) -> dict:
        """实体对齐"""
        return self.graph_builder.align_entities()
    




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

    # 测试 3: 图谱提取测试（增量维护方案）
    print("\n[Test 3] Extract graphs (no dedup)...")
    indexer.graph_builder.clear_graph()
    graph_docs = indexer.extract_and_save_batch(chunks)
    print(f"  Extracted {len(graph_docs)} graph documents")
    stats = indexer.graph_builder.stats()
    print(f"  Graph stats: {stats['num_nodes']} nodes, {stats['num_relationships']} relationships")
    print("  ✓ Passed")

    # 测试 4: 实体对齐测试
    print("\n[Test 4] Align entities...")
    align_result = indexer.align_entities()
    print(f"  Aligned: {align_result['groups_processed']} groups, {align_result['entities_merged']} entities merged")
    stats_after = indexer.graph_builder.stats()
    print(f"  Graph stats after align: {stats_after['num_nodes']} nodes, {stats_after['num_relationships']} relationships")
    print("  ✓ Passed")

    # 测试 5: 向量索引测试
    print("\n[Test 5] Build vectorstore...")
    vectorstore = indexer.build_vectorstore(chunks)
    print(f"  Vectorstore built with {vectorstore.index.ntotal} documents")
    print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
