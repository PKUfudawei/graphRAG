"""
GraphRAG Indexer - 整合向量索引和知识图谱索引
"""
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument

from graphrag.graph.builder import get_graph_builder
from rag.indexer import Indexer


class GraphRAGIndexer(Indexer):
    """GraphRAG 索引器 - 同时索引向量数据库和知识图谱

    工作流程:
    1. index_documents(): 分块文档（继承自 Indexer）
    2. build_graph_from_chunks(): 从 chunks 提取实体 -> 对齐 -> 建图
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

    def build_graph_from_chunks(self, chunks: List[Document]) -> dict:
        """从 chunks 构建知识图谱（提取 -> 对齐 -> 建图）。

        Args:
            chunks: 分块后的 Document 列表。

        Returns:
            统计信息字典。
        """
        return self.graph_builder.build_from_documents(chunks)

    def clear_graph(self):
        """清空图谱。"""
        self.graph_builder.clear_graph()


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

    # 测试 4: 查看实体 chunk_ids
    print("\n[Test 4] Check entity chunk_ids...")
    for entity in list(indexer.graph_builder._graph.nodes())[:5]:
        node_data = indexer.graph_builder._graph.nodes[entity]
        chunk_ids = node_data.get("chunk_ids", [])
        node_type = node_data.get("node_type", "N/A")
        print(f"  - {entity} ({node_type}): chunk_ids={chunk_ids}")
    print("  ✓ Passed")

    # 测试 5: 向量索引测试
    print("\n[Test 5] Build vectorstore...")
    vectorstore = indexer.build_vectorstore(chunks)
    print(f"  Vectorstore built with {vectorstore.index.ntotal} documents")
    print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
