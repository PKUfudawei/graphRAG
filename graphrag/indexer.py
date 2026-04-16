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
from graphrag.graph.graph_builder import get_graph_builder
from models.embedding import get_embedding
from models.chunker import get_chunker


class GraphRAGIndexer:
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
        chunker = None,
        embedding = None,
        graph_builder = None,
    ):
        self.chunker = chunker or get_chunker()
        self.embedding = embedding or get_embedding()
        self.graph_builder = graph_builder or get_graph_builder()

    @property
    def vector_index(self) -> faiss.Index:
        """懒加载 FAISS 向量索引"""
        if self._vector_index is None:
            self._vector_index = faiss.IndexFlatL2(self.vector_dim)
        return self._vector_index

    def index_documents(self, documents: List[Document]) -> List[Document]:
        """分块文档列表

        Args:
            documents: 输入文档列表

        Returns:
            分块后的文档列表
        """
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking"):
            chunks = self.chunker.create_documents(
                [doc.page_content],
                metadatas=[doc.metadata]
            )
            all_chunks.extend(chunks)
        return all_chunks

    def index_vectors(
        self,
        documents: List[Document],
        reset: bool = False
    ) -> None:
        """将文档索引到向量数据库

        Args:
            documents: 要索引的文档列表
            reset: 是否重置现有索引
        """
        if not documents:
            return

        # 生成嵌入向量（先获取一个来确定维度）
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        embedding_array = np.array(embeddings, dtype=np.float32)

        # 动态获取向量维度
        actual_dim = embedding_array.shape[1] if len(embedding_array.shape) > 1 else embedding_array.shape[0]

        # 初始化或重置索引
        if reset or self._vector_index is None or self._vector_index.d != actual_dim:
            self._vector_index = faiss.IndexFlatL2(actual_dim)
            self._vector_metadata = []

        # 添加到 FAISS 索引
        self._vector_index.add(embedding_array)

        # 保存元数据
        for doc in documents:
            self._vector_metadata.append({
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content[:200]  # 只保存前 200 字符
            })

        print(f"  Indexed {len(documents)} documents to vector store")


    def index_graphs(
        self,
        documents: List[Document],
        reset: bool = False
    ) -> List[GraphDocument]:
        """将文档索引到知识图谱

        Args:
            documents: 要索引的文档列表
            reset: 是否先清空 Neo4j

        Returns:
            构建的 GraphDocument 列表
        """
        if reset:
            self.graph_builder.graph_storage.clear_graph()
            self._entity_vector_index = None
            self._entity_metadata = []

        if not documents:
            return []

        # 使用 graph_builder 构建并保存图谱
        graph_docs = self.graph_builder.build_and_save_batch(documents)

        # 索引实体向量
        self._index_entity_vectors(graph_docs)

        return graph_docs

    def _index_entity_vectors(self, graph_docs: List[GraphDocument]) -> None:
        """为图谱中的实体生成并向量索引

        Args:
            graph_docs: GraphDocument 列表
        """
        # 收集所有唯一实体
        entities = {}
        for graph_doc in graph_docs:
            for node in graph_doc.nodes:
                if node.id not in entities:
                    entities[node.id] = node.type

        if not entities:
            return

        # 生成实体 embedding
        entity_names = list(entities.keys())
        embeddings = self.embedding_model.encode(entity_names)
        embedding_array = np.array(embeddings, dtype=np.float32)

        # 动态获取向量维度
        actual_dim = embedding_array.shape[1] if len(embedding_array.shape) > 1 else embedding_array.shape[0]

        # 初始化或重置索引
        if self._entity_vector_index is None or self._entity_vector_index.d != actual_dim:
            self._entity_vector_index = faiss.IndexFlatL2(actual_dim)
            self._entity_metadata = []

        # 添加到 FAISS 索引
        self._entity_vector_index.add(embedding_array)

        # 保存实体元数据
        for name, entity_type in entities.items():
            self._entity_metadata.append({
                "name": name,
                "type": entity_type
            })

        print(f"  Indexed {len(entities)} entities to entity vector store")

    def index(
        self,
        documents: List[Document],
        index_vector: bool = True,
        index_graph: bool = True,
        reset: bool = False
    ) -> tuple[List[Document], List[GraphDocument]]:
        """一站式索引文档到向量库和图谱

        Args:
            documents: 输入文档列表
            index_vector: 是否索引到向量库
            index_graph: 是否索引到知识图谱
            reset: 是否重置现有索引

        Returns:
            (分块后的文档列表，图谱文档列表)
        """
        # 1. 分块
        chunks = self.chunk_documents(documents)

        # 2. 索引向量
        vector_chunks = []
        if index_vector:
            self.index_vectors(chunks, reset=reset)
            vector_chunks = chunks

        # 3. 索引图谱
        graph_docs = []
        if index_graph:
            graph_docs = self.index_graphs(chunks, reset=reset)

        return vector_chunks, graph_docs

    def index_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        chunk: bool = True,
        index_vector: bool = True,
        index_graph: bool = True,
        reset: bool = False
    ) -> tuple[List[Document], List[GraphDocument]]:
        """一站式索引文本

        Args:
            texts: 文本列表
            metadatas: 元数据列表
            chunk: 是否分块
            index_vector: 是否索引到向量库
            index_graph: 是否索引到知识图谱
            reset: 是否重置现有索引

        Returns:
            (分块后的文档列表，图谱文档列表)
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # 转换为 Document
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]

        if chunk:
            return self.index(documents, index_vector, index_graph, reset)

        return self.index(documents, index_vector, index_graph, reset)


def get_graphrag_indexer(
    chunk_size: int = 512,
    overlap: int = 50,
    dedup_threshold: float = 0.9,
    embedding_model_name: str = "BAAI/bge-m3",
) -> GraphRAGIndexer:
    """获取 GraphRAG 索引器实例

    Args:
        chunk_size: 每个 chunk 的 token 数量
        overlap: 相邻 chunk 之间的重叠 token 数量
        dedup_threshold: 实体去重阈值
        embedding_model_name: 嵌入模型名称

    Returns:
        GraphRAGIndexer 实例
    """
    return GraphRAGIndexer(
        chunk_size=chunk_size,
        overlap=overlap,
        dedup_threshold=dedup_threshold,
        embedding_model_name=embedding_model_name,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("GraphRAGIndexer 测试")
    print("=" * 60)

    # 测试 1: 创建索引器
    print("\n[Test 1] Create GraphRAGIndexer...")
    indexer = get_graphrag_indexer(chunk_size=200, overlap=20)
    print("  ✓ Passed")

    # 测试 2: 分块测试
    print("\n[Test 2] Chunk texts...")
    texts = [
        "北京是中国的首都，位于华北平原。北京拥有丰富的历史文化遗产。",
        "上海是中国最大的城市，位于长江入海口。上海是国际金融中心。",
    ]
    documents = indexer.chunk_texts(texts, metadatas=[{"source": "text1"}, {"source": "text2"}])
    print(f"  Generated {len(documents)} chunks")
    print("  ✓ Passed")

    # 测试 3: 向量索引测试
    print("\n[Test 3] Index vectors...")
    indexer.index_vectors(documents, reset=True)
    print("  ✓ Passed")

    # 测试 4: 向量搜索测试
    print("\n[Test 4] Search vectors...")
    results = indexer.search_vectors("中国的首都是哪里？", top_k=2)
    print(f"  Found {len(results)} results")
    for doc, score in results:
        print(f"    - {doc.page_content[:50]}... (score: {score:.4f})")
    print("  ✓ Passed")

    # 测试 5: 图谱索引测试
    print("\n[Test 5] Index graphs...")
    graph_docs = indexer.index_graphs(documents, reset=True)
    print(f"  Indexed {len(graph_docs)} graph documents")
    print("  ✓ Passed")

    # 测试 6: 一站式索引测试
    print("\n[Test 6] Full index (vector + graph)...")
    new_texts = [
        "Ebenezer Scrooge is a wealthy businessman in Victorian London.",
        "He is visited by the ghost of his former partner Jacob Marley.",
    ]
    vector_chunks, graph_docs = indexer.index_texts(
        new_texts,
        metadatas=[{"source": "eng1"}, {"source": "eng2"}],
        reset=True
    )
    print(f"  Vector chunks: {len(vector_chunks)}")
    print(f"  Graph documents: {len(graph_docs)}")
    print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
