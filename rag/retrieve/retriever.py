"""
Retriever 模块 - 支持向量检索和重排序
"""

from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

try:
    from .reranker import get_reranker as _get_reranker
except ImportError:
    from reranker import get_reranker as _get_reranker


class Retriever:
    """RAG 检索器，支持向量检索和重排序"""

    def __init__(
        self,
        vectorstore: FAISS,
        reranker: Optional[Any] = None,
        top_k: int = 5,
    ):
        """
        初始化检索器

        Args:
            vectorstore: FAISS 向量存储
            reranker: 可选的重排序器实例
            top_k: 返回的文档数量
            rerank_model: reranker 模型名称（当 reranker 为 None 时使用）
            rerank_device: reranker 模型设备
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.reranker = reranker

    def retrieve(self, query: str) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询文本
            k: 返回的文档数量，默认使用初始化时的 top_k 值

        Returns:
            检索到的文档列表
        """
        # 使用 FAISS 进行相似度搜索
        docs = self.vectorstore.similarity_search(query, k=self.top_k)

        # 如果有 reranker，进行重排序
        if self.reranker:
            docs = self.reranker.rerank(query, docs)

        return docs

    def hybrid_search(
        self,
        query: str,
    ) -> List[Document]:
        """
        混合检索（向量检索 + 关键词检索）

        Args:
            query: 查询文本
            k: 返回的文档数量
            vector_weight: 向量检索的权重

        Returns:
            混合检索后的文档列表
        """
        # 向量检索
        vector_results = self.vectorstore.similarity_search(query, k=self.top_k)

        # 如果有 reranker，使用 reranker 进行最终排序
        if self.reranker is not None:
            return self.reranker.rerank(query, vector_results)

        return vector_results


def get_retriever(
    vectorstore: FAISS,
    top_k: int = 5,
    reranker_model: Optional[str] = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cuda:1",
    rerank_top_k = None,
) -> Retriever:
    """
    获取检索器实例

    Args:
        vectorstore: FAISS 向量存储
        top_k: 返回的文档数量
        reranker_model: reranker 模型名称（设为 None 则不使用 reranker）
        reranker_device: reranker 模型设备

    Returns:
        Retriever 实例
    """
    
    reranker = _get_reranker(
        model=reranker_model, 
        device=reranker_device, 
        top_k=rerank_top_k if rerank_top_k else top_k,
    ) if reranker_model else None
    return Retriever(
        vectorstore=vectorstore,
        top_k=top_k,
        reranker=reranker,
    )


if __name__ == "__main__":
    import sys
    import os

    # 添加父目录到路径，以便直接运行时也能导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 测试 - 使用 indexer 创建向量数据库
    from index import get_indexer

    print("=" * 60)
    print("Retriever 模块测试")
    print("=" * 60)

    # 使用 indexer 创建向量索引
    indexer = get_indexer(
        chunk_model="cl100k_base",
        chunk_size=100,
        overlap=20,
        embed_model="BAAI/bge-m3",
        embed_device="cuda:0",
    )

    texts = [
        "The capital of China is Beijing. Beijing is the capital city of the People's Republic of China.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects.",
        "Python is a programming language. Python is widely used for web development and data science.",
        "Machine learning is a subset of artificial intelligence. ML algorithms learn from data.",
        "JavaScript is a programming language for web development. It runs in browsers.",
    ]

    documents, vectorstore = indexer.index_texts(
        texts,
        metadatas=[{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}, {"source": "doc4"}, {"source": "doc5"}]
    )
    print(f"✓ Indexer created {len(documents)} chunks")
    print()

    # 测试 1: 不使用 reranker 的检索器
    print("-" * 40)
    print("测试 1: 不使用 reranker 的检索器")
    print("-" * 40)
    retriever_no_rerank = Retriever(
        vectorstore=vectorstore,
        top_k=3,
        reranker=None,
    )
    results = retriever_no_rerank.retrieve("What is the capital of China?")
    print(f"✓ 检索到 {len(results)} 个文档:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:50]}...")
    print()

    # 测试 2: 使用 get_retriever 创建带 reranker 的检索器
    print("-" * 40)
    print("测试 2: 使用 get_retriever 创建带 reranker 的检索器")
    print("-" * 40)
    retriever = get_retriever(
        vectorstore,
        top_k=5,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_device="cuda:1",
        rerank_top_k=3,
    )
    results = retriever.retrieve("What is the capital of China?")
    print(f"✓ 检索到 {len(results)} 个文档 (rerank_top_k=3):")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:50]}...")
    print()

    # 测试 3: hybrid_search 方法
    print("-" * 40)
    print("测试 3: hybrid_search 方法")
    print("-" * 40)
    results = retriever.hybrid_search("programming language for web")
    print(f"✓ hybrid_search 检索到 {len(results)} 个文档:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:50]}...")
    print()

    # 测试 4: 不同的查询
    print("-" * 40)
    print("测试 4: 不同的查询 - 'machine learning'")
    print("-" * 40)
    results = retriever.retrieve("machine learning")
    print(f"✓ 检索到 {len(results)} 个文档:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:50]}...")
    print()

    print("=" * 60)
    print("所有测试通过!")
    print("=" * 60)
