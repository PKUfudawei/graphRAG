"""
Retriever 模块 - 支持向量检索、关键词检索和重排序
"""

import os
import sys

# 添加父目录到路径，以便直接运行时也能导入
_sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)

from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever as LangchainBM25Retriever
from models.reranker import get_reranker


class Retriever:
    """RAG 检索器，支持向量检索、关键词检索和重排序"""

    def __init__(
        self,
        vectorstore: FAISS,
        reranker: Optional[Any] = None,
        top_k: int = 10,
        bm25_retriever: Optional[LangchainBM25Retriever] = None,
    ):
        """
        初始化检索器

        Args:
            vectorstore: FAISS 向量存储
            reranker: 可选的重排序器实例
            top_k: 返回的文档数量
            bm25_retriever: 可选的 BM25 检索器实例（langchain_community.retrievers.BM25Retriever）
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.reranker = reranker
        self.bm25_retriever = bm25_retriever

    def set_bm25_retriever(self, documents: List[Document]):
        """
        设置 BM25 检索器

        Args:
            documents: 用于构建 BM25 索引的文档列表
        """
        self.bm25_retriever = LangchainBM25Retriever.from_documents(documents)

    def retrieve(self, query: str) -> List[Document]:
        """
        检索相关文档（向量检索）

        Args:
            query: 查询文本

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
        k: Optional[int] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[Document]:
        """
        混合检索（向量检索 + BM25 关键词检索）

        Args:
            query: 查询文本
            k: 返回的文档数量，默认使用初始化时的 top_k 值
            vector_weight: 向量检索的权重（默认 0.5）
            bm25_weight: BM25 检索的权重（默认 0.5）

        Returns:
            混合检索后的文档列表
        """
        k = k or self.top_k
        results = []

        # 向量检索
        vector_results = self.vectorstore.similarity_search(query, k=k)
        vector_scores = self._normalize_scores(len(vector_results))

        # 使用 doc_key 到 result 的映射来避免重复
        doc_map = {}
        for doc, score in zip(vector_results, vector_scores):
            doc_key = hash(doc.page_content) % (10**9)
            doc_map[doc_key] = {"doc": doc, "vector_score": score, "bm25_score": 0}

        # BM25 关键词检索（如果已初始化）
        if self.bm25_retriever is not None:
            bm25_results = self.bm25_retriever.get_relevant_documents(query, k=k)
            bm25_scores = self._normalize_scores(len(bm25_results))

            for i, doc in enumerate(bm25_results):
                doc_key = hash(doc.page_content) % (10**9)
                if doc_key in doc_map:
                    doc_map[doc_key]["bm25_score"] = bm25_scores[i]
                else:
                    doc_map[doc_key] = {"doc": doc, "vector_score": 0, "bm25_score": bm25_scores[i]}

        # 计算融合分数
        results = list(doc_map.values())
        for r in results:
            r["final_score"] = vector_weight * r["vector_score"] + bm25_weight * r["bm25_score"]

        # 按融合分数排序
        results.sort(key=lambda x: x["final_score"], reverse=True)

        # 如果有 reranker，使用 reranker 进行最终排序
        if self.reranker is not None:
            top_docs = [r["doc"] for r in results[: self.reranker.top_k]]
            reranked_docs = self.reranker.rerank(query, top_docs)
            return reranked_docs

        return [r["doc"] for r in results[:k]]

    def _normalize_scores(self, n_docs: int) -> List[float]:
        """
        归一化分数（基于排名）

        Args:
            n_docs: 文档总数

        Returns:
            归一化后的分数列表
        """
        if n_docs == 0:
            return []
        # 使用倒数归一化
        scores = [1 / (i + 1) for i in range(n_docs)]
        total = sum(scores)
        return [s / total for s in scores]


def get_retriever(
    vectorstore: FAISS,
    top_k: int = 5,
    reranker_model: Optional[str] = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cuda:1",
    rerank_top_k: Optional[int] = None,
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

    reranker = get_reranker(
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
    from rag.indexer import get_indexer
    from models.chunker import get_chunker
    from models.embedding import get_embedding

    print("=" * 60)
    print("Retriever 模块测试")
    print("=" * 60)

    # 使用 indexer 创建向量索引
    chunker = get_chunker(model="cl100k_base", chunk_size=100, overlap=20)
    embedding = get_embedding(model="BAAI/bge-m3", device="cuda:0")
    indexer = get_indexer(chunker=chunker, embedding=embedding)

    texts = [
        "The capital of China is Beijing. Beijing is the capital city of the People's Republic of China.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects.",
        "Python is a programming language. Python is widely used for web development and data science.",
        "Machine learning is a subset of artificial intelligence. ML algorithms learn from data.",
        "JavaScript is a programming language for web development. It runs in browsers.",
    ]

    documents = [Document(page_content=t, metadata={"source": f"doc{i+1}"}) for i, t in enumerate(texts)]
    chunks = indexer.index_documents(documents)
    vectorstore = indexer.build_vectorstore(chunks)
    print(f"✓ Indexer created {len(chunks)} chunks")
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

    # 测试 3: hybrid_search 方法（带 BM25）- 需要 rank_bm25 依赖
    print("-" * 40)
    print("测试 3: hybrid_search 方法（带 BM25）")
    print("-" * 40)
    try:
        retriever.set_bm25_retriever(chunks)
        results = retriever.hybrid_search("programming language for web")
        print(f"✓ hybrid_search 检索到 {len(results)} 个文档:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:50]}...")
    except ImportError as e:
        print(f"⚠ 跳过：{e}")
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
