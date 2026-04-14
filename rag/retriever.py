from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class Retriever:
    """RAG 检索器，支持向量检索和重排序"""

    def __init__(
        self,
        vectorstore: FAISS,
        reranker: Optional[Any] = None,
        k: int = 5,
    ):
        """
        初始化检索器

        Args:
            vectorstore: FAISS 向量存储
            reranker: 可选的重排序器
            k: 返回的文档数量
        """
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.k = k

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询文本
            k: 返回的文档数量，默认使用初始化时的 k 值

        Returns:
            检索到的文档列表
        """
        k = k or self.k

        # 使用 FAISS 进行相似度搜索
        docs = self.vectorstore.similarity_search(query, k=k)

        # 如果有 reranker，进行重排序
        if self.reranker is not None:
            docs = self._rerank(query, docs)

        return docs

    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """
        检索相关文档并返回相似度分数

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            (文档，分数) 元组列表
        """
        k = k or self.k
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        使用 reranker 对文档进行重排序

        Args:
            query: 查询文本
            documents: 待重排序的文档列表

        Returns:
            重排序后的文档列表
        """
        if self.reranker is None:
            return documents

        # 使用 CrossEncoderReranker 进行重排序
        result = self.reranker.compress_documents(
            query=query,
            documents=documents,
        )
        return list(result)

    def hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
        vector_weight: float = 0.7,
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
        k = k or self.k

        # 向量检索
        vector_results = self.vectorstore.similarity_search(query, k=k)

        # 如果有 reranker，使用 reranker 进行最终排序
        if self.reranker is not None:
            return self._rerank(query, vector_results)

        return vector_results
