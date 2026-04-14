import os
from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class SimpleReranker:
    """简单的重排序器，基于 CrossEncoder"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda:0"):
        self.model = CrossEncoder(model_name, device=device)

    def compress_documents(self, query: str, documents: List[Document]):
        """对文档进行重排序"""
        texts = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(texts)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs]

    def _rerank(self, query: str, documents: List[Document], top_n: int = 3):
        """返回带分数的重排序结果"""
        texts = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(texts)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        from dataclasses import dataclass
        @dataclass
        class RankedDocument:
            document: Document
            relevance_score: float

        return [RankedDocument(doc, score) for doc, score in scored_docs[:top_n]]


def get_reranker(device: str = "cuda:0") -> SimpleReranker:
    """
    获取重排序器实例

    Args:
        device:

    Returns:
        SimpleReranker 实例
    """
    return SimpleReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device=device,
    )


if __name__ == "__main__":
    # 查询
    query = "What is the capital of China?"

    # 候选文档列表
    raw_documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "Beijing is the capital city of the People's Republic of China.",
        "Python is a programming language.",
    ]
    documents = [Document(page_content=doc) for doc in raw_documents]

    # 创建 reranker 实例
    reranker = get_reranker(device="cuda:0")

    # 使用 reranker 对文档进行重排序
    result = reranker.compress_documents(
        query=query,
        documents=documents,
    )

    # 输出重排序后的文档及其分数
    print("重排序结果:")
    for doc in result:
        print(f"  {doc.page_content[:50]}...")

    # 获取带分数的详细结果
    result_with_scores = reranker._rerank(
        query=query,
        documents=documents,
        top_n=3,
    )

    print("\nTop 3 文档及分数:")
    for item in result_with_scores:
        print(f"  分数：{item.relevance_score:.4f}")
        print(f"  内容：{item.document.page_content[:60]}...", end='\n')
