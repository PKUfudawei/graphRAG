import os
from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class SimpleReranker:
    """简单的重排序器，基于 CrossEncoder"""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda:0", top_k=None):
        self.model = CrossEncoder(model, device=device)
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]):
        """对文档进行重排序"""
        texts = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(texts)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if self.top_k is not None:
            scored_docs = scored_docs[:self.top_k]

        return [doc for doc, _ in scored_docs]

    def rerank_with_score(self, query: str, documents: List[Document]):
        """返回带分数的重排序结果"""
        texts = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(texts)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        if self.top_k is not None:
            scored_docs = scored_docs[:self.top_k]

        return scored_docs
    

def get_reranker(model="BAAI/bge-reranker-v2-m3", device='cuda:0', top_k=3):
    return SimpleReranker(
        model=model,
        device=device,
        top_k=top_k
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
    reranker = get_reranker()

    # 使用 reranker 对文档进行重排序
    result = reranker.rerank(
        query=query,
        documents=documents,
    )

    # 输出重排序后的文档及其分数
    print("重排序结果:")
    for doc in result:
        print(f"  {doc.page_content[:50]}...")

    # 获取带分数的详细结果
    result_with_scores = reranker.rerank_with_score(
        query=query,
        documents=documents,
    )

    print(f"\nTop {reranker.top_k} 文档及分数:")
    for item in result_with_scores:
        print(f"  分数：{item[1]:.4f}")
        print(f"  内容：{item[0].page_content[:60]}...", end='\n')
