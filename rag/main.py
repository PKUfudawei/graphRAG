"""
RAG 系统主模块 - 使用函数形式演示 Vanilla RAG 流程

流程说明:
1. 文本分块 (Chunking) - 将长文本分割成小块
2. 文本嵌入 (Embedding) - 将文本转换为向量 (GPU 0)
3. 构建索引 (Indexing) - 使用 FAISS 构建向量索引
4. 检索 (Retrieval) - 根据查询检索相关文档
5. 重排序 (Reranking) - 对检索结果重排序 (GPU 1)
6. 生成答案 (Generation) - LLM 根据上下文生成答案 (GPU 2)
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS

try:
    from rag.chunker import Chunker
    from rag.embedder import get_embeddings
    from rag.retriever import Retriever
    from rag.reranker import get_reranker
    from rag.llm import llm
except ImportError:
    from chunker import Chunker
    from embedder import get_embeddings
    from retriever import Retriever
    from reranker import get_reranker
    from llm import llm


# ============================================================================
# 步骤 1: 初始化组件
# ============================================================================

def initialize_components(embed_device="cuda:0", rerank_device="cuda:1"):
    """
    初始化 RAG 所需的所有组件

    Args:
        embed_device: Embedding 模型使用的 GPU
        rerank_device: Reranker 模型使用的 GPU

    Returns:
        dict: 包含所有组件的字典
    """
    print(f"GPU 分配：Embedding={embed_device}, Reranker={rerank_device}")

    # 初始化 Embedding 模型
    embeddings = get_embeddings(device=embed_device)

    # 初始化 Chunker
    chunker = Chunker(
        chunk_size=500,
        overlap=50,
        embed_model="BAAI/bge-m3",
    )
    chunker.embeddings = embeddings  # 注入 embeddings

    # 初始化 Reranker
    reranker = get_reranker(device=rerank_device)

    return {
        "embeddings": embeddings,
        "chunker": chunker,
        "reranker": reranker,
    }


# ============================================================================
# 步骤 2: 文本分块
# ============================================================================

def chunk_texts(texts: List[str], sources: Optional[List[str]] = None,
                chunker: Chunker = None) -> List[Document]:
    """
    将文本列表分块为文档

    Args:
        texts: 文本列表
        sources: 文档来源列表（可选）
        chunker: Chunker 实例

    Returns:
        List[Document]: 分块后的文档列表
    """
    if sources is None:
        sources = [f"document_{i}" for i in range(len(texts))]

    all_documents = []
    for text, source in zip(texts, sources):
        docs = chunker.chunk_text(text, source=source)
        all_documents.extend(docs)

    print(f"文本分块完成，共 {len(all_documents)} 个文档块")
    return all_documents


# ============================================================================
# 步骤 3: 构建向量索引
# ============================================================================

def build_vectorstore(documents: List[Document], chunker: Chunker) -> FAISS:
    """
    构建 FAISS 向量索引

    Args:
        documents: 文档列表
        chunker: 包含 embeddings 的 Chunker 实例

    Returns:
        FAISS: 向量存储实例
    """
    vectorstore = chunker.build_vectorstore(documents)
    print("向量索引构建完成")
    return vectorstore


# ============================================================================
# 步骤 4: 初始化检索器
# ============================================================================

def create_retriever(vectorstore: FAISS, reranker, top_k: int = 5) -> Retriever:
    """
    创建检索器

    Args:
        vectorstore: FAISS 向量存储
        reranker: 重排序器
        top_k: 默认返回的文档数量

    Returns:
        Retriever: 检索器实例
    """
    return Retriever(
        vectorstore=vectorstore,
        reranker=reranker,
        k=top_k,
    )


# ============================================================================
# 步骤 5: 检索文档
# ============================================================================

def retrieve_documents(query: str, retriever: Retriever,
                       k: Optional[int] = None) -> List[Document]:
    """
    检索相关文档

    Args:
        query: 查询文本
        retriever: 检索器实例
        k: 返回的文档数量

    Returns:
        List[Document]: 检索到的文档列表
    """
    docs = retriever.retrieve(query, k=k)
    print(f"  检索到 {len(docs)} 个相关文档")
    return docs


# ============================================================================
# 步骤 6: 生成答案
# ============================================================================

def generate_answer(query: str, context_docs: List[Document]) -> str:
    """
    根据查询和上下文生成答案

    Args:
        query: 查询文本
        context_docs: 上下文文档列表

    Returns:
        str: 生成的答案
    """
    # 构建上下文
    context_text = "\n\n".join([
        f"【文档 {i+1}】:\n{doc.page_content}"
        for i, doc in enumerate(context_docs)
    ])

    # 构建提示
    prompt = f"""你是一个有帮助的 AI 助手。请根据以下参考信息回答问题。
如果参考信息中不包含答案，请诚实地说明你不知道。

参考信息:
{context_text}

问题：{query}

请根据参考信息回答："""

    # 调用 LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ============================================================================
# 完整 RAG 流程
# ============================================================================

def vanilla_rag_query(query: str, retriever: Retriever,
                      max_context_docs: int = 3) -> dict:
    """
    完整的 Vanilla RAG 查询流程

    Args:
        query: 查询文本
        retriever: 检索器实例
        max_context_docs: 用于生成答案的最大文档数量

    Returns:
        dict: 包含答案和参考文档的字典
    """
    # 检索文档
    retrieved_docs = retrieve_documents(query, retriever)

    # 限制上下文文档数量
    context_docs = retrieved_docs[:max_context_docs]
    context_texts = [doc.page_content for doc in context_docs]

    # 生成答案
    answer = generate_answer(query, context_docs)

    return {
        "query": query,
        "answer": answer,
        "references": [
            {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in context_docs
        ],
    }


# ============================================================================
# 辅助函数：保存和加载索引
# ============================================================================

def save_vectorstore(vectorstore: FAISS, chunker: Chunker, path: str):
    """保存向量存储"""
    chunker.save_vectorstore(vectorstore, path)
    print(f"索引已保存到：{path}")


def load_vectorstore(path: str, chunker: Chunker):
    """加载向量存储"""
    print(f"加载向量存储：{path}")
    return chunker.load_vectorstore(path)


# ============================================================================
# 演示：完整的 RAG 流程
# ============================================================================

def demo_vanilla_rag():
    """演示完整的 Vanilla RAG 流程"""
    print("=" * 60)
    print("Vanilla RAG 流程演示")
    print("=" * 60)

    # 测试数据
    test_texts = [
        """
        Python 是一种高级编程语言，由 Guido van Rossum 于 1989 年发明。
        Python 的设计哲学强调代码的可读性和简洁性。
        Python 拥有丰富而强大的库，常被昵称为胶水语言。
        """,
        """
        机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习。
        深度学习是机器学习的一个子集，使用多层神经网络来处理复杂的问题。
        监督学习是机器学习的一种，使用标记的数据来训练模型。
        """,
        """
        北京是中国的首都，位于中国北部，是中国的政治、文化、国际交往和科技创新中心。
        北京拥有 3000 多年的建城史和 850 多年的建都史。
        2008 年，北京成功举办了夏季奥林匹克运动会。
        """,
    ]
    test_sources = ["python_intro.txt", "ml_intro.txt", "beijing_intro.txt"]

    # 步骤 1: 初始化组件
    print("\n[步骤 1] 初始化组件...")
    components = initialize_components(embed_device=0, rerank_device=1)
    chunker = components["chunker"]
    reranker = components["reranker"]

    # 步骤 2: 文本分块
    print("\n[步骤 2] 文本分块...")
    documents = chunk_texts(test_texts, test_sources, chunker)

    # 步骤 3: 构建向量索引
    print("\n[步骤 3] 构建向量索引...")
    vectorstore = build_vectorstore(documents, chunker)

    # 步骤 4: 创建检索器
    print("\n[步骤 4] 创建检索器...")
    retriever = create_retriever(vectorstore, reranker, top_k=3)
    print("检索器创建完成")

    # 步骤 5: 执行查询（包含检索和生成）
    print("\n" + "=" * 60)
    print("[步骤 5] 执行查询")
    print("=" * 60)

    queries = [
        "Python 语言是谁发明的？",
        "什么是深度学习？",
        "北京举办过什么重要的国际赛事？",
    ]

    for query in queries:
        print(f"\n{'-' * 60}")
        print(f"查询：{query}")
        print("-" * 60)

        result = vanilla_rag_query(query, retriever, max_context_docs=3)

        print(f"\n答案：{result['answer']}")

        print("\n参考文档:")
        for i, ref in enumerate(result['references'], 1):
            print(f"  [{i}] 来源：{ref['source']}")
            print(f"      内容：{ref['content'][:100]}...")

    # 保存索引
    print("\n" + "=" * 60)
    print("[额外] 保存索引...")
    print("=" * 60)
    os.makedirs("./data", exist_ok=True)
    save_vectorstore(vectorstore, chunker, "./data/vectorstore_demo")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_vanilla_rag()
