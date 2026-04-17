"""
Indexer 模块 - 整合 chunker 和 embedder 进行文档索引
"""

import os
import sys
from typing import List, Optional
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS

sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from models.chunker import get_chunker
from models.embedding import get_embedding


class Indexer:
    """文档索引器 - 整合分块和嵌入功能"""

    def __init__(
        self,
        chunker: Optional[TokenTextSplitter] = None,
        embedding: Optional[Embeddings] = None,
    ):
        """
        初始化索引器

        Args:
            chunker: 预创建的 chunker 实例
            embedding: 预创建的 Embeddings 实例
            chunk_size: 每个 chunk 的 token 数量
            overlap: 相邻 chunk 之间的重叠 token 数量
            truncations: 在哪些关键词处截断文档
            embed_model: 嵌入模型名称
            embed_device: 嵌入模型设备
            chunk_model: tiktoken encoding 名称
        """
        self.chunker = chunker or get_chunker()
        self.embedding = embedding or get_embedding()

    def index_documents(self, documents: List[Document]) -> List[Document]:
        """
        索引文档列表（分块 + 添加 chunk_id）

        Args:
            documents: langchain Document 列表

        Returns:
            分块后的文档列表（每个 chunk 的 metadata 中包含 chunk_id）
        """
        # 使用 split_documents 方法，它会自动添加 chunk_id
        all_chunks = self.chunker.split_documents(documents)
        return all_chunks

    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        构建 FAISS 向量索引

        Args:
            documents: 文档列表

        Returns:
            FAISS 向量存储实例
        """
        embed_model = self.embedding.embed_model if hasattr(self.embedding, 'embed_model') else self.embedding
        return FAISS.from_documents(documents, embed_model)

    def save_vectorstore(self, vectorstore: FAISS, path: str=None):
        """保存向量存储"""
        path = path or self.vectorstore_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        vectorstore.save_local(path)
        print(f"Vectorstore saved to {path}")

    def load_vectorstore(self, path: str) -> FAISS:
        """加载向量存储"""
        embed_model = self.embedding.embed_model if hasattr(self.embedding, 'embed_model') else self.embedding
        return FAISS.load_local(
            path,
            embed_model,
            allow_dangerous_deserialization=True
        )


def get_indexer(
    chunker: Optional[TokenTextSplitter] = None,
    embedding = None,
) -> Indexer:
    """
    获取索引器实例

    Args:
        chunker: 预创建的 chunker 实例
        chunk_size: 每个 chunk 的 token 数量
        overlap: 相邻 chunk 之间的重叠 token 数量
        truncations: 在哪些关键词处截断文档
        embed_model: 嵌入模型名称
        embed_device: 嵌入模型设备
        chunk_model: tiktoken encoding 名称

    Returns:
        Indexer 实例
    """
    chunker = chunker or get_chunker()
    embedding = embedding or get_embedding()
    return Indexer(
        chunker=chunker,
        embedding=embedding,
    )


if __name__ == "__main__":
    texts = ["""
    This is the main content of the document.
    It contains important information about the topic.
    More content here to make it longer.

    # Acknowledgements
    We would like to thank the following people for their contributions.
    This section should be truncated from the main content.
    """] * 3

    # 将 texts 转换为 Document 对象用于测试
    documents = [Document(page_content=t, metadata={"source": f"doc_{i}"}) for i, t in enumerate(texts)]

    # 测试 1: 默认 truncations=['acknowledgement', 'acknowledgment']
    print("=" * 60)
    print("Test 1:")
    chunker = get_chunker(chunk_size=100, overlap=20, truncations=['acknowledgement', 'acknowledgment'])
    indexer1 = get_indexer(chunker=chunker)
    result1 = indexer1.index_documents(documents)
    for i, doc in enumerate(result1):
        print(f"- Chunk {i}: {doc.page_content.strip()}")

    # 测试 2: truncations=[] (禁用)
    print()
    print("=" * 60)
    print("Test 2:")
    chunker = get_chunker(chunk_size=100, overlap=20, truncations=None)
    indexer2 = get_indexer(chunker=chunker)
    result2 = indexer2.index_documents(documents)
    for i, doc in enumerate(result2):
        print(f"- Chunk {i}: {doc.page_content.strip()}")

    # 测试 3: 自定义 truncations=['topic']
    print()
    print("=" * 60)
    print("Test 3:")
    chunker = get_chunker(chunk_size=100, overlap=20, truncations=['topic'])
    indexer3 = get_indexer(chunker=chunker)
    result3 = indexer3.index_documents(documents)
    for i, doc in enumerate(result3):
        print(f"- Chunk {i}: {doc.page_content.strip()}")

    # 测试 4: 自定义 truncations=['acknowledgement']
    print()
    print("=" * 60)
    print("Test 4:")
    chunker = get_chunker(chunk_size=100, overlap=20, truncations=['acknowledgment'])
    indexer4 = get_indexer(chunker=chunker)
    result4 = indexer4.index_documents(documents)
    for i, doc in enumerate(result4):
        print(f"- Chunk {i}: {doc.page_content.strip()}")
