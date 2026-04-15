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

from models.chunker import get_chunker as _get_chunker
from models.embedding import get_embedding as _get_embedder


class Indexer:
    """文档索引器 - 整合分块和嵌入功能"""

    def __init__(
        self,
        chunker: Optional[TokenTextSplitter] = None,
        embeddings: Optional[Embeddings] = None,
        chunk_model: str = "cl100k_base",
        chunk_size: int = 512,
        overlap: int = 50,
        truncations: Optional[List[str]] = None,
        embed_model: str = "BAAI/bge-m3",
        embed_device: str = "cuda:0",
    ):
        """
        初始化索引器

        Args:
            chunker: 预创建的 chunker 实例
            embeddings: 预创建的 Embeddings 实例
            chunk_size: 每个 chunk 的 token 数量
            overlap: 相邻 chunk 之间的重叠 token 数量
            truncations: 在哪些关键词处截断文档
            embed_model: 嵌入模型名称
            embed_device: 嵌入模型设备
            chunk_model: tiktoken encoding 名称
        """
        self.chunker = chunker or _get_chunker(
            model=chunk_model,
            chunk_size=chunk_size,
            overlap=overlap,
            truncations=truncations,
        )
        self.embeddings = embeddings or _get_embedder(
            model=embed_model,
            device=embed_device,
        )

    def index_documents(self, documents: List[Document]) -> List[Document]:
        """
        索引文档列表（分块 + 添加 chunk_id）

        Args:
            documents: langchain Document 列表

        Returns:
            分块后的文档列表（每个 chunk 的 metadata 中包含 chunk_id）
        """
        all_chunks = []
        global_chunk_id = 0
        for doc in tqdm(documents, desc="Indexing"):
            chunks = self.chunker.create_documents(
                [doc.page_content],
                metadatas=[doc.metadata]
            )
            for chunk in chunks:
                chunk.metadata["chunk_id"] = f"chunk_{global_chunk_id}"
                global_chunk_id += 1
            all_chunks.extend(chunks)
        return all_chunks

    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        构建 FAISS 向量索引

        Args:
            documents: 文档列表

        Returns:
            FAISS 向量存储实例
        """
        embeddings = self.embeddings.embed_model if hasattr(self.embeddings, 'embed_model') else self.embeddings
        return FAISS.from_documents(documents, embeddings)

    def save_vectorstore(self, vectorstore: FAISS, path: str):
        """保存向量存储"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        vectorstore.save_local(path)
        print(f"Vectorstore saved to {path}")

    def load_vectorstore(self, path: str) -> FAISS:
        """加载向量存储"""
        embeddings = self.embeddings.embed_model if hasattr(self.embeddings, 'embed_model') else self.embeddings
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )


def get_indexer(
    chunker: Optional[TokenTextSplitter] = None,
    chunk_model: str = "cl100k_base",
    chunk_size: int = 512,
    overlap: int = 50,
    truncations: Optional[List[str]] = None,
    embed_model: str = "BAAI/bge-m3",
    embed_device: str = "cuda:0",
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
    return Indexer(
        chunker=chunker,
        chunk_model=chunk_model,
        chunk_size=chunk_size,
        overlap=overlap,
        truncations=truncations,
        embed_model=embed_model,
        embed_device=embed_device,
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
    indexer1 = get_indexer(chunk_size=100, overlap=20)
    result1 = indexer1.index_documents(documents)
    for i, doc in enumerate(result1):
        print(f"  Chunk {i}: {doc.page_content.strip()}")

    # 测试 2: truncations=[] (禁用)
    print()
    print("=" * 60)
    print("Test 2:")
    indexer2 = get_indexer(chunk_size=100, overlap=20, truncations=[])
    result2 = indexer2.index_documents(documents)
    for i, doc in enumerate(result2):
        print(f"  Chunk {i}: {doc.page_content.strip()}")

    # 测试 3: 自定义 truncations=['topic']
    print()
    print("=" * 60)
    print("Test 3:")
    indexer3 = get_indexer(chunk_size=200, overlap=20, truncations=['topic'])
    result3 = indexer3.index_documents(documents)
    for i, doc in enumerate(result3):
        print(f"  Chunk {i}: {doc.page_content.strip()}")

    # 测试 4: 自定义 truncations=['acknowledgement']
    print()
    print("=" * 60)
    print("Test 4:")
    indexer4 = get_indexer(chunk_size=200, overlap=20, truncations=['acknowledgement'])
    result4 = indexer4.index_documents(documents)
    for i, doc in enumerate(result4):
        print(f"- Chunk {i}: {doc.page_content.strip()}")
