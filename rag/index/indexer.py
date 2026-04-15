"""
Indexer 模块 - 整合 chunker 和 embedder 进行文档索引
"""

import os
from typing import List, Optional
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS

try:
    from .chunker import get_chunker as _get_chunker
    from .embedder import get_embedder as _get_embedder
except ImportError:
    from chunker import get_chunker as _get_chunker
    from embedder import get_embedder as _get_embedder


class Indexer:
    """文档索引器 - 整合分块和嵌入功能"""

    def __init__(
        self,
        chunker: Optional[TokenTextSplitter] = None,
        embeddings: Optional[Embeddings] = None,
        chunk_model: str = "cl100k_base",
        chunk_size: int = 512,
        overlap: int = 50,
        embed_model: str = "BAAI/bge-m3",
        embed_device: str = "cuda:0",
    ):
        """
        初始化索引器

        Args:
            chunker: 预创建的 TokenTextSplitter 实例
            embeddings: 预创建的 Embeddings 实例
            chunk_size: 每个 chunk 的 token 数量
            overlap: 相邻 chunk 之间的重叠 token 数量
            embed_model: 嵌入模型名称
            embed_device: 嵌入模型设备
            chunk_model: tiktoken encoding 名称
        """
        self.chunker = chunker or _get_chunker(
            model=chunk_model,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        self.embeddings = embeddings or _get_embedder(
            model=embed_model,
            device=embed_device,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        分块文档列表

        Args:
            documents: 文档列表

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

    def chunk_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
        """
        分块文本列表

        Args:
            texts: 文本列表
            metadatas: 元数据列表（可选）

        Returns:
            分块后的文档列表
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        all_chunks = []
        for text, metadata in tqdm(zip(texts, metadatas), desc="Chunking"):
            chunks = self.chunker.create_documents(
                [text],
                metadatas=[metadata]
            )
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
        return FAISS.from_documents(documents, self.embeddings)

    def index_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        build_vectorstore: bool = True
    ) -> tuple[List[Document], Optional[FAISS]]:
        """
        一站式索引文本（分块 + 嵌入 + 构建索引）

        Args:
            texts: 文本列表
            metadatas: 元数据列表
            build_vectorstore: 是否构建向量索引

        Returns:
            (分块后的文档列表，FAISS 向量存储实例或 None)
        """
        documents = self.chunk_texts(texts, metadatas)
        vectorstore = self.build_vectorstore(documents) if build_vectorstore else None
        return documents, vectorstore

    def index_files(
        self,
        file_paths: List[str],
        encoding: str = "utf-8",
        build_vectorstore: bool = True
    ) -> tuple[List[Document], Optional[FAISS]]:
        """
        一站式索引文件（支持 Markdown 和纯文本）

        Args:
            file_paths: 文件路径列表
            encoding: 文件编码（仅用于 txt 文件）
            build_vectorstore: 是否构建向量索引

        Returns:
            (分块后的文档列表，FAISS 向量存储实例或 None)
        """
        from pathlib import Path

        all_docs = []

        for path in tqdm(file_paths, desc="Indexing files"):
            file_path = Path(path)
            ext = file_path.suffix.lower()

            if ext in {".md", ".markdown"}:
                # 使用 langchain 的 Markdown 加载器
                try:
                    from langchain_community.document_loaders import UnstructuredMarkdownLoader
                    loader = UnstructuredMarkdownLoader(str(path))
                    docs = loader.load()
                    # 统一添加 source 元数据
                    for doc in docs:
                        doc.metadata["source"] = str(path)
                    all_docs.extend(docs)
                except ImportError:
                    # 降级为纯文本读取
                    with open(path, "r", encoding=encoding) as f:
                        text = f.read()
                    all_docs.append(Document(page_content=text, metadata={"source": str(path)}))
            else:
                # 纯文本文件
                with open(path, "r", encoding=encoding) as f:
                    text = f.read()
                all_docs.append(Document(page_content=text, metadata={"source": str(path)}))

        # 构建向量索引
        vectorstore = self.build_vectorstore(all_docs) if build_vectorstore else None
        return all_docs, vectorstore

    def save_vectorstore(self, vectorstore: FAISS, path: str):
        """保存向量存储"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        vectorstore.save_local(path)
        print(f"Vectorstore saved to {path}")

    def load_vectorstore(self, path: str) -> FAISS:
        """加载向量存储"""
        return FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )


def get_indexer(
    chunk_model: str = "cl100k_base",
    chunk_size: int = 512,
    overlap: int = 50,
    embed_model: str = "BAAI/bge-m3",
    embed_device: str = "cuda:0",
) -> Indexer:
    """
    获取索引器实例

    Args:
        chunk_size: 每个 chunk 的 token 数量
        overlap: 相邻 chunk 之间的重叠 token 数量
        embed_model: 嵌入模型名称
        embed_device: 嵌入模型设备
        chunk_model: tiktoken encoding 名称

    Returns:
        Indexer 实例
    """
    return Indexer(
        chunk_model=chunk_model,
        chunk_size=chunk_size,
        overlap=overlap,
        embed_model=embed_model,
        embed_device=embed_device,
    )


if __name__ == "__main__":
    # 测试
    indexer = get_indexer(chunk_size=100, overlap=20)

    texts = [
        "This is a test document. " * 20,
        "This is another document. " * 20,
    ]

    documents, vectorstore = indexer.index_texts(texts, metadatas=[{"source": "test1"}, {"source": "test2"}])
    print(f"Generated {len(documents)} chunks")
    print(f"Vectorstore size: {vectorstore.index.ntotal}")
