"""
Chunker 模块 - 文档分块功能
"""

import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from tqdm import tqdm


class Chunker(TokenTextSplitter):
    """自定义分块器，支持在指定关键词处截断文档"""

    def __init__(
        self,
        model: str = "cl100k_base",
        chunk_size: int = 512,
        overlap: int = 50,
        truncations: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        初始化分块器

        Args:
            model: tiktoken encoding 名称
            chunk_size: 每个 chunk 的 token 数量
            overlap: 相邻 chunk 之间的重叠 token 数量
            truncations: 在哪些关键词处截断文档（不区分大小写）
            **kwargs: 其他参数传递给 TokenTextSplitter
        """
        import tiktoken

        # 保存 truncations 到实例（None 表示不截断）
        self.truncations = truncations
        # 编译正则表达式
        if self.truncations:
            patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in self.truncations]
            self._patterns = patterns
        else:
            self._patterns = []

        # 调用父类初始化
        TokenTextSplitter.__init__(
            self,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            **kwargs,
        )
        # 设置 encoding
        self._encoding_name = model
        self._encoding = tiktoken.get_encoding(model)

    def truncate(self, text: str) -> str:
        """
        在 truncations 关键词处截断文本

        Args:
            text: 原始文本

        Returns:
            截断后的文本（只保留第一个截断点之前的内容）
        """
        if not self._patterns:
            return text

        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                return text[:match.start()]

        return text

    def split_text(self, text: str) -> List[str]:
        """
        先截断再分块

        Args:
            text: 原始文本

        Returns:
            分块后的文本列表
        """
        truncated = self.truncate(text)
        return super().split_text(truncated)

    def split_documents(
        self, documents: List[Document], **kwargs
    ) -> List[Document]:
        """
        先截断文档再分块

        Args:
            documents: 原始文档列表

        Returns:
            分块后的文档列表
        """
        texts, metadatas = [], []
        for doc in tqdm(documents, desc='Truncating documents'):
            truncated_text = self.truncate(doc.page_content)
            texts.append(truncated_text)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)


def get_chunker(
    model: str = "cl100k_base",
    chunk_size: int = 512,
    overlap: int = 50,
    truncations: Optional[List[str]] = None,
    **kwargs,
) -> Chunker:
    """
    获取分块器实例

    Args:
        model: tiktoken encoding 名称
        chunk_size: 每个 chunk 的 token 数量
        overlap: 相邻 chunk 之间的重叠 token 数量
        truncations: 在哪些关键词处截断文档（不区分大小写）
                     None (默认) 使用默认截断词 ['acknowledgement', 'acknowledgment']
                     [] 禁用截断
                     列表 使用自定义截断词
        **kwargs: 其他参数传递给 Chunker

    Returns:
        Chunker 实例
    """
    # 处理 truncations 参数：None 表示使用默认值，空列表表示禁用
    print(f"Chunker truncations for each document: {truncations}")

    return Chunker(
        model=model,
        chunk_size=chunk_size,
        overlap=overlap,
        truncations=truncations,
        **kwargs,
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

    # 将 texts 转换为 Document 对象用于 split_documents 测试
    documents = [Document(page_content=t, metadata={"source": f"doc_{i}"}) for i, t in enumerate(texts)]

    # 测试 1: 默认 truncations=['acknowledgement', 'acknowledgment'] - split_documents
    print("=" * 60)
    print("Test 1:")
    chunker1 = get_chunker(chunk_size=100, overlap=20)
    result1 = chunker1.split_documents(documents)
    for i, doc in enumerate(result1):
        print(f"  Chunk {i}: {doc.page_content.strip()}")

    # 测试 2: truncations=[] (禁用) - split_documents
    print()
    print("=" * 60)
    print("Test 2:")
    chunker2 = get_chunker(chunk_size=100, overlap=20, truncations=[])
    result2 = chunker2.split_documents(documents)
    for i, doc in enumerate(result2):
        print(f"  Chunk {i}: {doc.page_content.strip()}")

    # 测试 3: 自定义 truncations=['topic'] - split_documents
    print()
    print("=" * 60)
    print("Test 3:")
    chunker3 = get_chunker(chunk_size=200, overlap=20, truncations=['topic'])
    result3 = chunker3.split_documents(documents)
    for i, doc in enumerate(result3):
        print(f"  Chunk {i}: {doc.page_content.strip()}")

    # 测试 4: 自定义 truncations=['#'] (在 markdown 标题处截断) - split_documents
    print()
    print("=" * 60)
    print("Test 4:")
    chunker4 = get_chunker(chunk_size=200, overlap=20, truncations=['acknowledgement'])
    result4 = chunker4.split_documents(documents)
    for i, doc in enumerate(result4):
        print(f"- Chunk {i}: {doc.page_content.strip()}")
