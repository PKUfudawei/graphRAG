from langchain_text_splitters import TokenTextSplitter


def get_chunker(model="cl100k_base", chunk_size=512, overlap=50, **kwargs):
    """
    获取文本分块器实例

    Args:
        model: tiktoken encoding 名称 (cl100k_base 用于 gpt-3.5-turbo, gpt-4 等)
        chunk_size: 每个 chunk 的 token 数量
        overlap: 相邻 chunk 之间的重叠 token 数量
        **kwargs: 其他参数传递给 TokenTextSplitter

    Returns:
        TokenTextSplitter 实例
    """
    return TokenTextSplitter.from_tiktoken_encoder(
        encoding_name=model,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        **kwargs
    )


if __name__ == "__main__":
    # 获取分块器实例
    chunker = get_chunker()

    # 测试文本
    text = """
    This is a test text for the chunker functionality.
    This is the second paragraph with more information.
    This is the third paragraph with additional details.
    This is the fourth paragraph to test proper splitting.
    This is the fifth paragraph to ensure the chunker works correctly.
    """ * 10

    # 分块
    documents = chunker.create_documents([text], metadatas=[{"source": "test"}])
    print(f"Generated {len(documents)} chunks")

    # 验证 token 数量
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    for i, doc in enumerate(documents[:3]):
        token_count = len(encoding.encode(doc.page_content))
        print(f"Chunk {i}: {token_count} tokens - {doc.page_content[:60]}...")
