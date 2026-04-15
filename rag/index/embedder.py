from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


def get_embedder(model="BAAI/bge-m3", device="cuda:0") -> HuggingFaceEmbeddings:
    """
    获取嵌入模型实例

    Args:
        device:

    Returns:
        HuggingFaceEmbeddings 实例
    """
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )


if __name__ == "__main__":
    # 获取嵌入模型实例
    embeddings = get_embedder()

    # The queries and documents to embed
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    # Encode the queries and documents using LangChain embeddings
    # Note: embed_query is for single query, embed_documents is for multiple documents
    query_embeddings = [embeddings.embed_query(q) for q in queries]
    document_embeddings = embeddings.embed_documents(documents)

    # Compute cosine similarity manually using numpy
    import numpy as np
    query_array = np.array(query_embeddings)
    doc_array = np.array(document_embeddings)

    # Normalize embeddings
    print("Norm of query:", np.linalg.norm(query_array, axis=-1))
    print("Norm of doc:", np.linalg.norm(doc_array, axis=-1))

    similarity = query_array.dot(doc_array.T)
    print(similarity)
    # Similar to: tensor([[0.7646, 0.1414],
    #                     [0.1355, 0.6000]])
