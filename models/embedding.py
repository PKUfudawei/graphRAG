from typing import Optional, List
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


class EmbeddingWrapper:
    """Wrapper to adapt LangChain embed_model to encode interface."""

    def __init__(self, embed_model):
        self.embed_model = embed_model

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the wrapped embed_model."""
        embeddings = self.embed_model.embed_documents(texts)
        return np.array(embeddings, dtype=np.float32)


def get_embedding(model="BAAI/bge-m3", device="cuda:0") -> EmbeddingWrapper:
    """
    获取嵌入模型实例

    Args:
        model: 嵌入模型名称
        device: 运行设备

    Returns:
        EmbedderWrapper 实例
    """
    return EmbeddingWrapper(HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    ))
    


if __name__ == "__main__":
    # 获取嵌入模型实例
    embed_model = get_embedding()

    # The queries and documents to embed
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    # Use the encode method from EmbedderWrapper
    query_embeddings = embed_model.encode(queries)
    document_embeddings = embed_model.encode(documents)

    # Compute cosine similarity manually using numpy
    import numpy as np

    # Normalize embeddings
    print("Norm of query:", np.linalg.norm(query_embeddings, axis=-1))
    print("Norm of doc:", np.linalg.norm(document_embeddings, axis=-1))

    similarity = query_embeddings.dot(document_embeddings.T)
    print(similarity)
    # Similar to: tensor([[0.7646, 0.1414],
    #                     [0.1355, 0.6000]])
