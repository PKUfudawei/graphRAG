import os
from openai import OpenAI
#from src.GraphRAG import GraphRAG
from src.Chunker import Chunker
from src.GraphBuilder import GraphBuilder
from src.CommunityAnalyzer import CommunityAnalyzer
from src.LLM import vLLMInterface
from sentence_transformers import SentenceTransformer


def chunk_file(path, embed_model):
    chunker = Chunker(
        chunk_size=256, overlap=32, encoding_model="cl100k_base", 
        embed_model=embed_model
    )

    chunks = chunker.chunk_file(
        path=path, save_chunks_path='checkpoints/chunks.json', 
        save_index_path='checkpoints/faiss.index'
    )
    return chunks


def build_graph(chunks, LLM, embed_model):
    graph_builder = GraphBuilder(LLM=LLM, embed_model=embed_model)
    graph_path = 'checkpoints/graph.json'
    if os.path.exists(graph_path):
        graph = graph_builder.load_graph(graph_path)
    else:
        graph = graph_builder.build_graph(chunks)
        graph_builder.save_graph(graph_path)

    return graph


def build_communities(graph):
    community_analyzer = CommunityAnalyzer()
    graph, hierarchy, global_summary = community_analyzer.analyze(graph)


def main():
    embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda:1")
    print(f"Embedding model: BAAI/bge-m3 with {sum(p.numel() for p in embed_model.parameters()):,} parameters")
    vllm = vLLMInterface(base_url="http://localhost:8000/v1", model="Qwen/Qwen3.5-9B", temperature=0.1, stream=False, enable_thinking=True)
    print(f"LLM model: Qwen/Qwen3.5-9B")
    chunks = chunk_file(path='data/book.txt', embed_model=embed_model)
    build_graph(chunks=chunks, LLM=vllm, embed_model=embed_model)


if __name__ == "__main__":
    main()
