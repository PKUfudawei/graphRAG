import os
from openai import OpenAI
#from src.GraphRAG import GraphRAG
from src.Chunker import Chunker
from src.GraphBuilder import GraphBuilder
from src.GraphAnalyzer import GraphAnalyzer
from src.LLM import vLLMInterface
from sentence_transformers import SentenceTransformer


def main():
    embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
    print(f"Embedding model: BAAI/bge-m3 with {sum(p.numel() for p in embed_model.parameters()):,} parameters")
    vllm = vLLMInterface(base_url="http://localhost:8000/v1", model="Qwen/Qwen3.5-9B", temperature=0, stream=False, enable_thinking=False)
    chunker = Chunker(
        chunk_size=256, overlap=32, encoding_model="cl100k_base", 
        embed_model=embed_model
    )

    chunks = chunker.chunk_file(
        path='data/book.txt', save_chunks_path='checkpoints/chunks.json', 
        save_index_path='checkpoints/chunks.index'
    )
    graph_builder = GraphBuilder(LLM=vllm, embed_model=embed_model)
    graph_path = 'checkpoints/graph.json'
    if os.path.exists(graph_path):
        graph = graph_builder.load_graph(graph_path)
    else:
        graph = graph_builder.build_graph(chunks)
        graph_builder.save_graph(graph_path)
    
    graph_analyzer = GraphAnalyzer(LLM=vllm, embed_model=embed_model, max_community_size=50)
    graph, communities, community_summaries = graph_analyzer.analyze(graph)
    graph_analyzer.save_faiss_index(
        community_summaries, index_path='checkpoints/community_summaries.index',
        meta_path='checkpoints/community_summaries.pickle'
    )


if __name__ == "__main__":
    main()
