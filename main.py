import os
from openai import OpenAI
#from src.GraphRAG import GraphRAG
from src.Chunker import Chunker
from src.GraphBuilder import GraphBuilder
from src.CommunityAnalyzer import CommunityAnalyzer
from sentence_transformers import SentenceTransformer


if 'OPENROUTER_API_KEY' not in os.environ or 'OPENAI_API_KEY' not in os.environ:
    from dotenv import load_dotenv
    load_dotenv()


def build_chunks(file_path='data/book.txt'):
    embed_model = SentenceTransformer(
        "BAAI/bge-m3", device="cuda:3", 
    )
    chunker = Chunker(
        chunk_size=512, overlap=50, encoding_model="cl100k_base", 
        embed_model=embed_model
    )

    chunker.chunk_file(
        path=file_path, save_chunks_path='checkpoints/chunks.json', 
        save_index_path='checkpoints/faiss.index'
    )


def build_graph(chunks):
    client = OpenAI(
        base_url=os.environ['OPENROUTER_BASE_URL'], 
        api_key=os.environ["OPENROUTER_API_KEY"]
    )
    graph_builder = GraphBuilder(client=client)
    graph_path = 'checkpoints/graph.json'
    if os.path.exists(graph_path):
        graph = graph_builder.load_graph(graph_path)
    else:
        graph = graph_builder.build_graph_with_chunks(chunks)
        graph_builder.save_graph(graph_path, graph=graph)


def build_communities(graph):
    community_analyzer = CommunityAnalyzer()
    graph, hierarchy, global_summary = community_analyzer.analyze(graph)


def main():
    build_chunks()


if __name__ == "__main__":
    main()
