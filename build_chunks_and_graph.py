import os
from openai import OpenAI
from src.GraphRAG import GraphRAG
from src.Chunker import Chunker
from src.GraphBuilder import GraphBuilder
from sentence_transformers import SentenceTransformer

os.makedirs('checkpoints', exist_ok=True)

embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda:3")

chunker = Chunker(
    chunk_size=512, overlap=50, encoding_model="cl100k_base", 
    embed_model=embed_model
)

chunker.chunk_file(
    file_path='data/book.txt', save_chunks_path='checkpoints/chunks.json', 
    save_index_path='checkpoints/faiss.index'
)

"""
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
"""