import os
from openai import OpenAI
from src.GraphRAG import GraphRAG
from src.Chunker import Chunker
from src.GraphBuilder import GraphBuilder

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

chunker = Chunker(chunk_size=512, overlap=50, encoding_model="cl100k_base")
chunks_path = 'checkpoints/chunks.json'
if os.path.exists(chunks_path):
    chunks = chunker.load_chunks(chunks_path)
else:
    chunks = chunker.chunk_file(file_path='data/book.txt')
    chunker.save_chunks(chunks_path, chunks=chunks)


client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'])
graph_builder = GraphBuilder(client=client)
graph_path = 'checkpoints/graph.json'
if os.path.exists(graph_path):
    graph = graph_builder.load_graph(graph_path)
else:
    graph = graph_builder.build_graph_with_chunks(chunks)
    graph_builder.save_graph(graph_path, graph=graph)