#!/usr/bin/env python3
import os, argparse
from glob import glob
#from src.GraphRAG import GraphRAG
from src.index.Chunker import Chunker
from src.index.GraphBuilder import GraphBuilder
from src.index.GraphAnalyzer import GraphAnalyzer
from src.LLM import vLLMInterface
from sentence_transformers import SentenceTransformer

os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = "1"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", default="data/meta_markdown/*.md", help="Path to the input document (e.g., .tex, .txt, or .pdf) for building the knowledge graph")
    parser.add_argument("--base_url", default="http://localhost:8000/v1", help="Base URL of the LLM API endpoint (e.g., local vLLM server)")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3.5-9B", help="Name of the LLM used for generation (served via the API)", choices=["Qwen/Qwen3.5-9B"])
    parser.add_argument("--embed_model", default="BAAI/bge-m3", help="Embedding model used to encode text for retrieval")
    parser.add_argument("--encoding_model", default="cl100k_base", help="Tokenizer encoding used for chunking and token counting")
    parser.add_argument("--chunks_path", default="checkpoints/chunks.json", help="Path to save or load text chunks")
    parser.add_argument("--chunks_index_path", default="checkpoints/chunks.index", help="Path to FAISS index for chunk embeddings")
    parser.add_argument("--graph_path", default="checkpoints/graph.json", help="Path to save or load the constructed knowledge graph")
    parser.add_argument("--nodes_index_path", default="checkpoints/nodes.index", help="Path to FAISS index for node (entity) embeddings")
    parser.add_argument("--communities_path", default="checkpoints/communities.json", help="Path to save or load detected community summaries")
    parser.add_argument("--communities_index_path", default="checkpoints/communities.index", help="Path to FAISS index for community embeddings")
    return parser.parse_args()


def main():
    args = parse_arguments()
    embed_model = SentenceTransformer(args.embed_model, device="cuda:1")
    embed_model_params = sum(p.numel() for p in embed_model.parameters())
    print(f"\tEmbedding model: {args.embed_model} with {embed_model_params:,} parameters")

    vllm = vLLMInterface(
        base_url=args.base_url, model=args.model, 
        temperature=0, enable_thinking=False
    )

    chunker = Chunker(
        chunk_size=512, overlap=64, encoding_model=args.encoding_model, 
        embed_model=embed_model
    )
    chunks = chunker.chunk_files(
        files=glob(args.files), chunks_path=args.chunks_path, 
        index_path=args.chunks_index_path
    )

    graph_builder = GraphBuilder(LLM=vllm, embed_model=embed_model)
    graph = graph_builder.build_graph(chunks)

    graph_analyzer = GraphAnalyzer(LLM=vllm, embed_model=embed_model, max_community_size=50)
    graph_with_communities, community_summaries = graph_analyzer.analyze(graph)
    graph_builder.save(
        graph=graph_with_communities,
        graph_path=args.graph_path,
        index_path=args.nodes_index_path
    )
    graph_analyzer.save(
        community_summaries=community_summaries,
        index_path=args.communities_index_path,
        meta_path=args.communities_path
    )


if __name__ == "__main__":
    main()
