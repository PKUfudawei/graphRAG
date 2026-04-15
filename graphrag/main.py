"""
GraphRAG 系统主模块 - 知识图谱 RAG
支持构建图索引和基于社区的检索
"""

import os
import sys
import argparse
from typing import List, Optional
from glob import glob
from pathlib import Path
from tqdm import tqdm

import networkx as nx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.index import get_entity_extractor, GraphBuilder, NodeDeduplicator
from graphrag.analyze import CommunityDetector, CommunitySummarizer
from graphrag.retrieve import get_graph_retriever
from graphrag.storage import GraphStorage, EmbeddingIndexManager
from graphrag.models import GraphDocumentWrapper, CommunitySummary

# 复用 rag 模块
from rag.index import get_indexer as get_text_indexer
from rag.index import get_embedder


# ==================== 配置参数 ====================
MAX_WORKERS = 16
DEDUP_THRESHOLD = 0.9
MAX_COMMUNITY_SIZE = 50


# ==================== 图构建流程 ====================

def build_graph_from_texts(
    texts: List[str],
    sources: Optional[List[str]] = None,
    llm=None,
    embed_model=None,
    deduplicate: bool = True,
) -> nx.MultiDiGraph:
    """
    从文本构建知识图谱

    Args:
        texts: 文本列表
        sources: 源标识列表
        llm: LangChain LLM 实例
        embed_model: 嵌入模型
        deduplicate: 是否去重

    Returns:
        NetworkX 图
    """
    if llm is None:
        from rag.llm import get_llm
        llm = get_llm()
    if embed_model is None:
        embed_model = get_embedder()

    # 阶段 1: 实体提取
    print("=" * 60)
    print("阶段 1: 实体与关系提取")
    print("=" * 60)

    extractor = get_entity_extractor(llm)
    documents = []

    for text, source in tqdm(zip(texts, sources or [None] * len(texts)), desc="Extracting"):
        doc = extractor.extract(text, source)
        documents.append(doc)

    total_nodes = sum(len(doc.nodes) for doc in documents)
    total_edges = sum(len(doc.edges) for doc in documents)
    print(f"\t从 {len(documents)} 个文档中提取了 {total_nodes} 个节点和 {total_edges} 条边")

    # 阶段 2: 图构建
    print("\n" + "=" * 60)
    print("阶段 2: 知识图谱构建")
    print("=" * 60)

    builder = GraphBuilder()
    builder.build_from_documents(documents)
    graph = builder.get_graph()
    print(f"\t图构建完成：{graph.number_of_nodes()} 个节点，{graph.number_of_edges()} 条边")

    # 阶段 3: 节点去重
    if deduplicate:
        print("\n" + "=" * 60)
        print("阶段 3: 节点去重")
        print("=" * 60)

        deduplicator = NodeDeduplicator(embed_model, DEDUP_THRESHOLD)
        node_names = list(graph.nodes())
        alias_map = deduplicator.find_aliases(node_names)

        # 应用别名映射
        new_graph = nx.MultiDiGraph()
        node_mapping = {node: alias_map.get(node, node) for node in graph.nodes()}

        for node in graph.nodes():
            canonical = node_mapping[node]
            if canonical not in new_graph:
                new_graph.add_node(canonical, **graph.nodes[node])
            else:
                new_graph.nodes[canonical]["weight"] += graph.nodes[node].get("weight", 1)

        for source, target, key, data in graph.edges(keys=True, data=True):
            new_source = node_mapping[source]
            new_target = node_mapping[target]
            if new_source == new_target:
                continue
            if not new_graph.has_edge(new_source, new_target):
                new_graph.add_edge(new_source, new_target, key=key, **data)
            else:
                for k in new_graph[new_source][new_target]:
                    if k == key:
                        new_graph[new_source][new_target][k]["weight"] += data.get("weight", 1)
                        break
                else:
                    new_graph.add_edge(new_source, new_target, key=key, **data)

        graph = new_graph
        print(f"\t去重完成：{graph.number_of_nodes()} 个节点，{graph.number_of_edges()} 条边")

    return graph


def analyze_graph(
    graph: nx.MultiDiGraph,
    llm=None,
    embed_model=None,
) -> tuple:
    """
    分析知识图谱：社区发现和总结

    Returns:
        (communities, community_summaries)
    """
    if llm is None:
        from rag.llm import get_llm
        llm = get_llm()
    if embed_model is None:
        embed_model = get_embedder()

    # 阶段 4: 社区发现
    print("\n" + "=" * 60)
    print("阶段 4: 社区发现")
    print("=" * 60)

    detector = CommunityDetector(MAX_COMMUNITY_SIZE)
    graph, communities, _ = detector.detect_communities(graph)
    print(f"\t发现 {len(communities)} 个社区")

    # 阶段 5: 社区总结
    print("\n" + "=" * 60)
    print("阶段 5: 社区总结")
    print("=" * 60)

    summarizer = CommunitySummarizer(llm, embed_model)
    community_summaries = summarizer.summarize_communities(graph, communities, MAX_WORKERS)
    print(f"\t完成 {len(community_summaries)} 个社区的总结")

    return communities, community_summaries


def save_graph_results(
    graph: nx.MultiDiGraph,
    community_summaries: List[CommunitySummary],
    embed_model,
    output_path: str,
) -> dict:
    """保存图和相关索引"""
    os.makedirs(output_path, exist_ok=True)

    paths = {}

    # 保存图
    graph_path = os.path.join(output_path, "graph.json")
    GraphStorage.save_graph(graph, graph_path)
    paths["graph"] = graph_path

    # 保存社区元数据
    if community_summaries:
        meta_path = os.path.join(output_path, "communities.json")
        GraphStorage.save_community_metadata(community_summaries, meta_path)
        paths["community_metadata"] = meta_path

        # 保存社区索引
        community_index_path = os.path.join(output_path, "community_index.faiss")
        community_index, community_ids = EmbeddingIndexManager.build_community_index(
            community_summaries
        )
        EmbeddingIndexManager.save_index(community_index, community_index_path, community_ids, "communities")
        paths["community_index"] = community_index_path

    # 保存节点索引
    node_index_path = os.path.join(output_path, "node_index.faiss")
    node_index, node_ids = EmbeddingIndexManager.build_node_index(
        list(graph.nodes()), embed_model
    )
    EmbeddingIndexManager.save_index(node_index, node_index_path, node_ids, "nodes")
    paths["node_index"] = node_index_path

    print(f"\n所有结果已保存到：{output_path}")
    return paths


# ==================== RAG 查询 ====================

def generate_answer(query: str, context_docs: List[Document], llm=None) -> str:
    """根据上下文生成答案"""
    if llm is None:
        from rag.llm import get_llm
        llm = get_llm()

    context_text = "\n\n".join(
        f"[Community {i+1}]: {doc.page_content}" for i, doc in enumerate(context_docs)
    )
    prompt = f"""根据以下知识图谱社区信息回答问题。如果信息不足，请说明不知道。

社区信息:
{context_text}

问题：{query}

回答："""
    return llm.invoke([HumanMessage(content=prompt)]).content


def graphrag_query(query: str, retriever, llm=None, embed_model=None, top_k: int = 3) -> dict:
    """GraphRAG 查询"""
    if embed_model is None:
        embed_model = get_embedder()

    docs = retriever.retrieve(query, embed_model, top_k=top_k)

    if not docs:
        return {
            "query": query,
            "answer": "未找到相关的知识图谱信息。",
            "references": [],
        }

    answer = generate_answer(query, docs, llm)

    return {
        "query": query,
        "answer": answer,
        "references": [
            {
                "community_id": d.metadata.get("community_id", "unknown"),
                "score": d.metadata.get("score", 0),
                "nodes": d.metadata.get("nodes", [])[:5],  # 只显示前 5 个节点
            }
            for d in docs
        ],
    }


# ==================== CLI 接口 ====================

def build_index(
    files: List[str],
    output_path: str,
    embed_device: str = "cuda:0",
    chunk_size: int = 512,
    overlap: int = 50,
) -> None:
    """构建图索引（支持 Markdown 和纯文本）"""
    embed_model = get_embedder(device=embed_device)

    # 加载并分块文本
    text_indexer = get_text_indexer(chunk_size=chunk_size, overlap=overlap, embed_device=embed_device)
    all_docs, _ = text_indexer.index_files(files, build_vectorstore=False)

    texts = [doc.page_content for doc in all_docs]
    sources = [doc.metadata.get("source", "unknown") for doc in all_docs]

    print(f"\n加载了 {len(texts)} 个文本块")

    # 构建图
    graph = build_graph_from_texts(texts, sources, embed_model=embed_model)

    # 分析图
    communities, community_summaries = analyze_graph(graph, embed_model=embed_model)

    # 保存结果
    save_graph_results(graph, community_summaries, embed_model, output_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="GraphRAG 系统 - 知识图谱 RAG")
    parser.add_argument("--build", type=str, help="构建图索引：指定文件路径或 glob 模式")
    parser.add_argument("--query", type=str, help="单次查询")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--index", type=str, default="./data/graph_index", help="图索引路径")
    parser.add_argument("--top-k", type=int, default=3, help="返回的社区数量")
    parser.add_argument("--embed-device", type=str, default="cuda:0", help="嵌入模型设备")
    parser.add_argument("--chunk-size", type=int, default=512, help="文本块大小")
    parser.add_argument("--overlap", type=int, default=50, help="文本块重叠")

    return parser.parse_args(), parser


def main():
    args, parser = parse_arguments()

    if not any([args.build, args.query, args.interactive]):
        parser.print_help()
        return

    # 构建模式
    if args.build:
        files = glob(args.build)
        if not files:
            print(f"Error: No files matched '{args.build}'")
            return
        build_index(
            files,
            args.index,
            args.embed_device,
            args.chunk_size,
            args.overlap,
        )
        return

    # 查询/交互模式
    index_path = args.index
    if not os.path.exists(os.path.join(index_path, "community_index.faiss")):
        print(f"Error: Graph index not found at {index_path}")
        return

    # 加载检索器
    retriever = get_graph_retriever(
        index_path=os.path.join(index_path, "community_index.faiss"),
        metadata_path=os.path.join(index_path, "communities.json"),
        graph_path=os.path.join(index_path, "graph.json"),
    )
    embed_model = get_embedder(device=args.embed_device)

    # 单次查询
    if args.query:
        result = graphrag_query(args.query, retriever, embed_model=embed_model, top_k=args.top_k)
        print(f"\nAnswer: {result['answer']}")
        if result["references"]:
            print("\nReferences:")
            for ref in result["references"]:
                print(f"  - Community {ref['community_id']} (score: {ref['score']:.4f})")
        return

    # 交互模式
    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        while True:
            try:
                q = input("\nQuestion: ").strip()
                if q.lower() in ["quit", "exit", "q"]:
                    break
                if not q:
                    continue
                result = graphrag_query(q, retriever, embed_model=embed_model, top_k=args.top_k)
                print(f"\nAnswer: {result['answer']}")
                if result["references"]:
                    print("\nReferences:")
                    for ref in result["references"]:
                        print(f"  - Community {ref['community_id']} (score: {ref['score']:.4f})")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
