"""
Knowledge Graph Pipeline Demo - 知识图谱构建与分析全流程演示

This script demonstrates the complete workflow of building and analyzing
a knowledge graph from text documents.

本脚本演示从文本文档构建和分析知识图谱的完整流程：
1. 实体与关系提取
2. 知识图谱构建
3. 节点去重
4. 社区发现
5. 社区总结
6. 结果存储
"""
import os
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx

from models.graph_models import GraphDocumentWrapper, CommunitySummary
from graphrag.index.entity_extractor import EntityExtractor, ParallelEntityExtractor
from graphrag.index.graph_builder import GraphBuilder
from graphrag.index.node_deduplicator import NodeDeduplicator
from analyzers.community_detector import CommunityDetector
from analyzers.community_summarizer import CommunitySummarizer
from storage.graph_storage import GraphStorage
from storage.embedding_index import EmbeddingIndex


# ==================== 配置参数 ====================

# 并发配置
MAX_WORKERS = 16

# 节点去重配置
DEDUP_THRESHOLD = 0.9

# 社区发现配置
MAX_COMMUNITY_SIZE = 50


# ==================== 阶段 1: 实体与关系提取 ====================

def extract_entities_parallel(
    texts: List[str],
    sources: Optional[List[str]],
    llm,
    max_workers: int = MAX_WORKERS
) -> List[GraphDocumentWrapper]:
    """
    并行提取实体和关系。

    Args:
        texts: 文本列表
        sources: 可选的源标识列表
        llm: LangChain 语言模型实例
        max_workers: 最大并发数

    Returns:
        GraphDocumentWrapper 列表
    """
    extractor = ParallelEntityExtractor(llm, max_workers)
    return extractor.extract_parallel(texts, sources)


def extract_entities_single(
    text: str,
    source: Optional[str] = None,
    llm = None
) -> GraphDocumentWrapper:
    """
    从单个文本中提取实体和关系。

    Args:
        text: 输入文本
        source: 可选的源标识
        llm: LangChain 语言模型实例

    Returns:
        GraphDocumentWrapper 实例
    """
    extractor = EntityExtractor(llm)
    return extractor.extract(text, source)


# ==================== 阶段 2: 图构建 ====================

def build_graph_from_documents(
    documents: List[GraphDocumentWrapper]
) -> nx.MultiDiGraph:
    """
    从文档列表构建知识图谱。

    Args:
        documents: GraphDocumentWrapper 列表

    Returns:
        NetworkX MultiDiGraph
    """
    builder = GraphBuilder()
    builder.build_from_documents(documents)
    return builder.get_graph()


# ==================== 阶段 3: 节点去重 ====================

def find_node_aliases(
    node_names: List[str],
    embed_model,
    threshold: float = DEDUP_THRESHOLD
) -> Dict[str, str]:
    """
    查找相似节点并生成别名映射。

    Args:
        node_names: 节点名称列表
        embed_model: 嵌入模型
        threshold: 相似度阈值

    Returns:
        别名映射字典 {别名：规范名称}
    """
    deduplicator = NodeDeduplicator(embed_model, threshold)
    return deduplicator.find_aliases(node_names)


def apply_alias_map(
    graph: nx.MultiDiGraph,
    alias_map: Dict[str, str]
) -> nx.MultiDiGraph:
    """
    将别名映射应用到图上，合并相似节点。

    Args:
        graph: 原始图
        alias_map: 别名映射字典

    Returns:
        去重后的新图
    """
    # 创建新图
    new_graph = nx.MultiDiGraph()

    # 构建节点映射
    node_mapping = {}
    for node in graph.nodes():
        canonical = alias_map.get(node, node)
        node_mapping[node] = canonical

    # 合并节点
    for node in graph.nodes():
        canonical = node_mapping[node]
        if canonical not in new_graph:
            new_graph.add_node(canonical, **graph.nodes[node])
        else:
            # 累加权重
            new_graph.nodes[canonical]['weight'] += graph.nodes[node].get('weight', 1)

    # 合并边
    for source, target, key, data in graph.edges(keys=True, data=True):
        new_source = node_mapping[source]
        new_target = node_mapping[target]

        # 跳过自环
        if new_source == new_target:
            continue

        if not new_graph.has_edge(new_source, new_target):
            new_graph.add_edge(new_source, new_target, key=key, **data)
        else:
            # 检查是否已有相同关系类型的边
            edge_found = False
            for k in new_graph[new_source][new_target]:
                if k == key:
                    new_graph[new_source][new_target][k]['weight'] += data.get('weight', 1)
                    edge_found = True
                    break
            if not edge_found:
                new_graph.add_edge(new_source, new_target, key=key, **data)

    return new_graph


# ==================== 阶段 4: 社区发现 ====================

def detect_communities(
    graph: nx.MultiDiGraph,
    max_community_size: int = MAX_COMMUNITY_SIZE
) -> Tuple[nx.MultiDiGraph, Dict[int, List[str]]]:
    """
    检测图中的社区结构。

    Args:
        graph: 知识图谱
        max_community_size: 社区最大大小

    Returns:
        (标注社区后的图，社区字典 {community_id: [节点列表]})
    """
    detector = CommunityDetector(max_community_size)
    graph, communities, _ = detector.detect_communities(graph)
    return graph, communities


# ==================== 阶段 5: 社区总结 ====================

def summarize_communities(
    graph: nx.MultiDiGraph,
    communities: Dict[int, List[str]],
    llm,
    embed_model,
    max_workers: int = MAX_WORKERS
) -> List[CommunitySummary]:
    """
    为每个社区生成文本总结和嵌入向量。

    Args:
        graph: 知识图谱
        communities: 社区字典
        llm: LangChain 语言模型实例
        embed_model: 嵌入模型
        max_workers: 最大并发数

    Returns:
        CommunitySummary 列表
    """
    summarizer = CommunitySummarizer(llm, embed_model)
    return summarizer.summarize_communities(graph, communities, max_workers)


# ==================== 阶段 6: 存储 ====================

def save_graph(graph: nx.MultiDiGraph, path: str) -> str:
    """保存图为 JSON 格式"""
    return GraphStorage.save_graph(graph, path)


def save_community_metadata(
    community_summaries: List[CommunitySummary],
    path: str
) -> str:
    """保存社区元数据"""
    return GraphStorage.save_community_metadata(community_summaries, path)


def save_node_index(
    graph: nx.MultiDiGraph,
    embed_model,
    path: str
) -> str:
    """构建并保存节点嵌入索引"""
    node_index, node_ids = EmbeddingIndex.build_node_index(
        list(graph.nodes()), embed_model
    )
    return EmbeddingIndex.save_index(node_index, path, node_ids, "nodes")


def save_community_index(
    community_summaries: List[CommunitySummary],
    path: str
) -> str:
    """构建并保存社区嵌入索引"""
    community_index, community_ids = EmbeddingIndex.build_community_index(
        community_summaries
    )
    return EmbeddingIndex.save_index(community_index, path, community_ids, "communities")


def load_graph(path: str) -> nx.MultiDiGraph:
    """加载保存的图"""
    return GraphStorage.load_graph(path)


def load_community_metadata(path: str) -> List[Dict[str, Any]]:
    """加载社区元数据"""
    return GraphStorage.load_community_metadata(path)


# ==================== 完整流程 ====================

def build_knowledge_graph(
    text_chunks: List[str],
    sources: Optional[List[str]],
    llm,
    embed_model,
    deduplicate: bool = True,
    max_workers: int = MAX_WORKERS
) -> nx.MultiDiGraph:
    """
    构建知识图谱的完整流程。

    Args:
        text_chunks: 文本块列表
        sources: 可选的源标识列表
        llm: LangChain 语言模型实例
        embed_model: 嵌入模型
        deduplicate: 是否进行节点去重
        max_workers: 最大并发数

    Returns:
        构建完成的知识图谱
    """
    print("=" * 60)
    print("阶段 1: 实体与关系提取")
    print("=" * 60)
    documents = extract_entities_parallel(text_chunks, sources, llm, max_workers)

    # 统计提取结果
    total_nodes = sum(len(doc.nodes) for doc in documents)
    total_edges = sum(len(doc.edges) for doc in documents)
    print(f"\t从 {len(documents)} 个文档中提取了 {total_nodes} 个节点和 {total_edges} 条边")

    print("\n" + "=" * 60)
    print("阶段 2: 知识图谱构建")
    print("=" * 60)
    graph = build_graph_from_documents(documents)
    print(f"\t图构建完成：{graph.number_of_nodes()} 个节点，{graph.number_of_edges()} 条边")

    if deduplicate:
        print("\n" + "=" * 60)
        print("阶段 3: 节点去重")
        print("=" * 60)
        node_names = list(graph.nodes())
        alias_map = find_node_aliases(node_names, embed_model)
        graph = apply_alias_map(graph, alias_map)
        print(f"\t去重完成：{graph.number_of_nodes()} 个节点，{graph.number_of_edges()} 条边")
        print(f"\t合并了 {len(node_names) - graph.number_of_nodes()} 个重复节点")

    return graph


def analyze_knowledge_graph(
    graph: nx.MultiDiGraph,
    llm,
    embed_model,
    max_workers: int = MAX_WORKERS
) -> Tuple[Dict[int, List[str]], List[CommunitySummary]]:
    """
    分析知识图谱：社区发现和总结。

    Args:
        graph: 知识图谱
        llm: LangChain 语言模型实例
        embed_model: 嵌入模型
        max_workers: 最大并发数

    Returns:
        (社区字典，社区总结列表)
    """
    print("\n" + "=" * 60)
    print("阶段 4: 社区发现")
    print("=" * 60)
    graph, communities = detect_communities(graph)
    print(f"\t发现 {len(communities)} 个社区")
    for comm_id, nodes in sorted(communities.items()):
        print(f"\t社区 {comm_id}: {len(nodes)} 个节点")

    print("\n" + "=" * 60)
    print("阶段 5: 社区总结")
    print("=" * 60)
    community_summaries = summarize_communities(graph, communities, llm, embed_model, max_workers)
    print(f"\t完成 {len(community_summaries)} 个社区的总结")

    return communities, community_summaries


def save_results(
    graph: nx.MultiDiGraph,
    community_summaries: List[CommunitySummary],
    embed_model,
    base_path: str
) -> Dict[str, str]:
    """
    保存所有结果。

    Args:
        graph: 知识图谱
        community_summaries: 社区总结列表
        embed_model: 嵌入模型
        base_path: 基础保存路径

    Returns:
        保存文件路径字典
    """
    print("\n" + "=" * 60)
    print("阶段 6: 结果存储")
    print("=" * 60)

    paths = {}

    # 保存图
    graph_path = os.path.join(base_path, "graph.json")
    paths["graph"] = save_graph(graph, graph_path)

    # 保存社区元数据
    if community_summaries:
        meta_path = os.path.join(base_path, "communities.json")
        paths["community_metadata"] = save_community_metadata(community_summaries, meta_path)

    # 保存节点索引
    node_index_path = os.path.join(base_path, "node_index.faiss")
    paths["node_index"] = save_node_index(graph, embed_model, node_index_path)

    # 保存社区索引
    if community_summaries:
        community_index_path = os.path.join(base_path, "community_index.faiss")
        paths["community_index"] = save_community_index(community_summaries, community_index_path)

    print(f"\n所有结果已保存到：{base_path}")
    return paths


def get_graph_stats(
    graph: nx.MultiDiGraph,
    communities: Optional[Dict[int, List[str]]] = None
) -> Dict[str, Any]:
    """
    获取图的统计信息。

    Args:
        graph: 知识图谱
        communities: 可选的社区字典

    Returns:
        统计信息字典
    """
    stats = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges()
    }

    if communities:
        stats["num_communities"] = len(communities)
        stats["community_sizes"] = {cid: len(nodes) for cid, nodes in communities.items()}

    return stats


# ==================== 演示入口 ====================

def main(
    text_chunks: List[str],
    llm,
    embed_model,
    output_dir: str = "./output",
    sources: Optional[List[str]] = None,
    skip_build: bool = False,
    skip_analysis: bool = False
):
    """
    知识图谱构建与分析主函数。

    Args:
        text_chunks: 文本块列表
        llm: LangChain 语言模型实例
        embed_model: 嵌入模型
        output_dir: 输出目录
        sources: 可选的源标识列表
        skip_build: 跳过构建阶段（使用已保存的图）
        skip_analysis: 跳过分析阶段
    """
    print("\n" + "#" * 60)
    print("# 知识图谱构建与分析全流程演示")
    print("#" * 60 + "\n")

    graph = None
    communities = None
    community_summaries = None

    # 检查是否有已保存的结果
    if skip_build and os.path.exists(os.path.join(output_dir, "graph.json")):
        print("加载已保存的图...")
        graph = load_graph(os.path.join(output_dir, "graph.json"))
        if os.path.exists(os.path.join(output_dir, "communities.json")):
            community_summaries = load_community_metadata(
                os.path.join(output_dir, "communities.json")
            )

    # 阶段 1-3: 构建知识图谱
    if not skip_build:
        graph = build_knowledge_graph(
            text_chunks, sources, llm, embed_model,
            deduplicate=True, max_workers=MAX_WORKERS
        )

    # 阶段 4-5: 分析知识图谱
    if not skip_analysis and graph is not None:
        communities, community_summaries = analyze_knowledge_graph(
            graph, llm, embed_model, max_workers=MAX_WORKERS
        )

    # 阶段 6: 保存结果
    if graph is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_results(graph, community_summaries or [], embed_model, output_dir)

    # 打印统计信息
    if graph is not None:
        print("\n" + "=" * 60)
        print("统计信息")
        print("=" * 60)
        stats = get_graph_stats(graph, communities)
        for key, value in stats.items():
            if key != "community_sizes":
                print(f"\t{key}: {value}")
            else:
                print(f"\t{key}:")
                for cid, size in value.items():
                    print(f"\t\t社区 {cid}: {size} 个节点")

    return graph, communities, community_summaries


if __name__ == "__main__":
    # 演示用法示例
    print("""
使用方法:

from main import main

# 准备文本
text_chunks = [
    "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    "Einstein won the Nobel Prize in Physics in 1921.",
    # ... 更多文本
]

# 初始化 LLM 和嵌入模型
from langchain_community.llms import ollama
from sentence_transformers import SentenceTransformer

llm = ollama.llms.Ollama(model="qwen2.5")
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 运行完整流程
graph, communities, summaries = main(
    text_chunks=text_chunks,
    llm=llm,
    embed_model=embed_model,
    output_dir="./output"
)
""")
