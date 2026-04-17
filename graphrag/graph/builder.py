"""Graph builder for building knowledge graph from documents using NetworkX.

工作流程:
1. extract_batch(): 并行提取所有 chunks 的实体和关系（不建图）
2. align_and_build(): 消歧对齐后构建 NetworkX 图

实体属性:
- chunk_ids: 集合，记录实体来自哪些 chunk
- type: 实体类型（最常用）
"""
from collections import Counter, defaultdict
import os
import pickle
import networkx as nx
from typing import Optional

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from .extractor import get_extractor
from .resolver import get_resolver


class GraphBuilder:
    """Build knowledge graph from documents using NetworkX.

    Args:
        extractor: 实体提取器。
        resolver: 实体消歧器（用于 align_entities）。
        max_workers: 并行处理的最大工作线程数。
        storage_path: 图持久化路径。
    """

    def __init__(
        self,
        extractor=None,
        resolver=None,
        max_workers: int = 16,
        storage_path: str = "graph.pkl",
    ):
        self.max_workers = max_workers
        self.storage_path = storage_path
        self.extractor = extractor or get_extractor()
        self.resolver = resolver or get_resolver()
        self.built_graph = None

    @property
    def graph(self) -> nx.DiGraph:
        """Lazy load NetworkX graph."""
        return self.built_graph if self.built_graph else self.load_graph()

    @staticmethod
    def load_graph(self, path=None) -> nx.DiGraph:
        """Load graph from pickle file."""
        path = path or self.storage_path
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return nx.DiGraph()

    def save_graph(self):
        """Save graph to pickle file."""
        dir_path = os.path.dirname(self.storage_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.storage_path, "wb") as f:
            pickle.dump(self.built_graph, f)

    def clear_graph(self):
        """Clear all data from graph."""
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def stats(self) -> dict[str, int]:
        """Get graph statistics.

        Returns:
            Dictionary with node and relationship counts.
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_relationships": self.graph.number_of_edges()
        }

    # ==================== 提取阶段 ====================

    def extract_batch(
        self,
        documents: list[Document],
        mode: str = "thread"
    ) -> list[GraphDocument]:
        """并行提取所有 chunks 的实体和关系（不建图，不去重）。

        Args:
            documents: 输入的 LangChain Document 列表（chunks）。
            mode: 提取模式。选项：
                  - "async": 异步并发执行
                  - "thread": 线程池并发执行（默认）
                  - "sync": 顺序执行

        Returns:
            提取的 GraphDocument 列表（未去重，未建图）。
        """
        # 直接使用 extractor 的 extract_batch 方法
        return self.extractor.extract_batch(documents, mode=mode)

    # ==================== 对齐和建图阶段 ====================

    def align_and_build(self, graph_docs: list[GraphDocument]) -> dict[str, int]:
        """对提取的 GraphDocuments 进行消歧对齐，然后构建 NetworkX 图。

        流程:
        1. 收集所有实体，记录 chunk_ids
        2. 用 embedding 找 alias，进行消歧
        3. 构建实体映射 {original_name: canonical_name}
        4. 用 canonical 名称构建 NetworkX 图，保留 chunk_ids 集合

        Args:
            graph_docs: extract_batch() 返回的 GraphDocument 列表。

        Returns:
            统计信息字典。
        """
        # Step 1: 收集所有实体和关系，记录 chunk_ids
        print("\n[Align Step 1] Collecting entities and relationships...")

        # entity_name -> {chunk_ids: set, types: Counter}
        entity_info: dict[str, dict] = defaultdict(lambda: {"source": set(), "chunk_ids": set(), "types": Counter()})
        # 收集所有关系 (source, target, rel_type, chunk_id)
        all_relationships = []

        for graph_doc in graph_docs:
            chunk_id = int(graph_doc.source.metadata.get("chunk_id", -1))
            entity_source = graph_doc.source.metadata.get("source", "")

            for node in graph_doc.nodes:
                entity_info[node.id]["source"].add(entity_source)
                entity_info[node.id]["chunk_ids"].add(chunk_id)
                entity_info[node.id]["types"][node.type] += 1

            for rel in graph_doc.relationships:
                source_id = rel.source.id
                target_id = rel.target.id
                all_relationships.append({
                    "source": source_id,
                    "target": target_id,
                    "rel_type": rel.type,
                    "chunk_id": chunk_id
                })

        entity_names = list(entity_info.keys())
        print(f"  Found {len(entity_names)} unique entities from {len(graph_docs)} chunks")

        if not entity_names:
            print("No entities found.")
            return {"entities": 0, "relationships": 0, "alias_groups": 0}

        # Step 2: 用 embedding 找 alias
        print("\n[Align Step 2] Finding aliases using embedding similarity...")
        alias_map = self.resolver.find_aliases(entity_names)

        # 构建 canonical 映射
        name_to_canonical = {name: alias_map.get(name, name) for name in entity_names}

        # 统计 alias 组
        alias_groups = defaultdict(list)
        for name, canonical in name_to_canonical.items():
            alias_groups[canonical].append(name)
        alias_groups = {k: v for k, v in alias_groups.items() if len(v) > 1}

        print(f"  Found {len(alias_groups)} alias groups")

        # Step 3: 构建 NetworkX 图（使用 canonical 名称）
        print("\n[Align Step 3] Building NetworkX graph...")
        new_graph = nx.DiGraph()

        # 添加节点（使用 canonical 名称，合并 chunk_ids）
        canonical_info: dict[str, dict] = defaultdict(lambda: {"chunk_ids": set(), "types": Counter()})
        for name, canonical in name_to_canonical.items():
            canonical_info[canonical]["chunk_ids"].update(entity_info[name]["chunk_ids"])
            canonical_info[canonical]["types"].update(entity_info[name]["types"])

        for canonical, info in canonical_info.items():
            most_common_type = info["types"].most_common(1)[0][0]
            new_graph.add_node(
                canonical,
                type=most_common_type,
                chunk_ids=sorted(info["chunk_ids"])  # 转换为列表以便 pickle
            )

        # 添加关系（使用 canonical 名称，去重）
        seen_edges = set()
        for rel in all_relationships:
            source_canonical = name_to_canonical.get(rel["source"], rel["source"])
            target_canonical = name_to_canonical.get(rel["target"], rel["target"])

            if source_canonical not in new_graph or target_canonical not in new_graph:
                continue

            edge_key = (source_canonical, target_canonical, rel["rel_type"])
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            new_graph.add_edge(
                source_canonical,
                target_canonical,
                rel_type=rel["rel_type"]
            )

        # 赋值并持久化到磁盘
        self.built_graph = new_graph
        self.save_graph()

        stats = self.stats()
        print(f"\n[Build Complete] Graph built: {stats['num_nodes']} nodes, {stats['num_relationships']} relationships")

        return self.built_graph

    # ==================== 便捷方法：一站式提取 + 对齐 + 建图 ====================

    def build_from_documents(self, documents: list[Document]) -> dict[str, int]:
        """一站式从 documents 构建知识图谱。

        流程：extract_batch() -> align_and_build()

        Args:
            documents: 输入的 LangChain Document 列表（chunks）。

        Returns:
            统计信息字典。
        """
        graph_docs = self.extract_batch(documents)
        return self.align_and_build(graph_docs)


def get_graph_builder(
    extractor=None,
    resolver=None,
    max_workers=16,
    storage_path="graph.pkl",
) -> GraphBuilder:
    """Get a GraphBuilder instance.

    Args:
        extractor: Optional entity extractor instance.
        resolver: Optional entity resolver instance.
        max_workers: Maximum number of concurrent workers. Default is 16.
        storage_path: Path to persist the graph. Default is "graph_data.pkl".

    Returns:
        GraphBuilder instance.
    """
    return GraphBuilder(
        extractor=extractor,
        resolver=resolver,
        max_workers=max_workers,
        storage_path=storage_path,
    )


if __name__ == "__main__":
    from langchain_core.documents import Document

    print("=" * 60)
    print("GraphBuilder 测试：build_from_documents + stats")
    print("=" * 60)

    # 创建测试文档
    test_docs = [
        Document(
            page_content="北京是中国的首都，位于华北平原。北京市是中华人民共和国的政治文化中心。",
            metadata={"chunk_id": 0, "source": "doc1"}
        ),
        Document(
            page_content="北京市 GDP 超过 4 万亿元，是中国四大直辖市之一。北京拥有众多高校和科研院所。",
            metadata={"chunk_id": 1, "source": "doc2"}
        ),
        Document(
            page_content="上海是中国的经济中心，位于长江入海口。上海市是国际金融中心。",
            metadata={"chunk_id": 2, "source": "doc3"}
        ),
        Document(
            page_content="上海浦东新区是中国改革开放的前沿。上海港是世界最大的集装箱港口。",
            metadata={"chunk_id": 3, "source": "doc4"}
        ),
    ]

    storage_path = "test_graph.pkl"
    try:
        # Step 1: 创建 Builder
        print("\n[Step 1] 创建 GraphBuilder...")
        builder = get_graph_builder(storage_path=storage_path)
        builder.clear_graph()
        print("  ✓ GraphBuilder 创建成功")

        # Step 2: 一站式建图
        print("\n[Step 2] 使用 build_from_documents 一站式建图...")
        result = builder.build_from_documents(test_docs)
        print(f"  ✓ 建图完成:")
        print(f"    - 实体数：{result['entities']}")
        print(f"    - 关系数：{result['relationships']}")
        print(f"    - Aliases：{result['alias_groups']}")


        # 清理测试文件
        builder.clear_graph()
        print(f"\n[清理] 已删除测试文件 {storage_path}")

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
