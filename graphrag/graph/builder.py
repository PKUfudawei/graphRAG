"""Graph builder for building knowledge graph from documents using NetworkX."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
import pickle
import networkx as nx
from typing import Optional

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from tqdm import tqdm

from .extractor import get_extractor
from .resolver import get_resolver


class GraphBuilder:
    """Build knowledge graph from documents using NetworkX.

    工作流程:
    1. extract_and_save_batch(): 并行提取实体并直接添加到 NetworkX 图（不去重）
    2. align_entities(): 用 embedding 找 alias，按度数对齐合并

    Args:
        entity_extractor: 实体提取器。
        entity_resolver: 实体消歧器（用于 align_entities）。
        max_workers: 并行处理的最大工作线程数。
        storage_path: 图持久化路径。
    """

    def __init__(
        self,
        entity_extractor=None,
        entity_resolver=None,
        max_workers: int = 16,
        storage_path: str = "graph_data.pkl",
    ):
        self.max_workers = max_workers
        self.entity_extractor = entity_extractor or get_extractor()
        self.entity_resolver = entity_resolver or get_resolver()
        self.storage_path = Path(storage_path)
        self._graph: Optional[nx.DiGraph] = None

    @property
    def graph(self) -> nx.DiGraph:
        """Lazy load NetworkX graph."""
        if self._graph is None:
            self._graph = self._load_graph()
        return self._graph

    def _load_graph(self) -> nx.DiGraph:
        """Load graph from pickle file."""
        if self.storage_path.exists():
            with open(self.storage_path, "rb") as f:
                return pickle.load(f)
        return nx.DiGraph()

    def _save_graph(self):
        """Save graph to pickle file."""
        with open(self.storage_path, "wb") as f:
            pickle.dump(self._graph, f)

    def clear_graph(self):
        """Clear all data from graph."""
        self._graph = nx.DiGraph()
        if self.storage_path.exists():
            self.storage_path.unlink()

    def build(self, document: Document, chunk_id: int) -> GraphDocument:
        """Build a GraphDocument from a single document.

        Args:
            document: Input LangChain Document.
            chunk_id: The index of the document in the batch.

        Returns:
            GraphDocument with deduplicated nodes and relationships.
        """
        # Step 1: Extract entities and relations
        graph_doc = self.entity_extractor.extract(
            text=document.page_content,
            source=str(chunk_id)
        )

        # Step 2: Deduplicate entities
        graph_doc = self._deduplicate_graph(graph_doc, chunk_id)

        return graph_doc

    def build_batch(self, documents: list[Document]) -> list[GraphDocument]:
        """Build GraphDocuments from multiple documents concurrently.

        Args:
            documents: List of input LangChain Documents.

        Returns:
            List of GraphDocuments with deduplicated nodes and relationships.
        """
        if not documents:
            return []

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.build, doc, doc.metadata.get("chunk_id", idx))
                for idx, doc in enumerate(documents)
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Building graphs"
            ):
                results.append(future.result())

        # Sort results by chunk_id to maintain order
        results.sort(key=lambda g: int(g.source.metadata.get("source", 0)))
        return results

    def _deduplicate_graph(self, graph_doc: GraphDocument, chunk_id: int) -> GraphDocument:
        """Deduplicate nodes in a GraphDocument using entity resolver.

        Args:
            graph_doc: Input GraphDocument.
            chunk_id: The chunk ID for the source document.

        Returns:
            GraphDocument with deduplicated nodes.
        """
        if not graph_doc.nodes:
            return graph_doc

        # Get all unique node names
        node_names = [node.id for node in graph_doc.nodes]

        # Find aliases using embedding similarity
        alias_map = self.entity_resolver.find_aliases(node_names)

        # Build mapping from old name to canonical name
        name_to_canonical = {name: alias_map.get(name, name) for name in node_names}

        # Count types for each canonical name to find most common type
        canonical_type_counter: dict[str, Counter] = {}
        for node in graph_doc.nodes:
            canonical_name = name_to_canonical[node.id]
            if canonical_name not in canonical_type_counter:
                canonical_type_counter[canonical_name] = Counter()
            canonical_type_counter[canonical_name][node.type] += 1

        # Build new node map with canonical names
        canonical_nodes: dict[str, Node] = {}
        for node in graph_doc.nodes:
            canonical_name = name_to_canonical[node.id]
            if canonical_name not in canonical_nodes:
                # Get most common type for this canonical name
                most_common_type = canonical_type_counter[canonical_name].most_common(1)[0][0]
                canonical_nodes[canonical_name] = Node(
                    id=canonical_name,
                    type=most_common_type
                )

        # Rebuild relationships with canonical node references
        new_relationships = []
        for rel in graph_doc.relationships:
            source_id = getattr(rel.source, 'id', str(rel.source))
            target_id = getattr(rel.target, 'id', str(rel.target))
            source_canonical = name_to_canonical.get(source_id, source_id)
            target_canonical = name_to_canonical.get(target_id, target_id)

            # Skip if source or target doesn't exist after dedup
            if source_canonical in canonical_nodes and target_canonical in canonical_nodes:
                new_relationships.append(Relationship(
                    source=canonical_nodes[source_canonical],
                    target=canonical_nodes[target_canonical],
                    type=rel.type
                ))

        # Create source document with chunk_id
        source_doc = Document(
            page_content=graph_doc.source.page_content if graph_doc.source else "",
            metadata={"source": chunk_id}
        )

        return GraphDocument(
            nodes=list(canonical_nodes.values()),
            relationships=new_relationships,
            source=source_doc
        )

    # ==================== 增量维护方案 ====================

    def extract_and_save_batch(self, documents: list[Document]) -> list[GraphDocument]:
        """提取实体并添加到 NetworkX 图（不去重）。

        这是增量维护方案的第一步：并行提取所有文档的实体和关系，
        直接添加到 NetworkX 图，不做去重处理。去重在后续的 align_entities() 中完成。

        Args:
            documents: 输入的 LangChain Document 列表。

        Returns:
            提取的 GraphDocument 列表（未去重）。
        """
        if not documents:
            return []

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 并行提取实体
            futures = [
                executor.submit(
                    self.entity_extractor.extract,
                    doc.page_content,
                    str(doc.metadata.get("chunk_id", idx))
                )
                for idx, doc in enumerate(documents)
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting entities"
            ):
                results.append(future.result())

        # 按 chunk_id 排序
        results.sort(key=lambda g: int(g.source.metadata.get("source", 0)) if g.source else 0)

        # 批量添加到 NetworkX 图
        with tqdm(total=len(results), desc="Adding to NetworkX") as pbar:
            for graph_doc in results:
                # 添加节点
                for node in graph_doc.nodes:
                    self.graph.add_node(
                        node.id,
                        node_type=node.type,
                        **node.properties or {}
                    )

                # 添加关系
                for rel in graph_doc.relationships:
                    source_id = rel.source.id
                    target_id = rel.target.id

                    # 只添加已存在的节点之间的关系
                    if source_id in self.graph and target_id in self.graph:
                        self.graph.add_edge(
                            source_id,
                            target_id,
                            rel_type=rel.type,
                            **rel.properties or {}
                        )

                pbar.update()

        # 持久化到磁盘
        self._save_graph()

        return results

    def align_entities(self, batch_size: int = 100) -> dict[str, int]:
        """从 NetworkX 图读取所有实体，用 embedding 找 alias，然后对齐合并。

        这是增量维护方案的第二步：
        1. 从 NetworkX 读取所有实体名称及其度数
        2. 用 resolver.find_aliases() 找 alias 组
        3. 对每个 alias 组，选择度数最高的作为 canonical
        4. 转移所有关系到 canonical，清理重复关系，删除旧实体

        Args:
            batch_size: 每批处理的 alias 组数量。

        Returns:
            统计信息字典，包含处理的组数、合并的实体数、转移的关系数等。
        """
        # Step 1: 从 NetworkX 读取所有实体及其度数
        print("\n[Align Step 1] Loading entities from NetworkX...")
        entities_with_degree = [
            {"entity_id": node, "degree": degree}
            for node, degree in self.graph.degree()
        ]

        if not entities_with_degree:
            print("No entities found in NetworkX.")
            return {"groups_processed": 0, "entities_merged": 0, "relationships_transferred": 0}

        entity_names = [e["entity_id"] for e in entities_with_degree]
        # 构建 entity_id -> degree 映射
        degree_map = {e["entity_id"]: e["degree"] for e in entities_with_degree}

        print(f"Found {len(entity_names)} entities in NetworkX.")

        # Step 2: 用 embedding 找 alias
        print("\n[Align Step 2] Finding aliases using embedding similarity...")
        alias_map = self.entity_resolver.find_aliases(entity_names)

        # Step 3: 构建 alias 组 {canonical: [aliases]}
        alias_groups: dict[str, list[str]] = {}
        for entity, canonical in alias_map.items():
            if canonical not in alias_groups:
                alias_groups[canonical] = []
            if entity not in alias_groups[canonical]:
                alias_groups[canonical].append(entity)

        # 过滤掉只有单个实体的组
        alias_groups = {k: v for k, v in alias_groups.items() if len(v) > 1}

        print(f"Found {len(alias_groups)} alias groups to align.")

        if not alias_groups:
            return {"groups_processed": 0, "entities_merged": 0, "relationships_transferred": 0}

        # Step 4: 对每个 alias 组执行对齐
        print("\n[Align Step 3] Aligning entities...")
        total_merged = 0
        total_rels_transferred = 0
        groups_processed = 0

        # 分批处理
        group_list = list(alias_groups.items())
        for i in range(0, len(group_list), batch_size):
            batch = group_list[i:i + batch_size]

            for canonical, aliases in batch:
                # 选择度数最高的作为保留实体
                all_entities = [canonical] + [a for a in aliases if a != canonical]
                target = max(all_entities, key=lambda x: (degree_map.get(x, 0), x))
                to_delete = [e for e in all_entities if e != target]

                if not to_delete:
                    continue

                # 执行合并：转移关系 + 删除旧实体
                merged, rels_transferred = self._merge_entity_group(target, to_delete)
                total_merged += merged
                total_rels_transferred += rels_transferred
                groups_processed += 1

                if groups_processed % 10 == 0:
                    print(f"  Processed {groups_processed} groups, merged {total_merged} entities...")

        # 持久化到磁盘
        self._save_graph()

        print(f"\n[Align Complete] Processed {groups_processed} groups, "
              f"merged {total_merged} entities, "
              f"transferred {total_rels_transferred} relationships.")

        return {
            "groups_processed": groups_processed,
            "entities_merged": total_merged,
            "relationships_transferred": total_rels_transferred
        }

    def _merge_entity_group(self, target: str, to_delete: list[str]) -> tuple[int, int]:
        """合并一个 alias 组：将 to_delete 中的所有实体合并到 target。

        Args:
            target: 保留的实体 ID（度数最高）。
            to_delete: 要删除的实体 ID 列表。

        Returns:
            (删除的实体数，转移的关系数)。
        """
        rels_transferred = 0
        deleted_count = 0

        for old_id in to_delete:
            if old_id not in self.graph:
                continue

            # 转移出边
            out_neighbors = list(self.graph.successors(old_id))
            for other_id in out_neighbors:
                if other_id == target:
                    continue

                # 获取原边的属性
                old_edge_data = self.graph[old_id][other_id]
                rel_type = old_edge_data.get("rel_type", "RELATED_TO")

                # 检查是否已存在相同的关系
                if target in self.graph and other_id in self.graph[target]:
                    existing_edge_data = self.graph[target][other_id]
                    if existing_edge_data.get("rel_type") == rel_type:
                        continue  # 已存在，跳过

                # 添加新边
                self.graph.add_edge(target, other_id, rel_type=rel_type)
                rels_transferred += 1

            # 转移入边
            in_neighbors = list(self.graph.predecessors(old_id))
            for other_id in in_neighbors:
                if other_id == target:
                    continue

                # 获取原边的属性
                old_edge_data = self.graph[other_id][old_id]
                rel_type = old_edge_data.get("rel_type", "RELATED_TO")

                # 检查是否已存在相同的关系
                if other_id in self.graph and target in self.graph[other_id]:
                    existing_edge_data = self.graph[other_id][target]
                    if existing_edge_data.get("rel_type") == rel_type:
                        continue  # 已存在，跳过

                # 添加新边
                self.graph.add_edge(other_id, target, rel_type=rel_type)
                rels_transferred += 1

            # 删除旧实体
            self.graph.remove_node(old_id)
            deleted_count += 1

        return deleted_count, rels_transferred

    def stats(self) -> dict[str, int]:
        """Get graph statistics.

        Returns:
            Dictionary with node and relationship counts.
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_relationships": self.graph.number_of_edges()
        }

    def find_shortest_path(self, source: str, target: str) -> Optional[list[str]]:
        """Find shortest path between two entities.

        Args:
            source: Source entity ID.
            target: Target entity ID.

        Returns:
            List of entity IDs representing the path, or None if no path exists.
        """
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None

    def get_entity_neighbors(self, entity_id: str, max_neighbors: int = 10) -> list[tuple[str, str]]:
        """Get neighbors of an entity.

        Args:
            entity_id: Entity ID to get neighbors for.
            max_neighbors: Maximum number of neighbors to return.

        Returns:
            List of (neighbor_id, rel_type) tuples.
        """
        if entity_id not in self.graph:
            return []

        neighbors = []
        # 出边邻居
        for target in list(self.graph.successors(entity_id))[:max_neighbors]:
            rel_type = self.graph[entity_id][target].get("rel_type", "RELATED_TO")
            neighbors.append((target, rel_type))
        # 入边邻居
        for source in list(self.graph.predecessors(entity_id))[:max_neighbors]:
            rel_type = self.graph[source][entity_id].get("rel_type", "RELATED_TO")
            neighbors.append((source, rel_type))

        return neighbors


def get_graph_builder(
    entity_extractor=None,
    entity_resolver=None,
    max_workers=16,
    storage_path="graph_data.pkl",
) -> GraphBuilder:
    """Get a GraphBuilder instance.

    Args:
        entity_extractor: Optional entity extractor instance.
        entity_resolver: Optional entity resolver instance.
        max_workers: Maximum number of concurrent workers. Default is 16.
        storage_path: Path to persist the graph. Default is "graph_data.pkl".

    Returns:
        GraphBuilder instance.
    """
    return GraphBuilder(
        entity_extractor=entity_extractor,
        entity_resolver=entity_resolver,
        max_workers=max_workers,
        storage_path=storage_path,
    )


if __name__ == "__main__":
    from langchain_core.documents import Document

    print("=" * 60)
    print("GraphBuilder NetworkX 增量维护方案测试")
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

    try:
        # Step 1: 创建 Builder
        print("\n[Step 1] 创建 GraphBuilder...")
        builder = get_graph_builder(storage_path="test_graph.pkl")
        builder.clear_graph()  # 清空之前的测试数据
        print("  ✓ GraphBuilder 创建成功")

        # Step 2: 提取 GraphDocuments 并添加到 NetworkX（不去重）
        print("\n[Step 2] 从 documents 提取并添加到 NetworkX...")
        graph_docs = builder.extract_and_save_batch(test_docs)
        print(f"  ✓ 提取完成：{len(graph_docs)} 个 GraphDocuments")
        for i, gd in enumerate(graph_docs):
            print(f"    - Chunk {i}: {len(gd.nodes)} 节点，{len(gd.relationships)} 关系")

        # Step 3: 查看添加到 NetworkX 后的统计
        print("\n[Step 3] 查看 NetworkX 统计（对齐前）...")
        stats_before = builder.stats()
        print(f"  ✓ 节点数：{stats_before['num_nodes']}")
        print(f"  ✓ 关系数：{stats_before['num_relationships']}")

        # Step 4: 实体对齐
        print("\n[Step 4] 执行实体对齐...")
        align_result = builder.align_entities()
        print(f"  ✓ 对齐完成:")
        print(f"    - 处理的组数：{align_result['groups_processed']}")
        print(f"    - 合并的实体：{align_result['entities_merged']}")
        print(f"    - 转移的关系：{align_result['relationships_transferred']}")

        # Step 5: 查看对齐后的统计
        print("\n[Step 5] 查看 NetworkX 统计（对齐后）...")
        stats_after = builder.stats()
        print(f"  ✓ 节点数：{stats_after['num_nodes']} (减少 {stats_before['num_nodes'] - stats_after['num_nodes']})")
        print(f"  ✓ 关系数：{stats_after['num_relationships']}")

        # Step 6: 列出所有实体
        print("\n[Step 6] 列出图中的所有实体...")
        all_entities = list(builder.graph.nodes())
        print(f"  ✓ 共 {len(all_entities)} 个实体:")
        for entity in all_entities[:10]:
            degree = builder.graph.degree(entity)
            print(f"    - {entity} (度数：{degree})")
        if len(all_entities) > 10:
            print(f"    ... 还有 {len(all_entities) - 10} 个实体")

        # Step 7: 测试路径查找（使用实际存在的实体）
        print("\n[Step 7] 测试路径查找功能...")
        if len(all_entities) >= 2:
            # 找一个有关系的实体对
            for edge in list(builder.graph.edges())[:1]:
                source, target = edge
                path = builder.find_shortest_path(source, target)
                if path:
                    print(f"  ✓ {source} -> {target} 的路径：{' -> '.join(path)}")
                break
        else:
            print("  - 实体数量不足，跳过路径查找测试")

        # Step 8: 测试邻居查询
        print("\n[Step 8] 测试邻居查询功能...")
        if all_entities:
            # 找度数最高的实体
            entity_with_most_neighbors = max(all_entities, key=lambda x: builder.graph.degree(x))
            neighbors = builder.get_entity_neighbors(entity_with_most_neighbors)
            if neighbors:
                print(f"  ✓ {entity_with_most_neighbors} 的邻居 ({len(neighbors)}个):")
                for neighbor, rel_type in neighbors[:5]:
                    print(f"    - {neighbor} ({rel_type})")

        # 清理测试文件
        if Path("test_graph.pkl").exists():
            Path("test_graph.pkl").unlink()
            print("\n[清理] 已删除测试文件 test_graph.pkl")

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
