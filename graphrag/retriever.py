"""
GraphRAG Retriever - 向量检索 + 图谱检索 + 多跳遍历
"""
from typing import List, Optional, Dict, Any, Set, Tuple
from collections import deque
import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

class GraphRAGRetriever:
    """GraphRAG 检索器 - 支持向量检索、图谱检索和多跳遍历

    Args:
        graph: Graph 实例（包含 NetworkX 图）
        entity_index: FAISS 实体向量索引
        entity_metadata: 实体元数据列表
        embedding: 嵌入模型
        vectorstore: LangChain FAISS vectorstore（可选）
    """

    def __init__(
        self,
        graph,
        entity_index: Optional[faiss.Index] = None,
        entity_metadata: Optional[List[dict]] = None,
        embedding=None,
        vectorstore=None,
    ):
        self.graph = graph
        self.entity_index = entity_index
        self.entity_metadata = entity_metadata or []
        self.embedding = embedding
        self.vectorstore = vectorstore

    def search_vectors(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            (文档，相似度分数) 元组列表
        """
        if self.vectorstore is None or self.embedding is None:
            return []

        # 手动生成查询嵌入并搜索
        # EmbeddingWrapper 使用 encode 方法
        query_embedding = self.embedding.encode([query])[0]

        # 使用 FAISS 底层索引搜索
        index = self.vectorstore.index
        scores, indices = index.search(np.array([query_embedding], dtype=np.float32), k=top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.vectorstore.docstore._dict):
                doc_id = list(self.vectorstore.docstore._dict.keys())[idx]
                doc = self.vectorstore.docstore.search(doc_id)
                results.append((doc, float(score)))

        return results

    def search_entities(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, dict, float]]:
        """基于 embedding 搜索相似实体

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            (实体名称，元数据，相似度分数) 元组列表
        """
        if self.entity_index is None or self.entity_index.ntotal == 0:
            return []

        # 生成查询嵌入
        query_embedding = self.embedding.encode([query])
        query_array = np.array(query_embedding, dtype=np.float32)

        # FAISS 搜索
        scores, indices = self.entity_index.search(
            query_array, min(top_k, self.entity_index.ntotal)
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.entity_metadata):
                metadata = self.entity_metadata[idx]
                entity_name = metadata.get("name", "")
                results.append((entity_name, metadata, float(score)))

        return results

    def traverse_multi_hop(
        self,
        start_entities: List[str],
        max_hops: int = 2,
        max_neighbors: int = 5
    ) -> GraphDocument:
        """从起始实体进行多跳遍历

        Args:
            start_entities: 起始实体名称列表
            max_hops: 最大跳数
            max_neighbors: 每跳最多扩展的邻居数

        Returns:
            包含遍历结果的 GraphDocument
        """
        graph = self.graph
        visited_nodes: Set[str] = set()
        nodes: List[Node] = []
        node_map: Dict[str, Node] = {}
        relationships: List[Relationship] = []

        # 首先查找并添加起始节点
        for entity in start_entities:
            # 尝试精确匹配
            if entity in graph:
                node_data = graph.nodes[entity]
                node = Node(id=entity, type=node_data.get("type", "Entity"))
                nodes.append(node)
                node_map[entity] = node
                visited_nodes.add(entity)

        # 使用 BFS 遍历
        queue = deque([(entity_id, 0) for entity_id in visited_nodes])

        while queue:
            current_entity, current_hop = queue.popleft()

            if current_hop >= max_hops:
                continue

            # 获取当前实体的邻居（双向）
            neighbors = []
            # 出边
            for target in graph.successors(current_entity):
                if target not in visited_nodes:
                    edge_data = graph[current_entity][target]
                    neighbors.append((target, edge_data.get("rel_type", "RELATED_TO")))
            # 入边
            for source in graph.predecessors(current_entity):
                if source not in visited_nodes:
                    edge_data = graph[source][current_entity]
                    neighbors.append((source, edge_data.get("rel_type", "RELATED_TO")))

            # 限制邻居数量
            neighbors = neighbors[:max_neighbors]

            for neighbor_id, rel_type in neighbors:
                # 确保当前节点在 map 中
                if current_entity not in node_map:
                    node_data = graph.nodes[current_entity]
                    node = Node(id=current_entity, type=node_data.get("type", "Entity"))
                    nodes.append(node)
                    node_map[current_entity] = node

                # 添加邻居节点
                visited_nodes.add(neighbor_id)
                neighbor_data = graph.nodes[neighbor_id]
                neighbor_node = Node(id=neighbor_id, type=neighbor_data.get("type", "Entity"))
                nodes.append(neighbor_node)
                node_map[neighbor_id] = neighbor_node

                # 添加关系
                source_node = node_map[current_entity]
                target_node = node_map[neighbor_id]
                relationships.append(Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel_type
                ))

                # 添加到队列继续遍历
                queue.append((neighbor_id, current_hop + 1))

        # 创建源文档
        source_doc = Document(
            page_content="",
            metadata={"source": "multi_hop_traversal", "hops": max_hops}
        )

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=source_doc
        )

    def retrieve(
        self,
        query: str,
        top_k_vectors: int = 5,
        top_k_entities: int = 3,
        max_hops: int = 2,
        max_neighbors: int = 5,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5
    ) -> List[Document]:
        """混合检索：向量检索 + 图谱检索 + 多跳遍历

        Args:
            query: 查询文本
            top_k_vectors: 向量检索返回数量
            top_k_entities: 实体检索返回数量
            max_hops: 多跳遍历最大跳数
            max_neighbors: 多跳遍历每跳最大邻居数
            vector_weight: 向量检索权重
            graph_weight: 图谱检索权重

        Returns:
            检索结果文档列表
        """
        results = []

        # 1. 向量检索
        vector_results = self.search_vectors(query, top_k=top_k_vectors)
        for doc, score in vector_results:
            doc.metadata["score"] = score * vector_weight
            doc.metadata["retrieval_type"] = "vector"
            results.append(doc)

        # 2. 实体检索
        entity_results = self.search_entities(query, top_k=top_k_entities)
        if entity_results:
            # 3. 多跳遍历
            start_entities = [entity_name for entity_name, _, _ in entity_results]
            graph_doc = self.traverse_multi_hop(
                start_entities=start_entities,
                max_hops=max_hops,
                max_neighbors=max_neighbors
            )

            # 将遍历结果转换为 Document
            if graph_doc.nodes or graph_doc.relationships:
                # 构建上下文文本
                context_lines = []

                # 添加实体信息
                for node in graph_doc.nodes:
                    chunk_ids = self.graph.nodes[node.id].get("chunk_ids", [])
                    context_lines.append(f"Entity: {node.id} (Type: {node.type}, Chunks: {chunk_ids})")

                # 添加关系信息
                for rel in graph_doc.relationships:
                    context_lines.append(
                        f"  {rel.source.id} --{rel.type}--> {rel.target.id}"
                    )

                # 使用最高相似度的实体分数作为图谱分数
                graph_score = entity_results[0][2] if entity_results else 0.0

                context_doc = Document(
                    page_content="\n".join(context_lines),
                    metadata={
                        "source": "graph_traversal",
                        "score": graph_score * graph_weight,
                        "retrieval_type": "graph",
                        "entities": [n.id for n in graph_doc.nodes],
                        "relationships": len(graph_doc.relationships)
                    }
                )
                results.append(context_doc)

        # 按分数排序
        results.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)

        return results


def get_graphrag_retriever(
    graph,
    entity_index: Optional[faiss.Index] = None,
    entity_metadata: Optional[List[dict]] = None,
    embedding=None,
    vectorstore=None,
) -> GraphRAGRetriever:
    """获取 GraphRAG 检索器实例

    Args:
        graph: graph 实例
        entity_index: FAISS 实体向量索引
        entity_metadata: 实体元数据列表
        embedding: 嵌入模型
        vectorstore: LangChain FAISS vectorstore

    Returns:
        GraphRAGRetriever 实例
    """
    return GraphRAGRetriever(
        graph=graph,
        entity_index=entity_index,
        entity_metadata=entity_metadata,
        embedding=embedding,
        vectorstore=vectorstore,
    )


if __name__ == "__main__":
    import pickle
    print("=" * 60)
    print("GraphRAGRetriever 测试")
    print("=" * 60)

    from graphrag.indexer import get_graphrag_indexer

    # 加载已生成的索引
    print("\n[Test 1] Loading index from ./database...")
    storage_path = "./database"

    # 加载 graph
    graph_path = f"{storage_path}/graph.pkl"
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    print(f"  Loaded graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # 加载 entities
    entities_path = f"{storage_path}/entities.pkl"
    with open(entities_path, "rb") as f:
        entities_data = pickle.load(f)
    entity_index = entities_data["index"]
    entity_metadata = entities_data["metadata"]
    print(f"  Loaded {entity_index.ntotal} entities")

    # 加载 vectorstore
    from langchain_community.vectorstores import FAISS
    vectorstore_path = f"{storage_path}/vectorstore"
    from models.embedding import get_embedding
    embedding = get_embedding()
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embedding,
        allow_dangerous_deserialization=True
    )
    print(f"  Loaded vectorstore")


    # 创建检索器
    print("\n[Test 2] Create retriever...")
    retriever = get_graphrag_retriever(
        graph=graph,
        entity_index=entity_index,
        entity_metadata=entity_metadata,
        embedding=embedding,
        vectorstore=vectorstore
    )
    print("  ✓ Passed")

    # 测试实体检索
    print("\n[Test 3] Search entities...")
    entity_results = retriever.search_entities("中国的首都", top_k=3)
    print(f"  Found {len(entity_results)} entities")
    for name, metadata, score in entity_results:
        print(f"    - {name} ({metadata.get('type', metadata.get('type', 'Unknown'))}) (score: {score:.4f})")
    print("  ✓ Passed")

    # 测试多跳遍历
    print("\n[Test 4] Multi-hop traversal...")
    graph_doc = retriever.traverse_multi_hop(["北京市"], max_hops=2, max_neighbors=5)
    print(f"  Nodes: {len(graph_doc.nodes)}")
    print(f"  Relationships: {len(graph_doc.relationships)}")
    for node in graph_doc.nodes:
        print(f"    - {node.id} ({node.type})")
    for rel in graph_doc.relationships:
        print(f"    - {rel.source.id} --{rel.type}--> {rel.target.id}")
    print("  ✓ Passed")

    # 测试混合检索
    print("\n[Test 5] Hybrid retrieval...")
    hybrid_results = retriever.retrieve(
        "中国的首都是哪里？",
        top_k_vectors=3,
        top_k_entities=3,
        max_hops=2,
        graph_weight=0.7
    )
    print(f"  Found {len(hybrid_results)} results")
    for doc in hybrid_results:
        print(f"    [{doc.metadata.get('retrieval_type')}] score: {doc.metadata.get('score', 0):.4f}")
        content = doc.page_content[:100] if len(doc.page_content) > 100 else doc.page_content
        print(f"      {content}...")
    print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
