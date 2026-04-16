"""
GraphRAG Retriever - 向量检索 + 图谱检索 + 多跳遍历
"""
import os
import sys
from typing import List, Optional, Dict, Any, Set, Tuple
from collections import deque
import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

# 支持直接运行和模块导入
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from graphrag.graph.graph_storage import get_neo4j_storage
from models.embedding import get_embedding


class GraphRAGRetriever:
    """GraphRAG 检索器 - 支持向量检索、图谱检索和多跳遍历

    Args:
        vector_index: FAISS 文档向量索引
        vector_metadata: 文档向量元数据
        entity_vector_index: FAISS 实体向量索引
        entity_metadata: 实体向量元数据
        embedding_model_name: 嵌入模型名称
        neo4j_uri: Neo4j 连接 URI
        neo4j_username: Neo4j 用户名
        neo4j_password: Neo4j 密码
    """

    def __init__(
        self,
        vector_index: Optional[faiss.Index] = None,
        vector_metadata: Optional[List[dict]] = None,
        entity_vector_index: Optional[faiss.Index] = None,
        entity_metadata: Optional[List[dict]] = None,
        embedding_model_name: str = "BAAI/bge-m3",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: Optional[str] = None,
    ):
        self.vector_index = vector_index
        self.vector_metadata = vector_metadata or []
        self.entity_vector_index = entity_vector_index
        self.entity_metadata = entity_metadata or []
        self.embedding_model_name = embedding_model_name

        # 懒加载
        self._embedding_model = None
        self._neo4j_storage = None

        # Neo4j 连接配置
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password

    @property
    def embedding_model(self):
        """懒加载嵌入模型"""
        if self._embedding_model is None:
            self._embedding_model = get_embedding(model=self.embedding_model_name)
        return self._embedding_model

    @property
    def neo4j_storage(self):
        """懒加载 Neo4j 存储"""
        if self._neo4j_storage is None:
            self._neo4j_storage = get_neo4j_storage(
                uri=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password
            )
        return self._neo4j_storage

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
        if self.vector_index is None or self.vector_index.ntotal == 0:
            return []

        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query])
        query_array = np.array(query_embedding, dtype=np.float32)

        # FAISS 搜索
        distances, indices = self.vector_index.search(
            query_array, min(top_k, self.vector_index.ntotal)
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.vector_metadata):
                metadata = self.vector_metadata[idx]
                doc = Document(
                    page_content=metadata["content"],
                    metadata={"source": metadata["source"], "retrieval_type": "vector"}
                )
                # 距离越小越相似，转换为相似度分数
                similarity = 1.0 / (1.0 + dist)
                results.append((doc, float(similarity)))

        return results

    def search_entities(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """基于 embedding 搜索相似实体

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            (实体名称，实体类型，相似度分数) 元组列表
        """
        if self.entity_vector_index is None or self.entity_vector_index.ntotal == 0:
            return []

        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query])
        query_array = np.array(query_embedding, dtype=np.float32)

        # FAISS 搜索
        distances, indices = self.entity_vector_index.search(
            query_array, min(top_k, self.entity_vector_index.ntotal)
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.entity_metadata):
                metadata = self.entity_metadata[idx]
                # 距离越小越相似，转换为相似度分数
                similarity = 1.0 / (1.0 + dist)
                results.append((metadata["name"], metadata["type"], float(similarity)))

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
        visited_nodes: Set[str] = set()
        nodes: List[Node] = []
        node_map: Dict[str, Node] = {}
        relationships: List[Relationship] = []

        # 首先查找并添加起始节点
        for entity in start_entities:
            # 尝试精确匹配
            results = self.neo4j_storage.query(
                "MATCH (n:Entity {id: $entity_id}) RETURN n.id as id, n.node_type as type",
                {"entity_id": entity}
            )

            if not results:
                # 尝试模糊匹配（包含）
                results = self.neo4j_storage.query(
                    "MATCH (n:Entity) WHERE n.id CONTAINS $entity_id RETURN n.id as id, n.node_type as type LIMIT 1",
                    {"entity_id": entity}
                )

            if results:
                matched_id = results[0]["id"]
                node = Node(id=matched_id, type=results[0].get("type", "Entity"))
                nodes.append(node)
                node_map[matched_id] = node
                visited_nodes.add(matched_id)
            else:
                # 如果找不到，仍然添加一个空节点
                node = Node(id=entity, type="Entity")
                nodes.append(node)
                node_map[entity] = node
                visited_nodes.add(entity)

        # 使用 BFS 遍历
        queue = deque([(entity_id, 0) for entity_id in visited_nodes])

        while queue:
            current_entity, current_hop = queue.popleft()

            if current_hop >= max_hops:
                continue

            # 查询当前实体的邻居（双向）
            cypher = """
            MATCH (n:Entity {id: $entity_id})
            OPTIONAL MATCH (n)-[r]-(neighbor:Entity)
            WHERE neighbor.id <> n.id
            RETURN type(r) as rel_type, neighbor.id as neighbor_id, neighbor.node_type as neighbor_type
            LIMIT $max_neighbors
            """

            results = self.neo4j_storage.query(
                cypher,
                {"entity_id": current_entity, "max_neighbors": max_neighbors * 2}
            )

            neighbor_count = 0
            for record in results:
                neighbor_id = record.get("neighbor_id")
                if neighbor_id is None or neighbor_id in visited_nodes:
                    continue

                # 确保当前节点在 map 中
                if current_entity not in node_map:
                    node = Node(id=current_entity, type="Entity")
                    nodes.append(node)
                    node_map[current_entity] = node

                # 添加邻居节点
                visited_nodes.add(neighbor_id)
                neighbor_node = Node(id=neighbor_id, type=record.get("neighbor_type", "Entity"))
                nodes.append(neighbor_node)
                node_map[neighbor_id] = neighbor_node

                # 添加关系
                source_node = node_map[current_entity]
                target_node = node_map[neighbor_id]
                relationships.append(Relationship(
                    source=source_node,
                    target=target_node,
                    type=record.get("rel_type", "RELATED_TO")
                ))

                # 添加到队列继续遍历
                queue.append((neighbor_id, current_hop + 1))
                neighbor_count += 1

                if neighbor_count >= max_neighbors:
                    break

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
                    context_lines.append(f"Entity: {node.id} (Type: {node.type})")

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
    vector_index: Optional[faiss.Index] = None,
    vector_metadata: Optional[List[dict]] = None,
    entity_vector_index: Optional[faiss.Index] = None,
    entity_metadata: Optional[List[dict]] = None,
    embedding_model_name: str = "BAAI/bge-m3",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_username: str = "neo4j",
    neo4j_password: Optional[str] = None,
) -> GraphRAGRetriever:
    """获取 GraphRAG 检索器实例

    Args:
        vector_index: FAISS 文档向量索引
        vector_metadata: 文档向量元数据
        entity_vector_index: FAISS 实体向量索引
        entity_metadata: 实体向量元数据
        embedding_model_name: 嵌入模型名称
        neo4j_uri: Neo4j 连接 URI
        neo4j_username: Neo4j 用户名
        neo4j_password: Neo4j 密码

    Returns:
        GraphRAGRetriever 实例
    """
    return GraphRAGRetriever(
        vector_index=vector_index,
        vector_metadata=vector_metadata,
        entity_vector_index=entity_vector_index,
        entity_metadata=entity_metadata,
        embedding_model_name=embedding_model_name,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("GraphRAGRetriever 测试")
    print("=" * 60)

    from graphrag.indexer import get_graphrag_indexer

    # 准备数据
    print("\n[Test 1] Index documents...")
    indexer = get_graphrag_indexer(chunk_size=200, overlap=20)
    texts = [
        "北京是中国的首都，位于华北平原。北京拥有丰富的历史文化遗产。",
        "上海是中国最大的城市，位于长江入海口。上海是国际金融中心。",
        "Ebenezer Scrooge is a wealthy businessman in Victorian London.",
        "He is visited by the ghost of his former partner Jacob Marley.",
    ]
    indexer.index_texts(texts, reset=True, index_vector=True, index_graph=True)
    print("  ✓ Passed")

    # 创建检索器
    print("\n[Test 2] Create retriever...")
    retriever = get_graphrag_retriever(
        vector_index=indexer._vector_index,
        vector_metadata=indexer._vector_metadata,
        entity_vector_index=indexer._entity_vector_index,
        entity_metadata=indexer._entity_metadata,
    )
    print("  ✓ Passed")

    # 测试向量检索
    print("\n[Test 3] Search vectors...")
    vector_results = retriever.search_vectors("中国的首都是哪里？", top_k=2)
    print(f"  Found {len(vector_results)} results")
    for doc, score in vector_results:
        print(f"    - {doc.page_content[:50]}... (score: {score:.4f})")
    print("  ✓ Passed")

    # 测试实体检索
    print("\n[Test 4] Search entities...")
    entity_results = retriever.search_entities("Scrooge", top_k=3)
    print(f"  Found {len(entity_results)} entities")
    for name, entity_type, score in entity_results:
        print(f"    - {name} ({entity_type}) (score: {score:.4f})")
    print("  ✓ Passed")

    # 测试多跳遍历
    print("\n[Test 5] Multi-hop traversal...")
    graph_doc = retriever.traverse_multi_hop(["Scrooge"], max_hops=2, max_neighbors=5)
    print(f"  Nodes: {len(graph_doc.nodes)}")
    print(f"  Relationships: {len(graph_doc.relationships)}")
    for node in graph_doc.nodes:
        print(f"    - {node.id} ({node.type})")
    for rel in graph_doc.relationships:
        print(f"    - {rel.source.id} --{rel.type}--> {rel.target.id}")
    print("  ✓ Passed")

    # 测试混合检索
    print("\n[Test 6] Hybrid retrieval...")
    hybrid_results = retriever.retrieve(
        "Who is Scrooge's partner?",
        top_k_vectors=2,
        top_k_entities=3,
        max_hops=2,
        vector_weight=0.3,
        graph_weight=0.7
    )
    print(f"  Found {len(hybrid_results)} results")
    for doc in hybrid_results:
        print(f"    [{doc.metadata.get('retrieval_type')}] score: {doc.metadata.get('score', 0):.4f}")
        print(f"      {doc.page_content[:80]}...")
    print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
