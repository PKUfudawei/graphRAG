"""
GraphRAG Retriever - 向量检索 + 图谱检索 + 多跳遍历
"""
import os
import sys
from typing import List, Optional, Set, Tuple
from collections import deque
import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import Node, Relationship

# 添加父目录到路径
_sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)

from rag.retriever import Retriever


class GraphRAGRetriever(Retriever):
    """GraphRAG 检索器 - 继承自 Retriever，支持 naive/local/global 三种检索模式"""

    def __init__(
        self,
        graph,
        entity_index: Optional[faiss.Index] = None,
        entity_metadata: Optional[List[dict]] = None,
        relationship_index: Optional[faiss.Index] = None,
        relationship_metadata: Optional[List[dict]] = None,
        embedding=None,
        vectorstore=None,
        top_k: int = 10,
    ):
        # 调用父类初始化
        super().__init__(vectorstore=vectorstore, top_k=top_k)

        self.graph = graph
        self.entity_index = entity_index
        self.entity_metadata = entity_metadata or []
        self.relationship_index = relationship_index
        self.relationship_metadata = relationship_metadata or []
        self.embedding = embedding

    def _search_entities(
        self, query: str, top_k: int
    ) -> Tuple[List[str], float]:
        """实体检索（私有方法）

        Returns:
            (实体名称列表，最高相似度分数)
        """
        if self.entity_index is None or self.entity_index.ntotal == 0:
            return [], 0.0

        query_embedding = self.embedding.encode([query])
        query_array = np.array(query_embedding, dtype=np.float32)
        scores, indices = self.entity_index.search(
            query_array, min(top_k, self.entity_index.ntotal)
        )

        start_entities = []
        graph_score = 0.0
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.entity_metadata):
                metadata = self.entity_metadata[idx]
                entity_name = metadata.get("name", "")
                start_entities.append(entity_name)
                if graph_score == 0.0:
                    graph_score = float(score)

        return start_entities, graph_score

    def _search_relationships(
        self, query: str, top_k: int
    ) -> Tuple[List[Tuple[str, str, str]], float]:
        """关系检索（私有方法）- 通过向量相似度找到最相关的关系

        Args:
            query: 查询文本
            top_k: 返回 top_k 个最相关的关系

        Returns:
            ((源实体，目标实体，关系类型) 列表，最高相似度分数)
        """
        if self.relationship_index is None or self.relationship_index.ntotal == 0:
            return [], 0.0

        # 使用关系描述作为查询（类似 LightRAG 的做法）
        query_embedding = self.embedding.encode([query])
        query_array = np.array(query_embedding, dtype=np.float32)
        scores, indices = self.relationship_index.search(
            query_array, min(top_k, self.relationship_index.ntotal)
        )

        relationships = []
        max_score = 0.0
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.relationship_metadata):
                metadata = self.relationship_metadata[idx]
                relationships.append((
                    metadata["src_id"],
                    metadata["tgt_id"],
                    metadata.get("rel_type", "RELATED_TO")
                ))
                if max_score == 0.0:
                    max_score = float(score)

        return relationships, max_score

    def _get_entities_from_relationships(
        self, relationships: List[Tuple[str, str, str]]
    ) -> Tuple[List[Node], List[Relationship]]:
        """从关系中抽取实体和关系（不往外跳）

        Args:
            relationships: (源实体，目标实体，关系类型) 列表

        Returns:
            (节点列表，关系列表)
        """
        graph = self.graph
        nodes: List[Node] = []
        node_map: dict = {}
        rels: List[Relationship] = []
        seen_entities: Set[str] = set()

        for src_id, tgt_id, rel_type in relationships:
            if src_id not in seen_entities and src_id in graph:
                node_data = graph.nodes[src_id]
                node = Node(id=src_id, type=node_data.get("type", "Entity"))
                nodes.append(node)
                node_map[src_id] = node
                seen_entities.add(src_id)

            if tgt_id not in seen_entities and tgt_id in graph:
                node_data = graph.nodes[tgt_id]
                node = Node(id=tgt_id, type=node_data.get("type", "Entity"))
                nodes.append(node)
                node_map[tgt_id] = node
                seen_entities.add(tgt_id)

            # 添加关系
            if src_id in node_map and tgt_id in node_map:
                rels.append(Relationship(
                    source=node_map[src_id],
                    target=node_map[tgt_id],
                    type=rel_type
                ))

        return nodes, rels

    def _bfs_traverse(
        self, start_entities: List[str], max_hops: int, max_neighbors: int
    ) -> Tuple[List[Node], List[Relationship]]:
        """BFS 多跳遍历（私有方法）

        Returns:
            (节点列表，关系列表)
        """
        graph = self.graph
        visited_nodes: Set[str] = set()
        nodes: List[Node] = []
        node_map: dict = {}
        relationships: List[Relationship] = []

        # 添加起始节点
        for entity in start_entities:
            if entity in graph:
                node_data = graph.nodes[entity]
                node = Node(id=entity, type=node_data.get("type", "Entity"))
                nodes.append(node)
                node_map[entity] = node
                visited_nodes.add(entity)

        # BFS 遍历
        queue = deque([(eid, 0) for eid in visited_nodes])
        while queue:
            current_entity, current_hop = queue.popleft()
            if current_hop >= max_hops:
                continue

            # 获取双向邻居
            neighbors = []
            for target in graph.successors(current_entity):
                if target not in visited_nodes:
                    edge_data = graph[current_entity][target]
                    neighbors.append((target, edge_data.get("rel_type", "RELATED_TO")))
            for source in graph.predecessors(current_entity):
                if source not in visited_nodes:
                    edge_data = graph[source][current_entity]
                    neighbors.append((source, edge_data.get("rel_type", "RELATED_TO")))

            neighbors = neighbors[:max_neighbors]
            for neighbor_id, rel_type in neighbors:
                if current_entity not in node_map:
                    node_data = graph.nodes[current_entity]
                    node = Node(id=current_entity, type=node_data.get("type", "Entity"))
                    nodes.append(node)
                    node_map[current_entity] = node

                visited_nodes.add(neighbor_id)
                neighbor_data = graph.nodes[neighbor_id]
                neighbor_node = Node(id=neighbor_id, type=neighbor_data.get("type", "Entity"))
                nodes.append(neighbor_node)
                node_map[neighbor_id] = neighbor_node

                relationships.append(Relationship(
                    source=node_map[current_entity],
                    target=neighbor_node,
                    type=rel_type
                ))
                queue.append((neighbor_id, current_hop + 1))

        return nodes, relationships

    def _get_chunks_by_entity_ids(
        self, entity_ids: List[str], max_chunks: int = 10
    ) -> List[Document]:
        """从实体节点获取相关的文本 chunk（类似 LightRAG 的做法）

        Args:
            entity_ids: 实体 ID 列表
            max_chunks: 最大返回 chunk 数量

        Returns:
            Document 列表
        """
        if self.vectorstore is None:
            return []

        # 收集所有 chunk_ids，并统计出现频率
        chunk_id_count: dict[str, int] = {}
        chunk_id_to_entities: dict[str, list[str]] = {}

        for entity_id in entity_ids:
            if entity_id in self.graph:
                node_data = self.graph.nodes[entity_id]
                chunk_ids = node_data.get("chunk_ids", [])
                for chunk_id in chunk_ids:
                    chunk_id_count[chunk_id] = chunk_id_count.get(chunk_id, 0) + 1
                    if chunk_id not in chunk_id_to_entities:
                        chunk_id_to_entities[chunk_id] = []
                    chunk_id_to_entities[chunk_id].append(entity_id)

        # 按出现频率排序（频率高的优先，类似 LightRAG 的 WEIGHT 方法）
        sorted_chunk_ids = sorted(
            chunk_id_count.keys(),
            key=lambda x: chunk_id_count[x],
            reverse=True
        )

        # 构建 chunk_id 到 docstore key 的映射
        chunk_id_to_key: dict[str, str] = {}
        for key, doc in self.vectorstore.docstore._dict.items():
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                chunk_id_to_key[chunk_id] = key

        # 获取 chunk
        chunks: List[Document] = []
        for chunk_id in sorted_chunk_ids[:max_chunks]:
            if chunk_id in chunk_id_to_key:
                doc = self.vectorstore.docstore.search(chunk_id_to_key[chunk_id])
                doc.metadata["related_entities"] = chunk_id_to_entities[chunk_id]
                doc.metadata["entity_count"] = chunk_id_count[chunk_id]
                chunks.append(doc)

        return chunks

    def _get_chunks_by_relationships(
        self, relationships: List[Tuple[str, str, str]], max_chunks: int = 10
    ) -> List[Document]:
        """从关系边获取相关的文本 chunk（类似 LightRAG 的 global search）

        Args:
            relationships: (源实体，目标实体，关系类型) 列表
            max_chunks: 最大返回 chunk 数量

        Returns:
            Document 列表
        """
        if self.vectorstore is None:
            return []

        # 收集所有 chunk_ids，并统计出现频率
        chunk_id_count: dict[str, int] = {}
        chunk_id_to_relationships: dict[str, list[tuple]] = {}

        for src_id, tgt_id, rel_type in relationships:
            edge_data = self.graph.get_edge_data(src_id, tgt_id)
            if edge_data:
                # 使用 chunk_ids 字段存储 chunk_ids
                chunk_ids = edge_data.get("chunk_ids", [])
                if isinstance(chunk_ids, str):
                    # 如果是字符串，可能需要分割
                    chunk_ids = chunk_ids.split("\n") if "\n" in chunk_ids else [chunk_ids]

                for chunk_id in chunk_ids:
                    if chunk_id:  # 过滤空字符串
                        chunk_id_count[chunk_id] = chunk_id_count.get(chunk_id, 0) + 1
                        if chunk_id not in chunk_id_to_relationships:
                            chunk_id_to_relationships[chunk_id] = []
                        chunk_id_to_relationships[chunk_id].append((src_id, tgt_id, rel_type))

        # 按出现频率排序
        sorted_chunk_ids = sorted(
            chunk_id_count.keys(),
            key=lambda x: chunk_id_count[x],
            reverse=True
        )

        # 构建 chunk_id 到 docstore key 的映射
        chunk_id_to_key: dict[str, str] = {}
        for key, doc in self.vectorstore.docstore._dict.items():
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                chunk_id_to_key[chunk_id] = key

        # 获取 chunk
        chunks: List[Document] = []
        for chunk_id in sorted_chunk_ids[:max_chunks]:
            if chunk_id in chunk_id_to_key:
                doc = self.vectorstore.docstore.search(chunk_id_to_key[chunk_id])
                doc.metadata["related_relationships"] = chunk_id_to_relationships[chunk_id]
                doc.metadata["relationship_count"] = chunk_id_count[chunk_id]
                chunks.append(doc)

        return chunks

    def naive_search(self, query: str, top_k: int = 5) -> List[Document]:
        """朴素检索：纯向量检索，直接调用父类的 vector_search"""
        if self.vectorstore is None:
            return []
        # 临时修改 top_k
        original_top_k = self.top_k
        self.top_k = top_k
        docs = self.vector_search(query)
        self.top_k = original_top_k
        # 更新 metadata
        for i, doc in enumerate(docs):
            doc.metadata["retrieval_type"] = "naive"
            doc.metadata["rank"] = i
        return docs

    def local_search(
        self,
        query: str,
        top_k_entities: int = 3,
        max_hops: int = 1,
        max_neighbors: int = 3,
        max_chunks: int = 10
    ) -> List[Document]:
        """局部检索：实体检索 + 多跳遍历 + 相关 chunk，适合精确问答

        参考 LightRAG 的 local search 实现：
        - 通过向量相似度检索最相关的实体（节点）
        - 从实体出发进行多跳遍历，获取相关的实体和关系
        - 从实体节点的 chunk_ids 获取相关的文本 chunk
        """
        results = []

        # 实体检索
        start_entities, graph_score = self._search_entities(query, top_k_entities)
        if not start_entities:
            return results

        # 多跳遍历
        nodes, relationships = self._bfs_traverse(start_entities, max_hops, max_neighbors)

        # 收集所有涉及的实体 ID（包括起始实体和遍历得到的实体）
        all_entity_ids = [n.id for n in nodes]

        # 从实体获取相关 chunk（类似 LightRAG）
        chunks = self._get_chunks_by_entity_ids(all_entity_ids, max_chunks=max_chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata["retrieval_type"] = "local_chunk"
            chunk.metadata["rank"] = i
            results.append(chunk)

        # 构建图结构上下文
        context_lines = [f"Entity: {n.id} (Type: {n.type})" for n in nodes]
        context_lines.extend(f"  {rel.source.id} --{rel.type}--> {rel.target.id}" for rel in relationships)

        context_doc = Document(
            page_content="\n".join(context_lines),
            metadata={
                "source": "local_graph",
                "score": graph_score,
                "retrieval_type": "local_graph",
                "entities": all_entity_ids,
                "relationships": len(relationships)
            }
        )
        results.append(context_doc)
        return results

    def local_search_with_stats(
        self,
        query: str,
        top_k_entities: int = 3,
        max_hops: int = 1,
        max_neighbors: int = 3,
        max_chunks: int = 10
    ) -> dict:
        """局部检索（带统计信息）"""
        # 实体检索
        start_entities, graph_score = self._search_entities(query, top_k_entities)
        if not start_entities:
            return {'results': [], 'stats': {'matched_entities': 0, 'traversed_nodes': 0, 'traversed_relationships': 0, 'chunks_from_entities': 0}}

        # 多跳遍历
        nodes, relationships = self._bfs_traverse(start_entities, max_hops, max_neighbors)

        # 收集所有涉及的实体 ID
        all_entity_ids = [n.id for n in nodes]

        # 从实体获取相关 chunk
        chunks = self._get_chunks_by_entity_ids(all_entity_ids, max_chunks=max_chunks)
        results = []
        for i, chunk in enumerate(chunks):
            chunk.metadata["retrieval_type"] = "local_chunk"
            chunk.metadata["rank"] = i
            results.append(chunk)

        # 构建图结构上下文
        context_lines = [f"Entity: {n.id} (Type: {n.type})" for n in nodes]
        context_lines.extend(f"  {rel.source.id} --{rel.type}--> {rel.target.id}" for rel in relationships)

        context_doc = Document(
            page_content="\n".join(context_lines),
            metadata={
                "source": "local_graph",
                "score": graph_score,
                "retrieval_type": "local_graph",
                "entities": all_entity_ids,
                "relationships": len(relationships)
            }
        )
        results.append(context_doc)

        return {
            'results': results,
            'stats': {
                'matched_entities': len(start_entities),
                'traversed_nodes': len(nodes),
                'traversed_relationships': len(relationships),
                'chunks_from_entities': len(chunks)
            }
        }

    def global_search(
        self,
        query: str,
        top_k_relationships: int = 10,
        max_chunks: int = 10
    ) -> List[Document]:
        """全局检索：关系检索 + 相关 chunk，适合综合问题

        参考 LightRAG 的 global search 实现：
        - 通过向量相似度检索最相关的关系（边）
        - 从关系的 source_id 获取相关的文本 chunk
        - 只获取这些关系直接连接的实体，不进行多跳扩展
        """
        results = []

        # 1. 关系检索
        related_relationships, graph_score = self._search_relationships(query, top_k_relationships)

        if not related_relationships:
            return results

        # 2. 从关系边获取相关 chunk（类似 LightRAG）
        chunks = self._get_chunks_by_relationships(related_relationships, max_chunks=max_chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata["retrieval_type"] = "global_chunk"
            chunk.metadata["rank"] = i
            results.append(chunk)

        # 3. 从关系中抽取涉及的实体（不往外跳）
        nodes, relationships = self._get_entities_from_relationships(related_relationships)

        if not nodes:
            return results

        # 构建图结构上下文
        context_lines = [f"Entity: {n.id} (Type: {n.type})" for n in nodes]
        context_lines.extend(f"  {rel.source.id} --{rel.type}--> {rel.target.id}" for rel in relationships)

        context_doc = Document(
            page_content="\n".join(context_lines),
            metadata={
                "source": "global_graph",
                "score": graph_score,
                "retrieval_type": "global_graph",
                "entities": [n.id for n in nodes],
                "relationships": len(relationships)
            }
        )
        results.append(context_doc)

        return results

    def global_search_with_stats(
        self,
        query: str,
        top_k_relationships: int = 10,
        max_chunks: int = 10
    ) -> dict:
        """全局检索（带统计信息）"""
        results = []

        # 1. 关系检索
        related_relationships, graph_score = self._search_relationships(query, top_k_relationships)

        if not related_relationships:
            return {
                'results': results,
                'stats': {
                    'matched_relationships': 0,
                    'chunks_from_relations': 0,
                    'entities': 0
                }
            }

        # 2. 从关系边获取相关 chunk
        chunks = self._get_chunks_by_relationships(related_relationships, max_chunks=max_chunks)
        chunk_count = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata["retrieval_type"] = "global_chunk"
            chunk.metadata["rank"] = i
            results.append(chunk)

        # 3. 从关系中抽取涉及的实体
        nodes, relationships = self._get_entities_from_relationships(related_relationships)

        if not nodes:
            return {
                'results': results,
                'stats': {
                    'matched_relationships': len(related_relationships),
                    'chunks_from_relations': chunk_count,
                    'entities': 0
                }
            }

        # 构建图结构上下文
        context_lines = [f"Entity: {n.id} (Type: {n.type})" for n in nodes]
        context_lines.extend(f"  {rel.source.id} --{rel.type}--> {rel.target.id}" for rel in relationships)

        context_doc = Document(
            page_content="\n".join(context_lines),
            metadata={
                "source": "global_graph",
                "score": graph_score,
                "retrieval_type": "global_graph",
                "entities": [n.id for n in nodes],
                "relationships": len(relationships)
            }
        )
        results.append(context_doc)

        return {
            'results': results,
            'stats': {
                'matched_relationships': len(related_relationships),
                'chunks_from_relations': chunk_count,
                'entities': len(nodes)
            }
        }


def get_graphrag_retriever(
    graph,
    entity_index: Optional[faiss.Index] = None,
    entity_metadata: Optional[List[dict]] = None,
    relationship_index: Optional[faiss.Index] = None,
    relationship_metadata: Optional[List[dict]] = None,
    embedding=None,
    vectorstore=None,
) -> GraphRAGRetriever:
    """获取 GraphRAG 检索器实例"""
    return GraphRAGRetriever(
        graph=graph,
        entity_index=entity_index,
        entity_metadata=entity_metadata,
        relationship_index=relationship_index,
        relationship_metadata=relationship_metadata,
        embedding=embedding,
        vectorstore=vectorstore,
    )
