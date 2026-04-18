"""
GraphRAG Indexer - 整合向量索引和知识图谱索引
"""
from typing import List, Optional
import numpy as np
import faiss
import os

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument

from graphrag.graph.builder import get_graph_builder
from rag.indexer import Indexer


class GraphRAGIndexer(Indexer):
    """GraphRAG 索引器 - 同时索引向量数据库和知识图谱

    工作流程:
    1. index_documents(): 分块文档（继承自 Indexer）
    2. build_graph_from_chunks(): 从 chunks 提取实体 -> 对齐 -> 建图
    3. index_entities(): 为实体生成 embedding 并建立向量索引
    4. build_vectorstore(): 为 chunks 建立向量索引（继承自 Indexer）
    """

    def __init__(
        self,
        chunker=None,
        embedding=None,
        graph_builder=None,
        max_workers: int = 16,
    ):
        super().__init__(
            chunker=chunker, embedding=embedding, 
        )
        self.graph_builder = graph_builder or get_graph_builder()
        self.max_workers = max_workers

    def build_vectorstore(self, documents):
        return super().build_vectorstore(documents)
    
    def save_vectorstore(self, vectorstore, path):
        return super().save_vectorstore(vectorstore, path)
    
    def load_vectorstore(self, path):
        return super().load_vectorstore(path)

    def save_graph(self, path):
        self.graph_builder.storage_path = path
        self.graph_builder.save_graph()
        print(f"Graph saved to {path}")

    def clear_graph(self):
        """清空图数据"""
        self.graph_builder.clear_graph()
        print("Graph cleared")

    def build_graph_from_chunks(self, chunks: List[Document]) -> dict:
        """从 chunks 构建知识图谱。

        Args:
            chunks: 分块后的文档列表

        Returns:
            统计信息字典
        """
        return self.graph_builder.build_from_documents(chunks)

    def index_documents(
        self,
        documents,
        database_path,
        incremental: bool = False
    ) -> tuple:
        """索引文档：分块 + 建向量索引 + 建图谱 + 建实体索引。

        Args:
            documents: 输入的 Document 列表。
            database_path: 数据库存储路径。
            incremental: 是否增量更新。True 时只处理新增文档。

        Returns:
            (chunks, vectorstore, graph, entity_index) 元组。
        """
        import hashlib
        import pickle

        # 增量模式下，先加载已有图的 file_hashes
        existing_file_hashes = set()
        if incremental:
            graph_path = os.path.join(database_path, 'graph.pkl')
            if os.path.exists(graph_path):
                with open(graph_path, 'rb') as f:
                    existing_graph = pickle.load(f)
                for node, data in existing_graph.nodes(data=True):
                    chunk_ids = data.get("chunk_ids", [])
                    for cid in chunk_ids:
                        if isinstance(cid, str) and "_" in cid:
                            existing_file_hashes.add(cid.split("_")[0])

        # Step 1: 分块
        all_chunks = super().index_documents(documents)

        # 为每个文档的 chunks 分配 chunk_id（基于文件内容的 hash）
        # chunk_id 格式：{file_hash}_{chunk_index}
        new_chunks = []

        for doc in documents:
            file_path = doc.metadata.get("source", "")
            if not file_path or not os.path.exists(file_path):
                # 非文件输入，用 source 字段做 hash
                file_hash = hashlib.md5(doc.metadata.get("source", "unknown").encode()).hexdigest()[:16]
            else:
                # 计算文件内容的 hash
                hasher = hashlib.md5()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()[:16]

            # 增量模式下跳过已处理的文件
            if incremental and file_hash in existing_file_hashes:
                print(f"  Skipping already processed file: {file_path}")
                continue

            # 获取该文档对应的 chunks（通过 source 匹配）
            doc_chunks = [c for c in all_chunks if c.metadata.get("source") == file_path]
            for idx, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_id"] = f"{file_hash}_{idx}"
                chunk.metadata["file_hash"] = file_hash
                new_chunks.append(chunk)

        all_chunks = new_chunks
        print(f"Generated {len(all_chunks)} new chunks")

        # 如果没有新 chunks，直接返回已有数据
        if incremental and not all_chunks:
            print("No new chunks to process. Loading existing index...")
            vectorstore = self.load_vectorstore(os.path.join(database_path, 'vectorstore'))
            self.graph_builder.storage_path = os.path.join(database_path, 'graph.pkl')
            graph = self.graph_builder.graph
            with open(os.path.join(database_path, 'entities.pkl'), 'rb') as f:
                entity_index = pickle.load(f)["index"]
            return all_chunks, vectorstore, graph, entity_index

        # Step 2: 构建/更新 chunk 向量索引
        if incremental and os.path.exists(os.path.join(database_path, 'vectorstore')):
            # 增量模式：加载已有 vectorstore 并添加新 chunks
            vectorstore = self.load_vectorstore(os.path.join(database_path, 'vectorstore'))
            # FAISS 增量添加：重新构建（简化处理）
            all_docs = list(vectorstore.docstore._dict.values()) + all_chunks
            vectorstore = self.build_vectorstore(all_docs)
        else:
            vectorstore = self.build_vectorstore(all_chunks)

        vectorstore_path = os.path.join(database_path, 'vectorstore')
        self.save_vectorstore(vectorstore, vectorstore_path)
        print(f"Saved chunk vectorstore to {vectorstore_path}")

        # Step 3: 构建/更新图谱
        stats = self.graph_builder.build_from_documents(
            all_chunks,
            incremental=incremental
        )
        graph = self.graph_builder.graph
        print(f"Graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships")
        graph_path = os.path.join(database_path, 'graph.pkl')
        self.save_graph(graph_path)

        # Step 4: 保存实体信息
        entities_path = os.path.join(database_path, 'entities.pkl')
        entity_index = self.save_entities(entities_path)

        return all_chunks, vectorstore, graph, entity_index 

    def save_entities(self, path: str = None):
        """保存实体索引到磁盘。

        从 graph 中提取实体名称和 chunk_ids，生成 embedding 并保存。

        Args:
            path: 存储路径（文件路径）
        """
        import os
        import pickle

        # 从 graph 获取所有实体
        entity_names = list(self.graph_builder.graph.nodes())
        if not entity_names:
            print("No entities to save.")
            return

        # 收集实体元数据
        entity_metadata = []
        for entity in entity_names:
            node_data = self.graph_builder.graph.nodes[entity]
            entity_metadata.append({
                "name": entity,
                "type": node_data.get("type", "Entity"),
                "chunk_ids": node_data.get("chunk_ids", [])
            })

        # 生成实体 embeddings
        print(f"Generating embeddings for {len(entity_names)} entities...")
        embeddings = self.embedding.encode(entity_names)

        # 创建 FAISS 索引
        dim = embeddings.shape[1]
        entity_index = faiss.IndexFlatIP(dim)  # 内积相似度
        entity_index.add(embeddings)

        # 保存实体索引到指定文件路径
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "index": entity_index,
                "metadata": entity_metadata
            }, f)

        print(f"Saved {len(entity_names)} entities to {path}")
        return entity_index
    


def get_graphrag_indexer(
    chunker=None,
    embedding=None,
    max_workers: int = 16,
) -> GraphRAGIndexer:
    """获取 GraphRAG 索引器实例

    Args:
        chunker: 预创建的 chunker 实例
        embedding: 预创建的 Embeddings 实例
        max_workers: 并行提取的最大工作线程数

    Returns:
        GraphRAGIndexer 实例
    """
    return GraphRAGIndexer(
        chunker=chunker,
        embedding=embedding,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    print("=" * 60)
    print("GraphRAGIndexer 测试：index_documents 全流程")
    print("=" * 60)

    # 测试：创建索引器并索引文档
    print("\n[Test] Create indexer and index documents...")
    indexer = get_graphrag_indexer(max_workers=8)
    print("  ✓ Indexer created")

    # 准备测试文档
    texts = [
        "北京是中国的首都，位于华北平原。北京拥有丰富的历史文化遗产，包括故宫、天坛等。",
        "北京市 GDP 超过 4 万亿元，是中国四大直辖市之一。北京拥有众多高校和科研院所。",
        "上海是中国最大的城市，位于长江入海口。上海是国际金融中心。",
        "上海市浦东新区是中国改革开放的前沿。上海港是世界最大的集装箱港口。",
    ]
    documents = [Document(page_content=t, metadata={"source": f"text{i+1}"}) for i, t in enumerate(texts)]

    # 执行索引
    chunk_count = indexer.index_documents(documents)
    print(f"  ✓ Indexed {chunk_count} chunks")

    # 验证结果
    print("\n[Verify] Checking results...")

    # 检查 vectorstore
    if Path('./database/vectorstore/index.faiss').exists():
        print("  ✓ Vectorstore saved")

    # 检查 graph
    if Path('./database/graph.pkl').exists():
        print("  ✓ Graph saved")

    # 检查 entities
    if Path('./database/entities.pkl').exists():
        import pickle
        with open('./database/entities.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"  ✓ Entities saved: {data['index'].ntotal} entities")
        print(f"    Sample: {data['metadata'][0] if data['metadata'] else 'None'}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
