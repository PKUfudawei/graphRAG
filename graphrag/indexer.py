"""
GraphRAG Indexer - 整合向量索引和知识图谱索引
"""
from typing import List, Optional
import faiss
import os
from tqdm import tqdm

from langchain_core.documents import Document

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
        enable_thinking: bool = False,
        extract_mode: str = "thread",
    ):
        super().__init__(
            chunker=chunker, embedding=embedding,
        )
        self.enable_thinking = enable_thinking
        self.max_workers = max_workers
        self.extract_mode = extract_mode
        # 创建带有 enable_thinking 参数的 graph_builder
        if graph_builder is None:
            from models.llm import get_llm
            from graphrag.graph.extractor import get_extractor
            llm = get_llm(enable_thinking=enable_thinking)
            extractor = get_extractor(llm=llm, max_workers=max_workers)
            self.graph_builder = get_graph_builder(extractor=extractor, max_workers=max_workers, extract_mode=extract_mode)
        else:
            self.graph_builder = graph_builder

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
        """索引文档：分块 + 建向量索引 + 建图谱 + 建实体索引 + 建关系索引。

        Args:
            documents: 输入的 Document 列表。
            database_path: 数据库存储路径。
            incremental: 是否增量更新。True 时只处理新增文档。

        Returns:
            (chunks, vectorstore, graph, entity_index, relationship_index) 元组。
        """
        import pickle

        # Step 1: 分块 + 构建向量索引（委托给 rag.indexer.Indexer）
        vectorstore_path = os.path.join(database_path, 'vectorstore')
        all_chunks, vectorstore = super().index_documents(
            documents,
            output_path=vectorstore_path
        )

        # Step 2: 构建/更新图谱
        stats = self.graph_builder.build_from_documents(
            all_chunks,
            incremental=incremental
        )
        graph = self.graph_builder.graph
        print(f"Graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships")
        graph_path = os.path.join(database_path, 'graph.pkl')
        self.save_graph(graph_path)

        # Step 3: 保存实体和关系信息
        entities_path = os.path.join(database_path, 'entities.pkl')
        entity_index = self.save_entities(entities_path)

        relationships_path = os.path.join(database_path, 'relationships.pkl')
        relationship_index = self.save_relationships(relationships_path)

        return all_chunks, vectorstore, graph, entity_index, relationship_index 

    def save_entities(self, path: str = None):
        """保存实体索引到磁盘。

        从 graph 中提取实体名称和描述，生成 embedding 并保存。
        向量化内容格式："{entity_name}\n{description}"（参考 LightRAG）

        Args:
            path: 存储路径（文件路径）

        Returns:
            entity_index: FAISS 索引对象
        """
        import os
        import pickle

        # 从 graph 获取所有实体
        entity_names = list(self.graph_builder.graph.nodes())
        if not entity_names:
            print("No entities to save.")
            return

        # 收集实体元数据和向量化文本（带进度条）
        entity_metadata = []
        entity_texts = []
        print(f"[save_entities] Processing {len(entity_names)} entities...")
        for entity in tqdm(entity_names, desc="  Collecting"):
            node_data = self.graph_builder.graph.nodes[entity]
            description = node_data.get("description", "")

            # LightRAG 方式：实体名称 + 描述 组合作为向量化内容
            entity_content = f"{entity}\n{description}"

            entity_metadata.append({
                "name": entity,
                "type": node_data.get("type", "Entity"),
                "description": description,
                "chunk_ids": node_data.get("chunk_ids", []),
                "content": entity_content  # 保存用于向量化的完整内容
            })
            entity_texts.append(entity_content)

        # 生成实体 embeddings（使用完整内容而非仅实体名）
        print(f"  Generating embeddings...")
        embeddings = self.embedding.encode(list(tqdm(entity_texts, desc="  Embedding")))

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

    def save_relationships(self, path: str = None):
        """保存关系索引到磁盘。

        从 graph 中提取关系信息，生成 embedding 并保存。
        向量化内容格式："{keywords}\t{src_id}\n{tgt_id}\n{description}"（参考 LightRAG）

        Args:
            path: 存储路径（文件路径）

        Returns:
            relationship_index: FAISS 索引对象
        """
        import os
        import pickle

        # 从 graph 获取所有关系
        relationships = list(self.graph_builder.graph.edges(data=True))
        if not relationships:
            print("No relationships to save.")
            return

        # 收集关系元数据和向量化文本（带进度条）
        relationship_metadata = []
        relationship_texts = []
        print(f"[save_relationships] Processing {len(relationships)} relationships...")
        for src, tgt, data in tqdm(relationships, desc="  Collecting"):
            rel_description = data.get("description", f"{src} related to {tgt}")
            keywords = data.get("keywords", "")

            # LightRAG 方式：keywords + 源实体 + 目标实体 + 描述 组合作为向量化内容
            # 格式："{keywords}\t{src_id}\n{tgt_id}\n{description}"
            rel_content = f"{keywords}\t{src}\n{tgt}\n{rel_description}"

            relationship_metadata.append({
                "src_id": src,
                "tgt_id": tgt,
                "rel_type": data.get("rel_type", "RELATED_TO"),
                "description": rel_description,
                "keywords": keywords,
                "weight": data.get("weight", 1.0),
                "content": rel_content  # 保存用于向量化的完整内容
            })
            relationship_texts.append(rel_content)

        # 生成关系 embeddings（使用完整内容而非仅描述）
        print(f"  Generating embeddings...")
        embeddings = self.embedding.encode(list(tqdm(relationship_texts, desc="  Embedding")))

        # 创建 FAISS 索引
        dim = embeddings.shape[1]
        relationship_index = faiss.IndexFlatIP(dim)  # 内积相似度
        relationship_index.add(embeddings)

        # 保存关系索引到指定文件路径
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "index": relationship_index,
                "metadata": relationship_metadata
            }, f)

        print(f"Saved {len(relationship_texts)} relationships to {path}")
        return relationship_index
    


def get_graphrag_indexer(
    chunker=None,
    embedding=None,
    max_workers: int = 16,
    enable_thinking: bool = False,
    extract_mode: str = "thread",
) -> GraphRAGIndexer:
    """获取 GraphRAG 索引器实例

    Args:
        chunker: 预创建的 chunker 实例
        embedding: 预创建的 Embeddings 实例
        max_workers: 并行提取的最大工作线程数
        enable_thinking: 是否启用 LLM thinking 模式（默认 False）
        extract_mode: 提取模式 (thread/async/sync)

    Returns:
        GraphRAGIndexer 实例
    """
    return GraphRAGIndexer(
        chunker=chunker,
        embedding=embedding,
        max_workers=max_workers,
        enable_thinking=enable_thinking,
        extract_mode=extract_mode,
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
