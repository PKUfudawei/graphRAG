"""
GraphRAG Indexer - 整合实体提取、图构建和图存储
"""
import os
import sys
from typing import List, Optional
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument as LCGraphDocument
from langchain_text_splitters import TokenTextSplitter

# 支持直接运行和模块导入
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入依赖
from graphrag.index.graph_builder import GraphBuilder, build_graph
from graphrag.graph.entity_extractor import EntityExtractor, get_entity_extractor
from graphrag.graph.entity_deduplicator import EntityDeduplicator, get_entity_deduplicator
from graphrag.graph.graph_storage import Neo4jStorage, get_neo4j_storage


class GraphIndexer:
    """GraphRAG 索引器 - 整合实体提取、图构建和图存储"""

    def __init__(
        self,
        chunker: Optional[TokenTextSplitter] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        entity_deduplicator: Optional[EntityDeduplicator] = None,
        neo4j_storage: Optional[Neo4jStorage] = None,
        chunk_model: str = "cl100k_base",
        chunk_size: int = 512,
        overlap: int = 50,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: Optional[str] = None,
        dedup_threshold: float = 0.9,
    ):
        """
        初始化索引器

        Args:
            chunker: 预创建的 TokenTextSplitter 实例
            entity_extractor: 预创建的 EntityExtractor 实例
            entity_deduplicator: 预创建的 EntityDeduplicator 实例
            neo4j_storage: 预创建的 Neo4jStorage 实例
            chunk_model: tiktoken encoding 名称
            chunk_size: 每个 chunk 的 token 数量
            overlap: 相邻 chunk 之间的重叠 token 数量
            neo4j_uri: Neo4j 连接 URI
            neo4j_username: Neo4j 用户名
            neo4j_password: Neo4j 密码
            dedup_threshold: 实体去重阈值
        """
        self.chunker = chunker or self._get_chunker(chunk_model, chunk_size, overlap)
        self.entity_extractor = entity_extractor or get_entity_extractor()
        self.entity_deduplicator = entity_deduplicator
        self.neo4j_storage = neo4j_storage or get_neo4j_storage(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        self.dedup_threshold = dedup_threshold

    def _get_chunker(self, model: str, chunk_size: int, overlap: int) -> TokenTextSplitter:
        """获取分块器"""
        try:
            from models.chunker import get_chunker
            return get_chunker(model=model, chunk_size=chunk_size, overlap=overlap)
        except ImportError:
            return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """分块文档列表"""
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking"):
            chunks = self.chunker.create_documents(
                [doc.page_content],
                metadatas=[doc.metadata]
            )
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
        """分块文本列表"""
        if metadatas is None:
            metadatas = [{} for _ in texts]

        all_chunks = []
        for text, metadata in tqdm(zip(texts, metadatas), desc="Chunking"):
            chunks = self.chunker.create_documents(
                [text],
                metadatas=[metadata]
            )
            all_chunks.extend(chunks)
        return all_chunks

    def extract_graphs(
        self,
        documents: List[Document],
        parallel: bool = True
    ) -> List[LCGraphDocument]:
        """从文档列表提取知识图谱"""
        texts = [doc.page_content for doc in documents]
        sources = [doc.metadata.get('source', f'doc_{i}') for i, doc in enumerate(documents)]

        graph_docs = self.entity_extractor.extract_batch(
            texts,
            sources=sources,
            parallel=parallel
        )
        return graph_docs

    def deduplicate_graphs(self, graph_docs: List[LCGraphDocument]) -> List[LCGraphDocument]:
        """对提取的图谱进行实体去重"""
        if self.entity_deduplicator is None:
            self.entity_deduplicator = get_entity_deduplicator(threshold=self.dedup_threshold)

        # 收集所有实体名称
        all_entities = []
        for doc in graph_docs:
            all_entities.extend([node.id for node in doc.nodes])

        # 查找别名
        alias_map = self.entity_deduplicator.find_aliases(all_entities)

        # 应用别名映射
        deduplicated_docs = []
        for doc in graph_docs:
            new_nodes = []
            for node in doc.nodes:
                canonical_name = alias_map.get(node.id, node.id)
                if canonical_name != node.id:
                    node.id = canonical_name
                new_nodes.append(node)

            new_relationships = []
            for rel in doc.relationships:
                new_source_id = alias_map.get(rel.source.id, rel.source.id)
                new_target_id = alias_map.get(rel.target.id, rel.target.id)
                rel.source.id = new_source_id
                rel.target.id = new_target_id
                new_relationships.append(rel)

            deduplicated_docs.append(LCGraphDocument(
                nodes=new_nodes,
                relationships=new_relationships,
                source=doc.source
            ))

        return deduplicated_docs

    def build_graph(self, graph_docs: List[LCGraphDocument]) -> GraphBuilder:
        """从图谱文档构建 NetworkX 图"""
        builder = GraphBuilder()
        builder.add_documents(graph_docs)
        return builder

    def save_to_neo4j(self, graph_docs: List[LCGraphDocument]) -> None:
        """将图谱保存到 Neo4j"""
        for doc in tqdm(graph_docs, desc="Saving to Neo4j"):
            self.neo4j_storage.save_graph(doc)

    def index_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        build_graph: bool = True,
        save_to_neo4j: bool = True,
        deduplicate: bool = True,
    ) -> tuple[List[Document], List[LCGraphDocument], Optional[GraphBuilder]]:
        """
        一站式索引文本（分块 + 实体提取 + 图构建 + 图存储）

        Args:
            texts: 文本列表
            metadatas: 元数据列表
            build_graph: 是否构建 NetworkX 图
            save_to_neo4j: 是否保存到 Neo4j
            deduplicate: 是否进行实体去重

        Returns:
            (分块后的文档列表，图谱文档列表，NetworkX 图构建器或 None)
        """
        # 1. 分块
        documents = self.chunk_texts(texts, metadatas)

        # 2. 提取图谱
        graph_docs = self.extract_graphs(documents)

        # 3. 实体去重
        if deduplicate:
            graph_docs = self.deduplicate_graphs(graph_docs)

        # 4. 构建 NetworkX 图
        graph_builder = None
        if build_graph:
            graph_builder = self.build_graph(graph_docs)

        # 5. 保存到 Neo4j
        if save_to_neo4j:
            self.save_to_neo4j(graph_docs)

        return documents, graph_docs, graph_builder

    def index_files(
        self,
        file_paths: List[str],
        encoding: str = "utf-8",
        build_graph: bool = True,
        save_to_neo4j: bool = True,
        deduplicate: bool = True,
    ) -> tuple[List[Document], List[LCGraphDocument], Optional[GraphBuilder]]:
        """
        一站式索引文件（支持 Markdown 和纯文本）

        Args:
            file_paths: 文件路径列表
            encoding: 文件编码（仅用于 txt 文件）
            build_graph: 是否构建 NetworkX 图
            save_to_neo4j: 是否保存到 Neo4j
            deduplicate: 是否进行实体去重

        Returns:
            (分块后的文档列表，图谱文档列表，NetworkX 图构建器或 None)
        """
        from pathlib import Path

        all_docs = []
        for path in tqdm(file_paths, desc="Loading files"):
            file_path = Path(path)
            ext = file_path.suffix.lower()

            if ext in {".md", ".markdown"}:
                try:
                    from langchain_community.document_loaders import UnstructuredMarkdownLoader
                    loader = UnstructuredMarkdownLoader(str(path))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = str(path)
                    all_docs.extend(docs)
                except ImportError:
                    with open(path, "r", encoding=encoding) as f:
                        text = f.read()
                    all_docs.append(Document(page_content=text, metadata={"source": str(path)}))
            else:
                with open(path, "r", encoding=encoding) as f:
                    text = f.read()
                all_docs.append(Document(page_content=text, metadata={"source": str(path)}))

        return self.index_texts(
            [doc.page_content for doc in all_docs],
            metadatas=[doc.metadata for doc in all_docs],
            build_graph=build_graph,
            save_to_neo4j=save_to_neo4j,
            deduplicate=deduplicate,
        )


def get_graph_indexer(
    chunk_model: str = "cl100k_base",
    chunk_size: int = 512,
    overlap: int = 50,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_username: str = "neo4j",
    neo4j_password: Optional[str] = None,
    dedup_threshold: float = 0.9,
) -> GraphIndexer:
    """
    获取 GraphRAG 索引器实例

    Args:
        chunk_model: tiktoken encoding 名称
        chunk_size: 每个 chunk 的 token 数量
        overlap: 相邻 chunk 之间的重叠 token 数量
        neo4j_uri: Neo4j 连接 URI
        neo4j_username: Neo4j 用户名
        neo4j_password: Neo4j 密码
        dedup_threshold: 实体去重阈值

    Returns:
        GraphIndexer 实例
    """
    return GraphIndexer(
        chunk_model=chunk_model,
        chunk_size=chunk_size,
        overlap=overlap,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        dedup_threshold=dedup_threshold,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("GraphIndexer 测试")
    print("=" * 60)

    # 测试 1: 创建索引器
    print("\n[测试 1] 创建 GraphIndexer...")
    indexer = get_graph_indexer(chunk_size=200, overlap=20)
    print(f"  ✓ GraphIndexer 创建成功")

    # 测试 2: 分块测试
    print("\n[测试 2] 分块测试...")
    texts = [
        "北京是中国的首都，位于华北平原。北京拥有丰富的历史文化遗产。",
        "上海是中国最大的城市，位于长江入海口。上海是国际金融中心。",
    ]
    documents = indexer.chunk_texts(texts, metadatas=[{"source": "text1"}, {"source": "text2"}])
    print(f"  生成了 {len(documents)} 个分块")

    # 测试 3: 图谱提取测试（不实际调用 LLM）
    print("\n[测试 3] 图谱提取方法测试...")
    print(f"  提取器：{indexer.entity_extractor}")
    print(f"  ✓ 提取器已配置")

    # 测试 4: Neo4j 存储测试
    print("\n[测试 4] Neo4j 存储配置...")
    print(f"  Neo4j URI: {indexer.neo4j_storage.uri}")
    print(f"  ✓ Neo4j 存储已配置")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
