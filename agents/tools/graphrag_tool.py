"""GraphRAG 工具包装"""
import os
import sys
import pickle
from typing import Optional, List, Dict, Any

from .base import BaseTool, ToolResult


class GraphRAGTool(BaseTool):
    """GraphRAG 检索工具 - 支持 naive_search, local_search, global_search"""

    def __init__(self, storage_path: str = "./storage/graphrag_index"):
        self.storage_path = storage_path
        self._graph = None
        self._entity_index = None
        self._entity_metadata = None
        self._relationship_index = None
        self._relationship_metadata = None
        self._vectorstore = None
        self._embedding = None
        self._initialized = False

    def _initialize(self):
        """延迟初始化，加载必要的组件"""
        if self._initialized:
            return

        # 添加项目路径
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        from models.embedding import get_embedding
        from langchain_community.vectorstores import FAISS

        # 加载 graph
        graph_path = os.path.join(self.storage_path, 'graph.pkl')
        with open(graph_path, "rb") as f:
            self._graph = pickle.load(f)

        # 加载 entities
        entities_path = os.path.join(self.storage_path, 'entities.pkl')
        with open(entities_path, "rb") as f:
            entities_data = pickle.load(f)
        self._entity_index = entities_data["index"]
        self._entity_metadata = entities_data["metadata"]

        # 加载 relationships
        relationships_path = os.path.join(self.storage_path, 'relationships.pkl')
        with open(relationships_path, "rb") as f:
            relationships_data = pickle.load(f)
        self._relationship_index = relationships_data["index"]
        self._relationship_metadata = relationships_data["metadata"]

        # 加载 vectorstore - 使用 CPU 设备
        self._embedding = get_embedding(model="BAAI/bge-m3", device="cpu")
        embed_model = self._embedding.embed_model
        self._vectorstore = FAISS.load_local(
            os.path.join(self.storage_path, 'vectorstore'),
            embeddings=embed_model,
            allow_dangerous_deserialization=True,
        )

        self._initialized = True

    def get_name(self) -> str:
        return "graphrag"

    def naive_search(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """朴素检索：纯向量检索"""
        try:
            self._initialize()
            from graphrag.retriever import get_graphrag_retriever
            retriever = get_graphrag_retriever(
                graph=self._graph,
                entity_index=self._entity_index,
                entity_metadata=self._entity_metadata,
                embedding=self._embedding,
                vectorstore=self._vectorstore
            )
            docs = retriever.naive_search(query=query, top_k=top_k)
            return self._docs_to_tool_result(docs, "naive")
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def local_search(self, query: str, top_k_entities: int = 3,
                     max_hops: int = 1, max_neighbors: int = 3, **kwargs) -> ToolResult:
        """局部检索：实体检索 + 单跳遍历，适合精确问答"""
        try:
            self._initialize()
            from graphrag.retriever import get_graphrag_retriever
            retriever = get_graphrag_retriever(
                graph=self._graph,
                entity_index=self._entity_index,
                entity_metadata=self._entity_metadata,
                embedding=self._embedding,
                vectorstore=self._vectorstore
            )
            docs = retriever.local_search(
                query=query,
                top_k_entities=top_k_entities,
                max_hops=max_hops,
                max_neighbors=max_neighbors
            )
            return self._docs_to_tool_result(docs, "local")
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def global_search(self, query: str, top_k_vectors: int = 5, top_k_relationships: int = 10,
                      **kwargs) -> ToolResult:
        """全局检索：向量检索 + 关系检索（无往外跳），适合综合问题"""
        try:
            self._initialize()
            from graphrag.retriever import get_graphrag_retriever
            retriever = get_graphrag_retriever(
                graph=self._graph,
                entity_index=self._entity_index,
                entity_metadata=self._entity_metadata,
                relationship_index=self._relationship_index,
                relationship_metadata=self._relationship_metadata,
                embedding=self._embedding,
                vectorstore=self._vectorstore
            )
            docs = retriever.global_search(
                query=query,
                top_k_vectors=top_k_vectors,
                top_k_relationships=top_k_relationships
            )
            return self._docs_to_tool_result(docs, "global")
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def search(self, query: str, mode: str = "global", **kwargs) -> ToolResult:
        """执行 GraphRAG 检索（统一入口）

        Args:
            query: 查询文本
            mode: 检索模式 (naive/local/global)
            **kwargs: 各模式的额外参数
        """
        try:
            if mode == "naive":
                return self.naive_search(query, **kwargs)
            elif mode == "local":
                return self.local_search(query, **kwargs)
            else:  # global
                return self.global_search(query, **kwargs)
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def _docs_to_tool_result(self, docs: list, mode: str) -> ToolResult:
        """将 Document 列表转换为 ToolResult"""
        evidence = []
        answer_parts = []
        for i, doc in enumerate(docs):
            retrieval_type = doc.metadata.get("retrieval_type", "unknown")
            score = doc.metadata.get("score", 0.0)
            evidence_item = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "retrieval_type": retrieval_type,
                "score": score,
                "metadata": doc.metadata
            }
            evidence.append(evidence_item)
            type_str = {"naive": "向量", "local": "局部图谱", "global": "全局"}.get(mode, "结果")
            answer_parts.append(f"[{type_str} 结果 {i+1}]: {doc.page_content}")
        answer = "\n\n".join(answer_parts) if answer_parts else "未找到相关信息"
        return ToolResult(success=True, answer=answer, evidence=evidence)

    def structured_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """结构化搜索接口（用于 Executor）"""
        query = payload.get("query", "")
        mode = payload.get("mode", "global")  # naive/local/global

        result = self.search(query=query, mode=mode, **payload)

        return {
            "success": result.success,
            "answer": result.answer,
            "retrieval_results": result.evidence,
            "error": result.error
        }
