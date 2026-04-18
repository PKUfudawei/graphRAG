"""GraphRAG 工具包装"""
import os
import sys
import pickle
from typing import Optional, List, Dict, Any

from .base import BaseTool, ToolResult


class GraphRAGTool(BaseTool):
    """GraphRAG 检索工具"""

    def __init__(self, storage_path: str = "./storage/graphrag_index"):
        self.storage_path = storage_path
        self._graph = None
        self._entity_index = None
        self._entity_metadata = None
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

        # 加载 vectorstore
        self._embedding = get_embedding()
        embed_model = self._embedding.embed_model if hasattr(self._embedding, 'embed_model') else self._embedding
        self._vectorstore = FAISS.load_local(
            os.path.join(self.storage_path, 'vectorstore'),
            embed_model,
            allow_dangerous_deserialization=True
        )

        self._initialized = True

    def get_name(self) -> str:
        return "graphrag"

    def search(self, query: str, top_k_vectors: int = 5, top_k_entities: int = 3,
               max_hops: int = 2, max_neighbors: int = 5, **kwargs) -> ToolResult:
        """执行 GraphRAG 检索"""
        try:
            self._initialize()

            # 创建 retriever
            from graphrag.retriever import get_graphrag_retriever
            retriever = get_graphrag_retriever(
                graph=self._graph,
                entity_index=self._entity_index,
                entity_metadata=self._entity_metadata,
                embedding=self._embedding,
                vectorstore=self._vectorstore
            )

            # 执行检索
            docs = retriever.retrieve(
                query=query,
                top_k_vectors=top_k_vectors,
                top_k_entities=top_k_entities,
                max_hops=max_hops,
                max_neighbors=max_neighbors
            )

            # 转换为证据列表
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

                type_str = "图谱" if retrieval_type == "graph" else "向量"
                answer_parts.append(f"[{type_str} 结果 {i+1}]: {doc.page_content}")

            answer = "\n\n".join(answer_parts) if answer_parts else "未找到相关信息"

            return ToolResult(
                success=True,
                answer=answer,
                evidence=evidence
            )
        except Exception as e:
            return ToolResult(
                success=False,
                answer="",
                evidence=[],
                error=str(e)
            )

    def structured_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """结构化搜索接口（用于 Executor）"""
        query = payload.get("query", "")
        top_k_vectors = payload.get("top_k_vectors", 5)
        top_k_entities = payload.get("top_k_entities", 3)
        max_hops = payload.get("max_hops", 2)
        max_neighbors = payload.get("max_neighbors", 5)

        result = self.search(
            query=query,
            top_k_vectors=top_k_vectors,
            top_k_entities=top_k_entities,
            max_hops=max_hops,
            max_neighbors=max_neighbors
        )

        return {
            "success": result.success,
            "answer": result.answer,
            "retrieval_results": result.evidence,
            "error": result.error
        }
