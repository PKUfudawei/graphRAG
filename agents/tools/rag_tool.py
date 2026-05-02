"""RAG 工具包装"""
import os
import sys
from typing import Optional, List, Dict, Any

from .base import BaseTool, ToolResult


class RAGTool(BaseTool):
    """RAG 检索工具 - 支持 vector_search, bm25_search, hybrid_search"""

    def __init__(self, storage_path: str = "./storage/rag_index"):
        self.storage_path = storage_path
        self._vectorstore = None
        self._embedding = None
        self._retriever = None
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
        from rag.retriever import Retriever

        # 使用 CPU 设备加载 embedding 模型
        self._embedding = get_embedding(model="BAAI/bge-m3", device="cpu")

        # 加载 vectorstore
        embed_model = self._embedding.embed_model
        self._vectorstore = FAISS.load_local(
            self.storage_path,
            embeddings=embed_model,
            allow_dangerous_deserialization=True,
        )

        # 创建 retriever
        self._retriever = Retriever(vectorstore=self._vectorstore, top_k=5)
        self._initialized = True

    def get_name(self) -> str:
        return "rag"

    def vector_search(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """向量检索"""
        try:
            self._initialize()
            self._retriever.top_k = top_k
            docs = self._retriever.vector_search(query)
            return self._docs_to_tool_result(docs, "vector")
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def bm25_search(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """BM25 关键词检索"""
        try:
            self._initialize()
            self._retriever.top_k = top_k
            docs = self._retriever.bm25_search(query)
            return self._docs_to_tool_result(docs, "bm25")
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def hybrid_search(self, query: str, top_k: int = 5, vector_weight: float = 0.5, bm25_weight: float = 0.5, **kwargs) -> ToolResult:
        """混合检索（向量 + BM25）"""
        try:
            self._initialize()
            self._retriever.top_k = top_k
            docs = self._retriever.hybrid_search(query, k=top_k, vector_weight=vector_weight, bm25_weight=bm25_weight)
            return self._docs_to_tool_result(docs, "hybrid")
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def search(self, query: str, mode: str = "vector", top_k: int = 5, **kwargs) -> ToolResult:
        """执行 RAG 检索（统一入口）

        Args:
            query: 查询文本
            mode: 检索模式 (vector/bm25/hybrid)
            top_k: 返回文档数量
            **kwargs: 各模式的额外参数
        """
        try:
            if mode == "vector":
                return self.vector_search(query, top_k=top_k, **kwargs)
            elif mode == "bm25":
                return self.bm25_search(query, top_k=top_k, **kwargs)
            elif mode == "hybrid":
                return self.hybrid_search(query, top_k=top_k, **kwargs)
            else:
                return ToolResult(success=False, answer="", evidence=[], error=f"Unknown mode: {mode}")
        except Exception as e:
            return ToolResult(success=False, answer="", evidence=[], error=str(e))

    def _docs_to_tool_result(self, docs: list, mode: str) -> ToolResult:
        """将 Document 列表转换为 ToolResult"""
        evidence = []
        answer_parts = []
        for i, doc in enumerate(docs):
            score = doc.metadata.get("score", 0.0)
            evidence_item = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "retrieval_type": mode,
                "score": score,
                "metadata": doc.metadata
            }
            evidence.append(evidence_item)
            type_str = {"vector": "向量", "bm25": "BM25", "hybrid": "混合"}.get(mode, "结果")
            answer_parts.append(f"[{type_str} 结果 {i+1}]: {doc.page_content}")
        answer = "\n\n".join(answer_parts) if answer_parts else "未找到相关信息"
        return ToolResult(success=True, answer=answer, evidence=evidence)

    def structured_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """结构化搜索接口（用于 Executor）"""
        query = payload.get("query", "")
        mode = payload.get("mode", "vector")
        top_k = payload.get("top_k", 5)

        result = self.search(query=query, mode=mode, top_k=top_k, **payload)

        return {
            "success": result.success,
            "answer": result.answer,
            "retrieval_results": result.evidence,
            "error": result.error
        }
