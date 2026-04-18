"""RAG 工具包装"""
import os
import sys
from typing import Optional, List, Dict, Any

from .base import BaseTool, ToolResult


class RAGTool(BaseTool):
    """RAG 检索工具"""

    def __init__(self, storage_path: str = "./storage/rag_index"):
        self.storage_path = storage_path
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

        self._embedding = get_embedding()

        # 加载 vectorstore
        embed_model = self._embedding.embed_model if hasattr(self._embedding, 'embed_model') else self._embedding
        self._vectorstore = FAISS.load_local(
            self.storage_path,
            embed_model,
            allow_dangerous_deserialization=True
        )
        self._initialized = True

    def get_name(self) -> str:
        return "rag"

    def search(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """执行 RAG 检索"""
        try:
            self._initialize()

            # 执行检索
            docs = self._vectorstore.similarity_search(query, k=top_k)

            # 转换为证据列表
            evidence = []
            answer_parts = []
            for i, doc in enumerate(docs):
                score = doc.metadata.get("score", 0.0) if hasattr(doc, "metadata") else 0.0
                evidence_item = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": score,
                    "metadata": doc.metadata
                }
                evidence.append(evidence_item)
                answer_parts.append(f"[文档 {i+1}]: {doc.page_content}")

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
        top_k = payload.get("top_k", 5)

        result = self.search(query, top_k=top_k)

        return {
            "success": result.success,
            "answer": result.answer,
            "retrieval_results": result.evidence,
            "error": result.error
        }
