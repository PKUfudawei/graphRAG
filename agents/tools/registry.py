"""工具注册表"""
from typing import Dict, Type, Any, Optional, Callable
from .base import BaseTool, ToolResult
from .graphrag_tool import GraphRAGTool


class SearchModeTool(BaseTool):
    """GraphRAG 检索模式工具包装器"""

    def __init__(self, mode: str, storage_path: str = "./storage/graphrag_index"):
        self.mode = mode
        self.graphrag_tool = GraphRAGTool(storage_path=storage_path)

    def get_name(self) -> str:
        return self.mode

    def search(self, query: str, **kwargs) -> ToolResult:
        """执行搜索"""
        return getattr(self.graphrag_tool, f"{self.mode}_search")(query, **kwargs)

    def structured_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """结构化搜索接口"""
        query = payload.get("query", "")
        result = getattr(self.graphrag_tool, f"{self.mode}_search")(query, **payload)
        return {
            "success": result.success,
            "answer": result.answer,
            "retrieval_results": result.evidence,
            "error": result.error
        }


class ToolRegistry:
    """工具注册表"""

    _tools: Dict[str, Any] = {}  # 可以是类或工厂函数

    @classmethod
    def register(cls, name: str, tool: Any) -> None:
        """注册工具（类或工厂函数）"""
        cls._tools[name] = tool

    @classmethod
    def unregister(cls, name: str) -> None:
        """取消注册工具"""
        if name in cls._tools:
            del cls._tools[name]

    @classmethod
    def get_tool(cls, name: str) -> Optional[Any]:
        """获取工具类或工厂函数"""
        return cls._tools.get(name)

    @classmethod
    def create_tool(cls, name: str, **kwargs) -> Optional[BaseTool]:
        """创建工具实例"""
        tool = cls.get_tool(name)
        if tool:
            try:
                # 如果是工厂函数
                if callable(tool) and not isinstance(tool, type):
                    return tool(**kwargs)
                # 如果是类
                elif isinstance(tool, type):
                    return tool(**kwargs)
            except TypeError:
                pass
        return None

    @classmethod
    def list_tools(cls) -> list:
        """列出所有已注册的工具"""
        return list(cls._tools.keys())

    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._tools.clear()


# 注册三种检索模式作为独立工具
def create_naive_tool(**kwargs) -> BaseTool:
    return SearchModeTool(mode="naive", **kwargs)

def create_local_tool(**kwargs) -> BaseTool:
    return SearchModeTool(mode="local", **kwargs)

def create_global_tool(**kwargs) -> BaseTool:
    return SearchModeTool(mode="global", **kwargs)

ToolRegistry.register("naive", create_naive_tool)
ToolRegistry.register("local", create_local_tool)
ToolRegistry.register("global", create_global_tool)

__all__ = ["ToolRegistry", "BaseTool", "ToolResult", "SearchModeTool"]
