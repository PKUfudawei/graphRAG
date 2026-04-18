"""工具注册表"""
from typing import Dict, Type, Any, Optional
from .base import BaseTool, ToolResult
from .rag_tool import RAGTool
from .graphrag_tool import GraphRAGTool


class ToolRegistry:
    """工具注册表"""

    _tools: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, name: str, tool_class: Type[BaseTool]) -> None:
        """注册工具"""
        cls._tools[name] = tool_class

    @classmethod
    def unregister(cls, name: str) -> None:
        """取消注册工具"""
        if name in cls._tools:
            del cls._tools[name]

    @classmethod
    def get_tool(cls, name: str) -> Optional[Type[BaseTool]]:
        """获取工具类"""
        return cls._tools.get(name)

    @classmethod
    def create_tool(cls, name: str, **kwargs) -> Optional[BaseTool]:
        """创建工具实例"""
        tool_class = cls.get_tool(name)
        if tool_class:
            try:
                return tool_class(**kwargs)
            except TypeError:
                return tool_class()
        return None

    @classmethod
    def list_tools(cls) -> list:
        """列出所有已注册的工具"""
        return list(cls._tools.keys())

    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._tools.clear()


# 默认注册工具
ToolRegistry.register("rag", RAGTool)
ToolRegistry.register("graphrag", GraphRAGTool)

__all__ = ["ToolRegistry", "BaseTool", "ToolResult"]
