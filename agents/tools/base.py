"""工具基类和结果"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    answer: str
    evidence: list  # List[Dict]
    error: Optional[str] = None


class BaseTool(ABC):
    """工具基类"""

    @abstractmethod
    def search(self, query: str, **kwargs) -> ToolResult:
        """执行搜索"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """获取工具名称"""
        pass
