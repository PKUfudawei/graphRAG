"""任务相关的数据模型"""
from dataclasses import dataclass, field
from typing import List, Optional, Any
from enum import Enum
import uuid
from datetime import datetime


class TaskType(Enum):
    """任务类型枚举"""
    RAG = "rag"                    # 简单向量检索
    GRAPH_RAG = "graphrag"         # 图谱检索
    DEEP_RESEARCH = "deep_research"  # 深度研究


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """子任务定义"""
    task_id: str
    task_type: TaskType
    query: str
    description: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    parameters: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "query": self.query,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """从字典创建"""
        return cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            query=data["query"],
            description=data.get("description"),
            depends_on=data.get("depends_on", []),
            status=TaskStatus(data.get("status", "pending")),
            parameters=data.get("parameters", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        )


def generate_task_id() -> str:
    """生成唯一任务 ID"""
    return f"task_{uuid.uuid4().hex[:8]}"
