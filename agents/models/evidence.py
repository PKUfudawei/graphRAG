"""证据相关的数据模型"""
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from enum import Enum
import uuid
from datetime import datetime


class EvidenceSource(Enum):
    """证据来源"""
    RAG = "rag"
    GRAPH_RAG = "graphrag"
    VECTOR = "vector"
    GRAPH = "graph"


@dataclass
class Evidence:
    """单条证据"""
    evidence_id: str
    source: EvidenceSource
    content: str
    score: float
    task_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "evidence_id": self.evidence_id,
            "source": self.source.value,
            "content": self.content,
            "score": self.score,
            "task_id": self.task_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        """从字典创建"""
        return cls(
            evidence_id=data["evidence_id"],
            source=EvidenceSource(data["source"]),
            content=data["content"],
            score=data["score"],
            task_id=data["task_id"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        )


@dataclass
class EvidenceChain:
    """证据链 - 追踪推理过程中的所有证据"""
    chain_id: str
    query: str
    evidence_list: List[Evidence] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    graph_paths: List[List[str]] = field(default_factory=list)  # 图谱路径

    def add_evidence(self, evidence: Evidence) -> None:
        """添加证据"""
        self.evidence_list.append(evidence)

    def add_reasoning_step(self, step: str) -> None:
        """添加推理步骤"""
        self.reasoning_steps.append(step)

    def add_graph_path(self, path: List[str]) -> None:
        """添加图谱路径"""
        self.graph_paths.append(path)

    def get_evidence_by_task(self, task_id: str) -> List[Evidence]:
        """获取指定任务的所有证据"""
        return [e for e in self.evidence_list if e.task_id == task_id]

    def get_evidence_by_source(self, source: EvidenceSource) -> List[Evidence]:
        """获取指定来源的所有证据"""
        return [e for e in self.evidence_list if e.source == source]

    def get_top_evidence(self, top_k: int = 5) -> List[Evidence]:
        """获取评分最高的证据"""
        sorted_evidence = sorted(self.evidence_list, key=lambda x: x.score, reverse=True)
        return sorted_evidence[:top_k]

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "chain_id": self.chain_id,
            "query": self.query,
            "evidence_list": [e.to_dict() for e in self.evidence_list],
            "reasoning_steps": self.reasoning_steps,
            "graph_paths": self.graph_paths,
            "total_evidence": len(self.evidence_list)
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceChain":
        """从字典创建"""
        chain = cls(
            chain_id=data["chain_id"],
            query=data["query"],
            evidence_list=[Evidence.from_dict(e) for e in data.get("evidence_list", [])],
            reasoning_steps=data.get("reasoning_steps", []),
            graph_paths=data.get("graph_paths", [])
        )
        return chain


def generate_evidence_id() -> str:
    """生成唯一证据 ID"""
    return f"evidence_{uuid.uuid4().hex[:8]}"


def generate_chain_id() -> str:
    """生成唯一证据链 ID"""
    return f"chain_{uuid.uuid4().hex[:8]}"
