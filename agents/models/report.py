"""报告相关的数据模型"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from .evidence import Evidence, EvidenceChain


@dataclass
class ReportSection:
    """报告章节"""
    section_id: str
    title: str
    content: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "evidence_ids": self.evidence_ids
        }


@dataclass
class Report:
    """最终报告"""
    report_id: str
    query: str
    final_answer: str
    sections: List[ReportSection] = field(default_factory=list)
    evidence_chain: Optional[EvidenceChain] = None
    plan_id: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.now)

    # 结构化证据（用于前端展示）
    structured_evidence: Dict[str, Any] = field(default_factory=dict)

    def add_section(self, section: ReportSection) -> None:
        """添加章节"""
        self.sections.append(section)

    def build_structured_evidence(self) -> None:
        """构建结构化证据"""
        if self.evidence_chain:
            self.structured_evidence = {
                "total_evidence": len(self.evidence_chain.evidence_list),
                "evidence_by_source": {
                    "rag": len(self.evidence_chain.get_evidence_by_source("rag")),
                    "graphrag": len(self.evidence_chain.get_evidence_by_source("graphrag"))
                },
                "top_evidence": [e.to_dict() for e in self.evidence_chain.get_top_evidence(10)],
                "reasoning_steps": self.evidence_chain.reasoning_steps,
                "graph_paths": self.evidence_chain.graph_paths
            }

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "report_id": self.report_id,
            "query": self.query,
            "final_answer": self.final_answer,
            "sections": [s.to_dict() for s in self.sections],
            "plan_id": self.plan_id,
            "generated_at": self.generated_at.isoformat(),
            "structured_evidence": self.structured_evidence
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Report":
        """从字典创建"""
        return cls(
            report_id=data["report_id"],
            query=data["query"],
            final_answer=data["final_answer"],
            sections=[ReportSection(**s) for s in data.get("sections", [])],
            plan_id=data.get("plan_id"),
            generated_at=datetime.fromisoformat(data["generated_at"]) if data.get("generated_at") else datetime.now(),
            structured_evidence=data.get("structured_evidence", {})
        )


def generate_report_id() -> str:
    """生成唯一报告 ID"""
    return f"report_{uuid.uuid4().hex[:8]}"
