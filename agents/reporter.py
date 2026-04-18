"""Reporter Agent - 报告生成"""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from agents.models.plan import Plan
from agents.models.evidence import Evidence, EvidenceChain
from agents.models.report import Report, ReportSection, generate_report_id
from agents.executor import ExecutorResult, TaskExecutionResult

logger = logging.getLogger(__name__)


# Reporter Prompt
REPORTER_PROMPT = """
你是一个报告生成专家。你的任务是根据检索到的证据生成结构化的最终答案。

## 用户查询
{query}

## 检索到的证据

{evidence_context}

## 要求

1. 根据证据生成准确、简洁的答案
2. 在答案中标注证据来源（如 [RAG-1], [GraphRAG-2]）
3. 如果证据不足以回答问题，请说明
4. 答案应该结构清晰，便于理解

## 输出格式

直接返回答案文本，不需要额外的解释。
"""


class ReporterResult:
    """Reporter 执行结果"""
    def __init__(self, report: Optional[Report] = None, error: Optional[str] = None):
        self.report = report
        self.error = error
        self.success = report is not None and error is None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "report": self.report.to_dict() if self.report else None,
            "error": self.error
        }


class Reporter:
    """报告生成器"""

    def __init__(self, llm=None):
        """
        初始化 Reporter

        Args:
            llm: LLM 实例，如果为 None 则使用默认 LLM
        """
        self.llm = llm
        self._llm_initialized = False

    def _initialize_llm(self):
        """延迟初始化 LLM"""
        if self._llm_initialized:
            return

        if self.llm is None:
            from models.llm import get_llm
            self.llm = get_llm()

        self._llm_initialized = True

    def generate(self, executor_result: ExecutorResult, plan: Optional[Plan] = None) -> ReporterResult:
        """
        生成报告

        Args:
            executor_result: Executor 执行结果
            plan: 任务计划（可选）

        Returns:
            ReporterResult 对象
        """
        try:
            self._initialize_llm()

            evidence_chain = executor_result.evidence_chain
            query = evidence_chain.query

            # 构建证据上下文
            evidence_context = self._build_evidence_context(evidence_chain)

            # 生成答案
            answer = self._generate_answer(query, evidence_context)

            # 构建报告
            report = Report(
                report_id=generate_report_id(),
                query=query,
                final_answer=answer,
                evidence_chain=evidence_chain,
                plan_id=plan.plan_id if plan else None
            )

            # 构建结构化证据
            report.build_structured_evidence()

            return ReporterResult(report=report)

        except Exception as e:
            logger.error(f"Reporter 执行失败：{e}")
            return ReporterResult(error=str(e))

    def _build_evidence_context(self, evidence_chain: EvidenceChain) -> str:
        """
        构建证据上下文文本

        Args:
            evidence_chain: 证据链

        Returns:
            证据上下文文本
        """
        lines = []

        # 按来源分组
        rag_evidence = evidence_chain.get_evidence_by_source("rag")
        graphrag_evidence = evidence_chain.get_evidence_by_source("graphrag")

        if rag_evidence:
            lines.append("=== RAG 检索结果 ===")
            for i, ev in enumerate(rag_evidence, 1):
                content_preview = ev.content[:200] + "..." if len(ev.content) > 200 else ev.content
                lines.append(f"[RAG-{i}] (分数：{ev.score:.3f})")
                lines.append(f"  {content_preview}")
                if ev.metadata.get("source"):
                    lines.append(f"  来源：{ev.metadata.get('source')}")
                lines.append("")

        if graphrag_evidence:
            lines.append("=== GraphRAG 检索结果 ===")
            for i, ev in enumerate(graphrag_evidence, 1):
                content_preview = ev.content[:200] + "..." if len(ev.content) > 200 else ev.content
                lines.append(f"[GraphRAG-{i}] (分数：{ev.score:.3f})")
                lines.append(f"  {content_preview}")
                lines.append("")

        # 添加图谱路径（如果有）
        if evidence_chain.graph_paths:
            lines.append("=== 图谱路径 ===")
            for i, path in enumerate(evidence_chain.graph_paths, 1):
                lines.append(f"路径 {i}: {' -> '.join(path)}")
            lines.append("")

        return "\n".join(lines) if lines else "未找到相关证据"

    def _generate_answer(self, query: str, evidence_context: str) -> str:
        """
        生成最终答案

        Args:
            query: 用户查询
            evidence_context: 证据上下文

        Returns:
            答案文本
        """
        prompt = REPORTER_PROMPT.format(query=query, evidence_context=evidence_context)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer.strip()
        except Exception as e:
            logger.error(f"LLM 调用失败：{e}")
            return self._generate_fallback_answer(query, evidence_context)

    def _generate_fallback_answer(self, query: str, evidence_context: str) -> str:
        """
        生成降级答案（当 LLM 调用失败时）

        Args:
            query: 用户查询
            evidence_context: 证据上下文

        Returns:
            降级答案
        """
        return f"关于您的问题：{query}\n\n检索到的信息如下：\n\n{evidence_context}"


def get_reporter(llm=None) -> Reporter:
    """获取 Reporter 实例"""
    return Reporter(llm=llm)
