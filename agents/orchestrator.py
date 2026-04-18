"""Orchestrator - 多 Agent 编排器

统一入口：所有查询都走 Planner → Executor → Reporter 流程
确保每个查询都有完整的证据链追踪和可解释推理。
"""
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from agents.planner import Planner, PlannerResult
from agents.executor import Executor, ExecutorResult
from agents.reporter import Reporter, ReporterResult
from agents.models.plan import Plan
from agents.models.report import Report

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """编排结果"""
    success: bool
    query: str
    answer: str
    report: Optional[Report] = None
    plan: Optional[Plan] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "query": self.query,
            "answer": self.answer,
            "report": self.report.to_dict() if self.report else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "error": self.error
        }


class Orchestrator:
    """多 Agent 编排器

    所有查询统一走 Planner → Executor → Reporter 流程，
    确保每个查询都有完整的证据链追踪和可解释推理。
    """

    def __init__(
        self,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        reporter: Optional[Reporter] = None,
        rag_storage_path: str = "./storage/rag_index",
        graphrag_storage_path: str = "./storage/graphrag_index"
    ):
        """
        初始化编排器

        Args:
            planner: 任务规划器
            executor: 任务执行器
            reporter: 报告生成器
            rag_storage_path: RAG 索引存储路径
            graphrag_storage_path: GraphRAG 索引存储路径
        """
        self.planner = planner or Planner()
        self.executor = executor or Executor()
        self.reporter = reporter or Reporter()
        self.rag_storage_path = rag_storage_path
        self.graphrag_storage_path = graphrag_storage_path

    def process_query(self, query: str) -> OrchestratorResult:
        """
        处理用户查询（统一走 Multi-Agent 流程）

        流程:
        1. Planner: 任务分解（简单查询快速返回单任务，复杂查询调用 LLM 分解）
        2. Executor: 并行执行子任务，调用 RAG/GraphRAG 工具
        3. Reporter: 整合结果，生成带证据链的最终答案

        Args:
            query: 用户查询

        Returns:
            OrchestratorResult 对象
        """
        logger.info(f"处理查询：{query}")

        try:
            # Step 1: 规划
            logger.info("Step 1: 任务规划...")
            planner_result = self.planner.plan(query)

            if not planner_result.success:
                raise ValueError(f"规划失败：{planner_result.error}")

            plan = planner_result.plan
            logger.info(f"规划完成，共 {len(plan.tasks)} 个任务")

            # Step 2: 执行
            logger.info("Step 2: 任务执行...")
            executor_result = self.executor.execute_parallel(plan)
            logger.info(f"执行完成，成功：{executor_result.success}")

            # Step 3: 报告
            logger.info("Step 3: 生成报告...")
            reporter_result = self.reporter.generate(executor_result, plan)

            if not reporter_result.success:
                raise ValueError(f"报告生成失败：{reporter_result.error}")

            report = reporter_result.report

            return OrchestratorResult(
                success=True,
                query=query,
                answer=report.final_answer,
                report=report,
                plan=plan
            )

        except Exception as e:
            logger.error(f"处理查询失败：{e}")
            return OrchestratorResult(
                success=False,
                query=query,
                answer="",
                error=str(e)
            )


def get_orchestrator(
    rag_storage_path: str = "./storage/rag_index",
    graphrag_storage_path: str = "./storage/graphrag_index"
) -> Orchestrator:
    """获取编排器实例"""
    return Orchestrator(
        rag_storage_path=rag_storage_path,
        graphrag_storage_path=graphrag_storage_path
    )
