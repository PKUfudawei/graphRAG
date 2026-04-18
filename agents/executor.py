"""Executor Agent - 任务执行"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from agents.models.task import Task, TaskStatus, TaskType
from agents.models.plan import Plan
from agents.models.evidence import Evidence, EvidenceSource, EvidenceChain, generate_evidence_id
from agents.tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class TaskExecutionResult:
    """任务执行结果"""
    task: Task
    success: bool
    evidence: List[Evidence]
    answer: str
    error: Optional[str] = None
    retry_count: int = 0
    fallback_used: bool = False  # 是否使用了降级策略


class ExecutorResult:
    """Executor 执行结果"""
    def __init__(self, evidence_chain: EvidenceChain, task_results: List[TaskExecutionResult]):
        self.evidence_chain = evidence_chain
        self.task_results = task_results
        self.success = all(r.success for r in task_results)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "evidence_chain": self.evidence_chain.to_dict(),
            "task_results": [
                {
                    "task_id": r.task.task_id,
                    "success": r.success,
                    "evidence_count": len(r.evidence),
                    "error": r.error
                }
                for r in self.task_results
            ]
        }


class Executor:
    """任务执行器"""

    def __init__(
        self,
        max_parallel_workers: int = 4,
        max_retries: int = 1,
        enable_fallback: bool = True
    ):
        """
        初始化 Executor

        Args:
            max_parallel_workers: 最大并行工作线程数
            max_retries: 最大重试次数
            enable_fallback: 是否启用降级策略（graphrag -> rag）
        """
        self.max_parallel_workers = max_parallel_workers
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback

    def execute(self, plan: Plan) -> ExecutorResult:
        """
        执行计划中的所有任务

        Args:
            plan: 任务计划

        Returns:
            ExecutorResult 对象
        """
        evidence_chain = EvidenceChain(
            chain_id=plan.plan_id.replace("plan_", "chain_"),
            query=plan.original_query
        )

        task_results: List[TaskExecutionResult] = []

        # 按执行顺序执行任务
        execution_order = plan.execution_order if plan.execution_order else [t.task_id for t in plan.tasks]

        # 构建任务映射
        task_map = {t.task_id: t for t in plan.tasks}

        # 执行任务（支持依赖感知的并行执行）
        completed_tasks: Dict[str, bool] = {}  # task_id -> success

        for task_id in execution_order:
            task = task_map.get(task_id)
            if not task:
                logger.warning(f"任务 {task_id} 不存在，跳过")
                continue

            # 检查依赖
            if not self._check_dependencies(task, completed_tasks):
                logger.error(f"任务 {task_id} 的依赖未满足，跳过")
                task.status = TaskStatus.FAILED
                continue

            # 执行任务
            result = self._execute_single_task(task, evidence_chain)
            task_results.append(result)
            completed_tasks[task_id] = result.success

            # 更新任务状态
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED

        return ExecutorResult(evidence_chain=evidence_chain, task_results=task_results)

    def _check_dependencies(self, task: Task, completed_tasks: Dict[str, bool]) -> bool:
        """
        检查任务依赖是否满足

        Args:
            task: 当前任务
            completed_tasks: 已完成的任务映射

        Returns:
            依赖是否满足
        """
        for dep_id in task.depends_on:
            if dep_id not in completed_tasks or not completed_tasks[dep_id]:
                return False
        return True

    def _execute_single_task(self, task: Task, evidence_chain: EvidenceChain) -> TaskExecutionResult:
        """
        执行单个任务（带重试和降级）

        Args:
            task: 任务
            evidence_chain: 证据链

        Returns:
            TaskExecutionResult 对象
        """
        task.status = TaskStatus.RUNNING
        last_error: Optional[str] = None

        # 重试循环
        for retry in range(self.max_retries + 1):
            try:
                if retry > 0:
                    logger.info(f"任务 {task.task_id} 重试 {retry}/{self.max_retries}")

                result = self._execute_task_with_tool(task, evidence_chain)
                if result.success:
                    result.retry_count = retry
                    return result
                last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(f"任务 {task.task_id} 执行异常 (重试 {retry}): {e}")

            # 如果不是最后一次重试，尝试降级
            if retry < self.max_retries and self.enable_fallback:
                if self._try_fallback(task, evidence_chain, last_error):
                    # 降级成功，重新执行
                    continue

        # 所有重试和降级都失败
        logger.error(f"任务 {task.task_id} 最终失败：{last_error}")
        return TaskExecutionResult(
            task=task,
            success=False,
            evidence=[],
            answer="",
            error=last_error,
            retry_count=self.max_retries
        )

    def _execute_task_with_tool(self, task: Task, evidence_chain: EvidenceChain) -> TaskExecutionResult:
        """使用指定工具执行任务"""
        # 获取工具
        tool_class = ToolRegistry.get_tool(task.task_type.value)
        if not tool_class:
            raise ValueError(f"未找到工具：{task.task_type.value}")

        tool = tool_class()

        # 执行搜索
        payload = {"query": task.query, **task.parameters}
        result = tool.structured_search(payload)

        if not result.get("success"):
            raise ValueError(result.get("error", "工具执行失败"))

        # 转换为 Evidence
        evidence_list = []
        for item in result.get("retrieval_results", []):
            evidence = Evidence(
                evidence_id=generate_evidence_id(),
                source=EvidenceSource(task.task_type.value),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                task_id=task.task_id,
                metadata=item.get("metadata", {})
            )
            evidence_list.append(evidence)
            evidence_chain.add_evidence(evidence)

        # 添加推理步骤
        evidence_chain.add_reasoning_step(
            f"执行任务 {task.task_id} ({task.task_type.value}): {task.query}"
        )

        return TaskExecutionResult(
            task=task,
            success=True,
            evidence=evidence_list,
            answer=result.get("answer", "")
        )

    def _try_fallback(self, task: Task, evidence_chain: EvidenceChain, error: str) -> bool:
        """
        尝试降级策略

        Args:
            task: 原任务
            evidence_chain: 证据链
            error: 原错误信息

        Returns:
            是否降级成功（True 表示已降级，需要重新执行）
        """
        # graphrag -> rag 降级
        if task.task_type == TaskType.GRAPH_RAG:
            logger.info(f"任务 {task.task_id} 尝试降级：graphrag -> rag")
            task.task_type = TaskType.RAG
            evidence_chain.add_reasoning_step(
                f"任务 {task.task_id} 降级：graphrag 失败 ({error})，使用 rag 替代"
            )
            return True

        return False

    def execute_parallel(self, plan: Plan) -> ExecutorResult:
        """
        并行执行计划（依赖感知的并行）

        Args:
            plan: 任务计划

        Returns:
            ExecutorResult 对象
        """
        evidence_chain = EvidenceChain(
            chain_id=plan.plan_id.replace("plan_", "chain_"),
            query=plan.original_query
        )

        task_results: List[TaskExecutionResult] = []
        task_map = {t.task_id: t for t in plan.tasks}
        completed_tasks: Dict[str, TaskExecutionResult] = {}

        # 按批次执行（每批执行所有无依赖或依赖已满足的任务）
        pending_tasks = set(t.task_id for t in plan.tasks)

        while pending_tasks:
            # 找到可以执行的任务
            ready_tasks = []
            for task_id in list(pending_tasks):
                task = task_map[task_id]
                deps_satisfied = all(
                    dep_id in completed_tasks and completed_tasks[dep_id].success
                    for dep_id in task.depends_on
                )
                if deps_satisfied:
                    ready_tasks.append(task)

            if not ready_tasks:
                # 没有可执行的任务，可能是循环依赖或前置任务失败
                logger.warning("没有可执行的任务，剩余任务跳过")
                break

            # 并行执行就绪任务
            with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
                futures = {
                    executor.submit(self._execute_single_task, task, evidence_chain): task
                    for task in ready_tasks
                }

                for future in as_completed(futures):
                    result = future.result()
                    task_results.append(result)
                    completed_tasks[result.task.task_id] = result
                    pending_tasks.discard(result.task.task_id)

                    # 更新任务状态
                    task_map[result.task.task_id].status = (
                        TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                    )

        return ExecutorResult(evidence_chain=evidence_chain, task_results=task_results)


def get_executor(max_parallel_workers: int = 4) -> Executor:
    """获取 Executor 实例"""
    return Executor(max_parallel_workers=max_parallel_workers)
