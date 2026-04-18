"""
Multi-Agent 智能对话系统

基于 Plan-Execute-Report 架构，实现复杂任务自动分解、并行检索与可解释推理。
"""

from agents.planner import Planner, PlannerResult, get_planner
from agents.executor import Executor, ExecutorResult, TaskExecutionResult, get_executor
from agents.reporter import Reporter, ReporterResult, get_reporter
from agents.orchestrator import Orchestrator, OrchestratorResult, get_orchestrator
from agents.tools import ToolRegistry, BaseTool, ToolResult, RAGTool, GraphRAGTool

# 核心数据模型
from agents.models import (
    Task, TaskType, TaskStatus, generate_task_id,
    Evidence, EvidenceSource, EvidenceChain, generate_evidence_id, generate_chain_id,
    Plan, PlanStatus, generate_plan_id,
    Report, ReportSection, generate_report_id
)

__version__ = "0.1.0"

__all__ = [
    # Planner
    "Planner", "PlannerResult", "get_planner",
    # Executor
    "Executor", "ExecutorResult", "TaskExecutionResult", "get_executor",
    # Reporter
    "Reporter", "ReporterResult", "get_reporter",
    # Orchestrator
    "Orchestrator", "OrchestratorResult", "get_orchestrator",
    # Tools
    "ToolRegistry", "BaseTool", "ToolResult", "RAGTool", "GraphRAGTool",
    # Models
    "Task", "TaskType", "TaskStatus", "generate_task_id",
    "Evidence", "EvidenceSource", "EvidenceChain", "generate_evidence_id", "generate_chain_id",
    "Plan", "PlanStatus", "generate_plan_id",
    "Report", "ReportSection", "generate_report_id",
]
