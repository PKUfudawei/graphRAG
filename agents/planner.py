"""Planner Agent - 任务分解

使用 LLM 判断查询类型，决定是单任务还是多任务分解。
"""
import json
import logging
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

from agents.models.plan import Plan, PlanStatus, generate_plan_id
from agents.models.task import Task, TaskType, TaskStatus, generate_task_id

logger = logging.getLogger(__name__)


# Planner Prompt (LLM 统一判断单任务还是多任务)
PLANNER_PROMPT = """
你是一个任务规划专家。你的任务是将用户查询转换为可执行的检索任务。

## 可用工具

1. **rag** - 简单向量检索
   - 适用场景：事实性问题，如"什么是 X"、"X 在哪里"、"X 的定义"

2. **graphrag** - 图谱检索
   - 适用场景：关系性问题，如"X 和 Y 的关系"、"X 如何影响 Y"、"X 的上下游"

## 任务规划规则

1. **单任务场景**（简单查询）：
   - 事实性问题 → 使用 rag 工具
   - 关系性问题 → 使用 graphrag 工具

2. **多任务场景**（复杂查询）：
   - 涉及多个实体/概念的对比/分析 → 分解为多个子任务
   - 每个子任务聚焦一个具体的检索目标
   - 根据问题类型选择合适的工具
   - 如果子任务之间有依赖关系，在 depends_on 中指定

## 输出格式

返回一个 JSON 对象，格式如下：

```json
{{
    "plan_id": "plan_xxx",
    "original_query": "用户原始查询",
    "assumptions": ["假设 1", "假设 2"],
    "tasks": [
        {{
            "task_id": "task_xxx",
            "task_type": "rag" 或 "graphrag",
            "query": "具体查询",
            "description": "任务描述",
            "depends_on": []  // 依赖的任务 ID 列表
        }}
    ],
    "execution_order": ["task_xxx", "task_yyy"]  // 拓扑排序后的执行顺序
}}
```

## 单任务示例

用户查询："什么是人工智能"

输出：
```json
{{
    "plan_id": "plan_001",
    "original_query": "什么是人工智能",
    "assumptions": [],
    "tasks": [
        {{
            "task_id": "task_001",
            "task_type": "rag",
            "query": "什么是人工智能",
            "description": "检索人工智能的定义",
            "depends_on": []
        }}
    ],
    "execution_order": ["task_001"]
}}
```

## 多任务示例

用户查询："分析北京和上海的经济发展差异"

输出：
```json
{{
    "plan_id": "plan_001",
    "original_query": "分析北京和上海的经济发展差异",
    "assumptions": ["用户关注的是当前经济状况"],
    "tasks": [
        {{
            "task_id": "task_001",
            "task_type": "graphrag",
            "query": "北京的经济发展情况",
            "description": "检索北京经济发展相关信息",
            "depends_on": []
        }},
        {{
            "task_id": "task_002",
            "task_type": "graphrag",
            "query": "上海的经济发展情况",
            "description": "检索上海经济发展相关信息",
            "depends_on": []
        }},
        {{
            "task_id": "task_003",
            "task_type": "graphrag",
            "query": "北京和上海经济发展的对比",
            "description": "检索两地经济发展的对比分析",
            "depends_on": ["task_001", "task_002"]
        }}
    ],
    "execution_order": ["task_001", "task_002", "task_003"]
}}
```

## 当前查询

用户查询：{query}

请根据上述规则规划任务，只返回 JSON 对象，不要包含其他文字。
"""


class PlannerResult:
    """Planner 执行结果"""
    def __init__(self, plan: Optional[Plan] = None, error: Optional[str] = None):
        self.plan = plan
        self.error = error
        self.success = plan is not None and error is None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "plan": self.plan.to_dict() if self.plan else None,
            "error": self.error
        }


class Planner:
    """任务规划器

    使用 LLM 判断查询类型，决定是单任务还是多任务分解。
    """

    def __init__(self, llm=None):
        """
        初始化 Planner

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

    def plan(self, query: str) -> PlannerResult:
        """
        生成任务计划

        流程：调用 LLM 判断是单任务还是多任务分解

        Args:
            query: 用户查询

        Returns:
            PlannerResult 对象
        """
        try:
            return self._llm_plan(query)

        except Exception as e:
            logger.error(f"Planner 执行失败：{e}")
            return PlannerResult(error=str(e))

    def _llm_plan(self, query: str) -> PlannerResult:
        """
        调用 LLM 进行任务分解

        Args:
            query: 用户查询

        Returns:
            PlannerResult 对象
        """
        self._initialize_llm()

        # 构造 Prompt
        prompt = PLANNER_PROMPT.format(query=query)

        # 调用 LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, 'content') else str(response)

        # 解析 JSON
        plan_data = self._parse_json_response(content)

        # 构建 Plan 对象
        plan = self._build_plan(plan_data)

        return PlannerResult(plan=plan)

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON"""
        # 尝试提取 JSON 代码块
        content = content.strip()

        # 移除 ```json 和 ``` 标记
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败：{e}, 内容：{content[:200]}...")
            raise ValueError(f"无法解析 LLM 返回的 JSON: {e}")

    def _build_plan(self, data: Dict[str, Any]) -> Plan:
        """从字典构建 Plan 对象"""
        tasks = []
        for task_data in data.get("tasks", []):
            task = Task(
                task_id=task_data.get("task_id", generate_task_id()),
                task_type=TaskType(task_data.get("task_type", "rag")),
                query=task_data["query"],
                description=task_data.get("description"),
                depends_on=task_data.get("depends_on", []),
                status=TaskStatus.PENDING
            )
            tasks.append(task)

        # 如果没有任务，创建一个默认任务
        if not tasks:
            task = Task(
                task_id=generate_task_id(),
                task_type=TaskType.GRAPH_RAG,
                query=data.get("original_query", ""),
                description="默认检索任务",
                status=TaskStatus.PENDING
            )
            tasks.append(task)

        plan = Plan(
            plan_id=data.get("plan_id", generate_plan_id()),
            original_query=data.get("original_query", ""),
            tasks=tasks,
            status=PlanStatus.VALID,
            assumptions=data.get("assumptions", []),
            execution_order=data.get("execution_order", [t.task_id for t in tasks])
        )

        return plan


def get_planner(llm=None) -> Planner:
    """获取 Planner 实例"""
    return Planner(llm=llm)
