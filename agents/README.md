# Multi-Agent 智能对话系统

基于 Plan-Execute-Report 架构，实现复杂任务自动分解、并行检索与可解释推理（含证据链追踪）。

## 核心特性

- ✅ **统一入口** - 所有查询都走 Multi-Agent 流程，确保完整的证据链追踪
- ✅ **自动任务分解** - LLM 将复杂查询分解为可执行的子任务
- ✅ **简单查询优化** - 快速识别简单查询，直接返回单任务（不调用 LLM）
- ✅ **并行执行** - 依赖感知的并行任务执行
- ✅ **证据链追踪** - 完整记录推理过程中的所有证据
- ✅ **可解释推理** - 结构化证据 + 图谱路径展示
- ✅ **容错降级** - 自动重试 + GraphRAG→RAG 降级策略

## 架构设计

```
所有用户查询
    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent (统一入口)                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 1: Planner (任务规划)                               │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  简单查询快速识别（规则判断，不调用 LLM）            │  │  │
│  │  │  → 直接返回单任务 (rag 或 graphrag)                  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  复杂查询 LLM 分解                                    │  │  │
│  │  │  → 调用 LLM 分解为多任务                             │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 2: Executor (任务执行)                              │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  并行执行子任务（依赖感知）                         │  │  │
│  │  │  ┌─────────────┬─────────────────────┐            │  │  │
│  │  │  │ RAG Tool    │ GraphRAG Tool       │            │  │  │
│  │  │  │ (子任务 1)   │ (子任务 2)           │            │  │  │
│  │  │  └─────────────┴─────────────────────┘            │  │  │
│  │  │  • 记录证据链                                      │  │  │
│  │  │  • 支持重试和降级                                  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 3: Reporter (报告生成)                              │  │
│  │  • 整合所有证据                                           │  │
│  │  • 生成最终答案                                           │  │
│  │  • 构建结构化证据链                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
最终答案 + 证据链 + 推理步骤
```

## 目录结构

```
agents/
├── __init__.py           # 模块导出
├── models/               # 核心数据模型
│   ├── __init__.py
│   ├── task.py           # 任务定义 (Task, TaskType, TaskStatus)
│   ├── evidence.py       # 证据定义 (Evidence, EvidenceChain)
│   ├── plan.py           # 计划定义 (Plan, PlanStatus)
│   └── report.py         # 报告定义 (Report, ReportSection)
├── tools/                # 工具层
│   ├── __init__.py
│   ├── base.py           # 工具基类 (BaseTool, ToolResult)
│   ├── registry.py       # 工具注册表 (ToolRegistry)
│   ├── rag_tool.py       # RAG 工具包装
│   └── graphrag_tool.py  # GraphRAG 工具包装
├── planner.py            # Planner Agent (任务分解)
├── executor.py           # Executor Agent (并行执行)
├── reporter.py           # Reporter Agent (报告生成)
├── orchestrator.py       # Orchestrator (多 Agent 编排)
└── readme.md             # 本文档
```

## 核心组件

### 1. Orchestrator (编排器)

统一入口，所有查询都走 Multi-Agent 流程：

```python
from agents.orchestrator import get_orchestrator

orchestrator = get_orchestrator(
    rag_storage_path="./storage/rag_index",
    graphrag_storage_path="./storage/graphrag_index"
)

result = orchestrator.process_query("分析北京和上海的经济发展差异")
print(result.answer)           # 最终答案
print(result.plan)             # 任务计划
print(result.report)           # 完整报告（含证据链）
```

### 2. Planner (任务规划)

支持两种模式：

| 模式 | 触发条件 | 处理方式 |
|------|----------|----------|
| **快速识别** | 简单查询（<30 字，无复杂模式） | 规则判断，直接返回单任务（不调用 LLM） |
| **LLM 分解** | 复杂查询 | 调用 LLM 分解为多任务 |

```python
from agents.planner import Planner

planner = Planner()

# 简单查询：快速识别，返回单任务
result = planner.plan("什么是人工智能")
# → 1 个任务，使用 rag 工具

# 复杂查询：LLM 分解
result = planner.plan("分析北京和上海的经济发展差异")
# → 3 个任务，使用 graphrag 工具
```

### 3. Executor (任务执行)

依赖感知的并行任务执行，支持重试和降级：

```python
from agents.executor import Executor

# 配置：最大重试 1 次，启用降级策略
executor = Executor(
    max_parallel_workers=4,
    max_retries=1,           # 失败后重试次数
    enable_fallback=True     # 启用 graphrag -> rag 降级
)
result = executor.execute_parallel(plan)

# 查看任务执行详情
for r in result.task_results:
    print(f"{r.task.task_id}: 重试{r.retry_count}次，降级={r.fallback_used}")
```

#### 容错/降级策略

| 场景 | 处理 |
|------|------|
| 任务失败 | 自动重试（最多 `max_retries` 次） |
| GraphRAG 失败 | 降级为 RAG 继续执行 |
| 所有尝试失败 | 跳过任务，记录错误，继续其他任务 |

```
任务失败 → 重试 → 降级 (graphrag→rag) → 跳过
```

### 4. Reporter (报告生成)

整合执行结果，生成带证据链的最终答案：

```python
from agents.reporter import Reporter

reporter = Reporter()
result = reporter.generate(executor_result, plan)

if result.success:
    report = result.report
    print(report.final_answer)
    print(report.structured_evidence)  # 结构化证据
```

## 数据模型

### Task (任务)

```python
Task(
    task_id="task_001",
    task_type=TaskType.GRAPH_RAG,
    query="北京的经济情况",
    description="检索北京的经济信息",
    depends_on=["task_000"]  # 依赖的任务 ID
)
```

### Evidence (证据)

```python
Evidence(
    evidence_id="evidence_xxx",
    source=EvidenceSource.GRAPH_RAG,
    content="北京 GDP 超过 4 万亿元...",
    score=0.92,
    task_id="task_001"
)
```

### EvidenceChain (证据链)

```python
EvidenceChain(
    chain_id="chain_xxx",
    query="用户查询",
    evidence_list=[...],      # 所有证据
    reasoning_steps=[...],    # 推理步骤
    graph_paths=[...]         # 图谱路径
)
```

## 使用示例

### 简单查询

```python
from agents import get_orchestrator

orchestrator = get_orchestrator()

# 简单查询：快速识别，单任务执行
result = orchestrator.process_query("什么是人工智能")
print(result.answer)
print(result.report.structured_evidence)  # 有证据链！
```

### 复杂查询

```python
from agents import get_orchestrator

orchestrator = get_orchestrator()

# 复杂查询：LLM 分解，多任务并行执行
result = orchestrator.process_query("分析北京和上海的经济发展差异")

if result.success:
    print("答案:", result.answer)
    print("任务数:", len(result.plan.tasks))
    print("证据链:", result.report.structured_evidence)
```

## 测试

```bash
python test_agents.py
```

## 扩展

### 添加新工具

```python
from agents.tools import BaseTool, ToolResult, ToolRegistry

class MyCustomTool(BaseTool):
    def search(self, query: str, **kwargs) -> ToolResult:
        # 实现搜索逻辑
        return ToolResult(success=True, answer="...", evidence=[])
    
    def get_name(self) -> str:
        return "my_custom_tool"

# 注册工具
ToolRegistry.register("my_tool", MyCustomTool)
```

### 自定义 Planner 规则

```python
from agents.planner import Planner

# 调整复杂模式匹配阈值
planner = Planner(complex_threshold=3)
```

## 设计决策

### 为什么统一走 Multi-Agent？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Router 分流** | 简单查询快速 | 简单查询无证据链 |
| **统一 Multi-Agent** | 所有查询都有证据链 | 简单查询略慢 |

**选择统一 Multi-Agent 的原因：**
1. **功能完整性** - 所有查询都有证据链追踪
2. **架构简洁** - 单一入口，无分支
3. **可维护性** - 修改一处即可

**性能优化：** Planner 快速识别简单查询，不调用 LLM，延迟增加可忽略。
