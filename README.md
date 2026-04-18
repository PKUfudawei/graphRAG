# GraphRAG - 智能检索增强生成系统

基于 RAG + GraphRAG + Multi-Agent 的智能对话系统，支持复杂任务自动分解、并行检索与可解释推理（含证据链追踪）。

## 核心特性

- 🔍 **RAG** - 向量检索 + 关键词检索 + 重排序
- 🕸️ **GraphRAG** - 实体检索 + 图谱多跳遍历 + 混合检索
- 🤖 **Multi-Agent** - Plan-Execute-Report 架构，自动任务分解
- ⚡ **并行执行** - 依赖感知的并行任务执行
- 🔗 **证据链追踪** - 完整记录推理过程中的所有证据
- 📊 **可解释推理** - 结构化证据 + 图谱路径展示
- 🛡️ **容错降级** - 自动重试 + GraphRAG→RAG 降级策略

## 项目结构

```
graphRAG/
├── agents/                 # Multi-Agent 编排系统
│   ├── models/            # 核心数据模型
│   │   ├── task.py        # 任务定义
│   │   ├── evidence.py    # 证据链定义
│   │   ├── plan.py        # 计划定义
│   │   └── report.py      # 报告定义
│   ├── tools/             # 工具层
│   │   ├── rag_tool.py    # RAG 工具包装
│   │   └── graphrag_tool.py # GraphRAG 工具包装
│   ├── planner.py         # 任务规划器
│   ├── executor.py        # 任务执行器
│   ├── reporter.py        # 报告生成器
│   └── orchestrator.py    # 编排器（统一入口）
├── rag/                   # RAG 模块
│   ├── indexer.py         # 文档索引器
│   └── retriever.py       # 检索器（向量+BM25+ 重排序）
├── graphrag/              # GraphRAG 模块
│   ├── graph/             # 图谱构建
│   │   ├── builder.py     # 图谱构建器
│   │   ├── extractor.py   # 实体关系提取
│   │   ├── community_detector.py # 社区检测
│   │   └── community_summarizer.py # 社区摘要
│   ├── indexer.py         # 图谱索引器
│   └── retriever.py       # 图谱检索器（向量 + 实体 + 多跳）
├── models/                # 基础模型
│   ├── llm.py             # LLM 封装
│   ├── embedding.py       # 嵌入模型
│   └── reranker.py        # 重排序模型
├── datasets/              # 数据集
├── storage/               # 索引存储
└── test_agents.py         # Agent 测试
```

## 快速开始

### 安装依赖

```bash
# 使用 uv 安装
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 基础 RAG 使用

```python
from rag.indexer import get_indexer
from rag.retriever import get_retriever
from models.embedding import get_embedding
from models.chunker import get_chunker

# 1. 创建索引
chunker = get_chunker(model="cl100k_base", chunk_size=500, overlap=100)
embedding = get_embedding(model="BAAI/bge-m3", device="cuda:0")
indexer = get_indexer(chunker=chunker, embedding=embedding)

# 加载文档并索引
from langchain_core.documents import Document
documents = [Document(page_content="北京是中国的首都")]
chunks = indexer.index_documents(documents)
vectorstore = indexer.build_vectorstore(chunks)

# 2. 检索
retriever = get_retriever(vectorstore, top_k=5)
results = retriever.retrieve("中国的首都是哪里？")
for doc in results:
    print(doc.page_content)
```

### GraphRAG 使用

```python
from graphrag.indexer import get_graphrag_indexer
from graphrag.retriever import get_graphrag_retriever

# 1. 构建图谱索引
indexer = get_graphrag_indexer(
    storage_path="./storage/graphrag_index",
    chunk_size=500,
    device="cuda:0"
)

# 加载文档并构建图谱
documents = [...]  # 你的文档列表
indexer.index_documents(documents)

# 2. 图谱检索
retriever = indexer.get_retriever()
results = retriever.retrieve(
    query="北京和上海的关系",
    top_k_entities=3,
    max_hops=2
)
for doc in results:
    print(doc.page_content)
```

### Multi-Agent 使用（推荐）

统一入口，所有查询都有完整的证据链追踪：

```python
from agents import get_orchestrator

# 创建编排器
orchestrator = get_orchestrator(
    rag_storage_path="./storage/rag_index",
    graphrag_storage_path="./storage/graphrag_index"
)

# 处理查询
result = orchestrator.process_query("分析北京和上海的经济发展差异")

if result.success:
    print("答案:", result.answer)
    print("任务数:", len(result.plan.tasks))
    print("证据链:", result.report.structured_evidence)
else:
    print("错误:", result.error)
```

## 核心模块详解

### 1. RAG 模块

支持向量检索、BM25 关键词检索和重排序：

```python
from rag.retriever import get_retriever

retriever = get_retriever(
    vectorstore=vectorstore,
    top_k=5,
    reranker_model="BAAI/bge-reranker-v2-m3",  # 可选重排序
    reranker_device="cuda:0"
)

# 向量检索
results = retriever.retrieve("查询文本")

# 混合检索（向量+BM25）
retriever.set_bm25_retriever(documents)  # 先设置 BM25 索引
results = retriever.hybrid_search("查询文本", vector_weight=0.5, bm25_weight=0.5)
```

### 2. GraphRAG 模块

支持实体检索和多跳图谱遍历：

```python
from graphrag.retriever import get_graphrag_retriever

retriever = get_graphrag_retriever(
    graph=graph,
    entity_index=entity_index,
    entity_metadata=entity_metadata,
    embedding=embedding,
    vectorstore=vectorstore
)

# 实体检索
entities = retriever.search_entities("中国的首都", top_k=3)

# 多跳遍历
graph_doc = retriever.traverse_multi_hop(["北京市"], max_hops=2)

# 混合检索（向量 + 图谱）
results = retriever.retrieve(
    "中国的首都是哪里？",
    top_k_vectors=5,
    top_k_entities=3,
    max_hops=2,
    vector_weight=0.5,
    graph_weight=0.5
)
```

### 3. Multi-Agent 模块

基于 Plan-Execute-Report 架构：

```
用户查询
    ↓
┌─────────────────────────────────────────┐
│         Orchestrator (编排器)            │
│                                          │
│  Planner → Executor → Reporter          │
│  (规划)   (执行)     (报告)              │
│                                          │
└─────────────────────────────────────────┘
    ↓
最终答案 + 证据链
```

#### Planner（任务规划）

LLM 判断查询类型，决定单任务还是多任务分解：

```python
from agents import Planner

planner = Planner()

# 简单查询 → 单任务
result = planner.plan("什么是人工智能")
# → 1 个任务，使用 rag 工具

# 复杂查询 → 多任务
result = planner.plan("分析北京和上海的经济发展差异")
# → 3 个任务，使用 graphrag 工具
```

#### Executor（任务执行）

依赖感知的并行执行，支持重试和降级：

```python
from agents import Executor

executor = Executor(
    max_parallel_workers=4,  # 最大并行线程数
    max_retries=1,            # 失败重试次数
    enable_fallback=True      # 启用 graphrag→rag 降级
)

result = executor.execute_parallel(plan)

# 查看执行详情
for r in result.task_results:
    print(f"{r.task.task_id}: 重试{r.retry_count}次，降级={r.fallback_used}")
```

#### Reporter（报告生成）

整合所有证据，生成带证据链的最终答案：

```python
from agents import Reporter

reporter = Reporter()
result = reporter.generate(executor_result, plan)

if result.success:
    report = result.report
    print(report.final_answer)
    print(report.structured_evidence)  # 结构化证据
```

## 数据模型

### Task（任务）

```python
Task(
    task_id="task_001",
    task_type=TaskType.GRAPH_RAG,  # rag | graphrag
    query="北京的经济情况",
    description="检索北京的经济信息",
    depends_on=["task_000"]  # 依赖的任务 ID
)
```

### Evidence（证据）

```python
Evidence(
    evidence_id="evidence_xxx",
    source=EvidenceSource.GRAPH_RAG,  # rag | graph_rag
    content="北京 GDP 超过 4 万亿元...",
    score=0.92,
    task_id="task_001"
)
```

### EvidenceChain（证据链）

```python
EvidenceChain(
    chain_id="chain_xxx",
    query="用户查询",
    evidence_list=[...],      # 所有证据
    reasoning_steps=[...],    # 推理步骤
    graph_paths=[...]         # 图谱路径
)
```

## 配置说明

### LLM 配置

编辑 `models/llm.py` 或设置环境变量：

```python
from models.llm import get_llm

llm = get_llm(
    model="Qwen/Qwen3.5-27B",
    base_url="http://localhost:8000/v1",  # vLLM/Ollama 地址
    api_key="EMPTY"
)
```

### Embedding 配置

```python
from models.embedding import get_embedding

embedding = get_embedding(
    model="BAAI/bge-m3",  # 或 "sentence-transformers/all-MiniLM-L6-v2"
    device="cuda:0"
)
```

## 测试

```bash
# 测试 Agent 模块
python test_agents.py

# 测试 RAG 模块
python -m rag.retriever

# 测试 GraphRAG 模块
python -m graphrag.retriever
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

**性能优化：** Planner 使用 LLM 判断，简单查询返回单任务，延迟增加可忽略。

### 容错/降级策略

```
任务失败 → 重试 → 降级 (graphrag→rag) → 跳过
```

| 场景 | 处理 |
|------|------|
| 任务失败 | 自动重试（最多 `max_retries` 次） |
| GraphRAG 失败 | 降级为 RAG 继续执行 |
| 所有尝试失败 | 跳过任务，记录错误，继续其他任务 |

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

## 许可证

MIT

## 贡献

欢迎提交 Issue 和 Pull Request！
