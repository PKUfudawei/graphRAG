# GraphRAG — RAG + GraphRAG + Multi-Agent 检索增强生成

基于 **naive（向量）/ local（实体图）/ global（关系图）** 三模式检索，与 **Planner → Executor → Reporter** 多智能体编排的问答系统。支持复杂问题自动分解、依赖感知并行执行、实体消歧的知识图谱构建与证据链追踪。

> **注意**：本仓库是 `agentic-rag` 之外的独立实现（顶层模块 `agents/` `rag/` `graphrag/` `models/`），两者已分叉、**不可互通**。LLM / 建图 / 检索代码请勿跨仓库混用。

---

## 核心特性

- 🕸️ **三模式检索**（参考 LightRAG 分类）
  - `naive` — 纯向量两阶段召回（recall + 可选 rerank）
  - `local` — 实体检索 + 图 BFS 多跳遍历，返回实体关联的 chunk
  - `global` — 关系级 embedding 检索 + 跨文档 chunk 合成
- 🏗️ **知识图谱构建**
  - LLM 结构化抽取实体/关系（LightRAG 风格 schema，N-ary 拆二元）
  - **embedding 消歧**（`resolver.py`：bge-m3 + FAISS + Union-Find，相似度阈值 τ=0.9）
  - 增量更新（按文档 hash 跳过未变化文档）
- 🤖 **Multi-Agent 编排**：Planner 任务分解 → Executor 依赖感知并行 → Reporter 证据链报告
- 🔗 **证据链**：Task / Evidence / Plan / Report 全套数据模型，`structured_evidence` 结构化输出

---

## 项目结构

```
graphRAG/
├── agents/                  # 多智能体编排
│   ├── models/              # 数据模型
│   │   ├── task.py          # Task, TaskType(naive|local|global|deep_research), TaskStatus
│   │   ├── evidence.py      # Evidence, EvidenceSource, EvidenceChain
│   │   ├── plan.py          # Plan, PlanStatus
│   │   └── report.py        # Report, ReportSection
│   ├── tools/
│   │   ├── base.py          # BaseTool / ToolResult（抽象基类）
│   │   ├── registry.py      # ToolRegistry, SearchModeTool(mode=naive|local|global)
│   │   ├── rag_tool.py      # RAGTool（vector / bm25 / hybrid）
│   │   └── graphrag_tool.py # GraphRAGTool（naive / local / global）
│   ├── planner.py           # 任务规划
│   ├── executor.py          # 并行执行（重试；降级未实现）
│   ├── reporter.py          # 报告生成
│   └── orchestrator.py      # Orchestrator 统一入口
├── rag/
│   ├── indexer.py           # get_indexer(...).index_documents(docs) → (chunks, vectorstore)
│   └── retriever.py         # get_retriever(...)：vector_search / bm25_search / hybrid_search
├── graphrag/
│   ├── graph/
│   │   ├── builder.py       # 图构建（抽取→消歧→NetworkX DiGraph）
│   │   ├── extractor.py     # 实体/关系抽取（LightRAG 风格）
│   │   ├── resolver.py      # 实体消歧（Union-Find）
│   │   ├── community_detector.py    # ⚠️ 未接入主流程
│   │   └── community_summarizer.py  # ⚠️ 未接入主流程（当前含语法错误）
│   ├── indexer.py           # get_graphrag_indexer(...).index_documents(docs, database_path)
│   └── retriever.py         # naive_search / local_search / global_search
├── models/
│   ├── llm.py               # get_llm / get_json_llm
│   ├── embedding.py         # get_embedding（默认 BAAI/bge-m3, cpu）
│   ├── reranker.py          # cross-encoder reranker
│   └── chunker.py           # tiktoken 分块（默认 512/50，可截断）
└── scripts/                 # websearch 代理（实验性）
```

---

## 安装

```bash
# 推荐：使用 uv（依赖见 pyproject.toml）
uv sync

# 或 pip 安装本包
pip install -e .
```

- 需要 Python ≥ 3.10。
- 仓库**没有** `requirements.txt`（旧版 README 的 `pip install -r requirements.txt` 不适用）。
- 图社区检测相关依赖（`python-louvain` 等）已声明在 pyproject，但社区检测代码**未接入主流程**，非必需。

---

## 配置 LLM / Embedding

当前通过**代码传参**配置（**不支持环境变量覆盖**）：

```python
from models.llm import get_llm
from models.embedding import get_embedding

# LLM：默认指向本地 vLLM/Ollama 的 OpenAI 兼容端点，可替换为任意 OpenAI 兼容服务
llm = get_llm(
    model="Qwen/Qwen3.5-27B",
    base_url="http://localhost:8000/v1",   # 例如腾讯内网 tt-switch: http://127.0.0.1:15721/tencent/v1
    api_key="EMPTY",
    enable_thinking=False,
)

# Embedding：默认 BAAI/bge-m3，device 默认 cpu（无 GPU 环境不要传 cuda）
embedding = get_embedding(model="BAAI/bge-m3", device="cpu")
```

---

## 快速开始

### 1. 向量 RAG

```python
from rag.indexer import get_indexer
from rag.retriever import get_retriever
from models.chunker import get_chunker
from models.embedding import get_embedding
from langchain_core.documents import Document

documents = [Document(page_content="北京是中国的首都，常住人口超过两千万。")]

indexer = get_indexer(chunker=get_chunker(), embedding=get_embedding())
chunks, vectorstore = indexer.index_documents(documents)      # 返回 (chunks, vectorstore)

# 关闭 reranker（无 GPU / 不需要时显式传 None，否则默认加载到 reranker_device）
retriever = get_retriever(vectorstore, top_k=5, reranker_model=None)

retriever.vector_search("中国的首都是哪里？")                  # 纯向量
retriever.set_bm25_retriever(chunks)                           # 可选：BM25
retriever.hybrid_search("中国的首都是哪里？",
                        vector_weight=0.5, bm25_weight=0.5)    # 混合检索
```

> 注：`get_retriever` 默认 `reranker_model="BAAI/bge-reranker-v2-m3"`、`reranker_device="cuda:1"`。CPU 环境请显式传 `reranker_model=None`。

### 2. GraphRAG 建图与检索

```python
from graphrag.indexer import get_graphrag_indexer

indexer = get_graphrag_indexer()          # 默认 max_workers=16、thread 并发抽取
chunks, vectorstore, graph, entity_index, relationship_index = \
    indexer.index_documents(documents, database_path="./storage/graphrag_index")

# 产物（database_path 目录下）：
#   vectorstore/          向量库（naive 用）
#   graph.pkl             知识图谱
#   entities.pkl          {"index": FAISS, "metadata": [...]}   实体索引
#   relationships.pkl     {"index": FAISS, "metadata": [...]}   关系索引
```

检索可通过工具层（自动从 storage 目录加载上述产物）：

```python
from agents.tools import SearchModeTool

naive = SearchModeTool(mode="naive",  storage_path="./storage/graphrag_index")
local = SearchModeTool(mode="local",  storage_path="./storage/graphrag_index")

res = local.search("北京和上海在经济发展上有哪些联系？")
if res.success:
    for ev in res.evidence:
        print(ev.content)
```

或直接使用底层 retriever（naive / local / global 的真实签名）：

```python
from graphrag.retriever import get_graphrag_retriever

retriever = get_graphrag_retriever(
    graph=graph,
    entity_index=entity_index,          entity_metadata=<metadata list>,
    relationship_index=relationship_index, relationship_metadata=<metadata list>,
    embedding=embedding, vectorstore=vectorstore,
)
retriever.naive_search("...", top_k=5)
retriever.local_search("...", top_k_entities=3, max_hops=1, max_neighbors=3, max_chunks=10)
retriever.global_search("...", top_k_relationships=10, max_chunks=10)
```

**三种模式的语义：**

| 模式 | 机制 | 适合 |
|------|------|------|
| `naive` | 向量检索，文本 chunk 语义相似 | 事实/数值/定义类问题 |
| `local` | 实体向量检索 → BFS 多跳遍历邻居 → 取关联 chunk | 实体关系、特定命名实体（探测器、实验、机构） |
| `global` | 关系向量检索 → 跨文档 chunk 合成 | 跨文档综合、对比、主题概览 |

### 3. Multi-Agent 编排

统一入口（每个查询都产生完整证据链报告）：

```python
from agents import get_orchestrator

orchestrator = get_orchestrator()        # storage 参数见下方"注意"
result = orchestrator.process_query("分析北京和上海的经济发展差异，以及两地交通联系")

if result.success:
    print("答案:", result.answer)
    print("任务数:", len(result.plan.tasks))
    print("结构化证据:", result.report.structured_evidence)
else:
    print("错误:", result.error)
```

执行链路：`Planner.plan(query)` 分解为若干 `Task`（任务类型 ∈ `TaskType.{NAIVE, LOCAL, GLOBAL}`）→ `Executor.execute_parallel(plan)` 按依赖分层并行执行 → `Reporter.generate(executor_result, plan)` 汇总为带证据链的 `Report`。

**注意（当前实现的关键约束）：**
- Executor 通过 **`ToolRegistry`** 获取工具实例，**使用前需先把工具注册进注册表**。多模式场景按 `naive/local/global` 各注册一个 `SearchModeTool`：
  ```python
  from agents.tools import ToolRegistry, SearchModeTool
  ToolRegistry.register("naive",  SearchModeTool(mode="naive",  storage_path="./storage/graphrag_index"))
  ToolRegistry.register("local",  SearchModeTool(mode="local",  storage_path="./storage/graphrag_index"))
  ToolRegistry.register("global", SearchModeTool(mode="global", storage_path="./storage/graphrag_index"))
  ```
- `get_orchestrator(rag_storage_path=..., graphrag_storage_path=...)` 接收这两个参数，但**当前不会把它们下传给工具层**——工具层使用自身默认路径 `./storage/rag_index` / `./storage/graphrag_index`。因此请把索引构建到工具默认路径，或在注册时显式指定 `storage_path`。

---

## 数据模型

### Task / TaskType

```python
from agents.models import Task, TaskType

Task(
    task_id="task_001",
    task_type=TaskType.LOCAL,   # NAIVE | LOCAL | GLOBAL | DEEP_RESEARCH
    query="北京的经济情况",
    description="检索北京的经济信息",
    depends_on=["task_000"],    # 依赖任务 ID，决定执行层级
)
```

### Evidence / EvidenceChain

```python
from agents.models import Evidence, EvidenceSource

Evidence(
    evidence_id="evidence_001",
    source=EvidenceSource.LOCAL,   # NAIVE | LOCAL | GLOBAL
    content="北京 GDP 超过 4 万亿元……",
    score=0.92,
    task_id="task_001",
)
# EvidenceChain: chain_id + query + evidence_list + reasoning_steps + graph_paths
```

---

## 已知限制 / 当前状态

README 以上内容以当前代码为准；已知的坑与未完成部分如下，改动代码前请先确认：

1. **`global` 模式的工具层 bug**：`GraphRAGTool.global_search`（`agents/tools/graphrag_tool.py`）调用 `GraphRAGRetriever.global_search` 时多传了 `top_k_vectors` 参数，而 retriever 的 `global_search(query, top_k_relationships, max_chunks)` 不接受该参数 → **经工具层/编排走 global 会返回 `success=False`**。底层 `GraphRAGRetriever.global_search` 本身正常。
2. **无运行时降级**：`Executor._try_fallback` 目前恒返回 `False`，任务失败只做重试（`max_retries` 次），**不会** graphrag→rag 降级。
3. **社区检测/摘要未接线**：`graphrag/graph/community_detector.py`、`community_summarizer.py` 未接入 `index_documents` 主流程；`community_summarizer.py` 当前含语法错误，无法导入。
4. **LLM 不支持环境变量**：`models/llm.py` 的 `get_llm` 不会读取 `LLM_MODEL/LLM_BASE_URL/LLM_API_KEY`（如需环境变量可仿照 agentic-rag 改造）。
5. **Orchestrator 的 storage 参数是"死参数"**（见上文 Multi-Agent 注意）。
6. `scripts/` 只有实验性 websearch 代理，无命令行入口。

---

## 相关

- 设计上参考 **LightRAG**（三模式检索 + 抽取 schema）与 **GraphRAG** 式实体消歧思路。
- 完整评测脚本与归档见配套仓库 `agentic-rag`（`benchmarks/AI4EIC2023`）。

## 许可证

MIT
