"""Agents 模块测试"""
import sys
import os

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def test_planner():
    """测试 Planner"""
    print("\n=== 测试 Planner ===")
    from agents.planner import Planner, TaskType

    planner = Planner()

    # 测试 1: 简单查询（需要 LLM）
    print("\n[测试 1] 简单查询 LLM 判断")
    query = "什么是人工智能"
    result = planner.plan(query)

    if result.success:
        plan = result.plan
        print(f"  查询：{query}")
        print(f"  任务数：{len(plan.tasks)}")
        print(f"  工具类型：{plan.tasks[0].task_type.value}")
        print("  ✓ LLM 判断成功")
    else:
        print(f"  LLM 判断失败（预期：缺少 langchain_openai）：{result.error}")

    # 测试 2: 关系性问题（需要 LLM）
    print("\n[测试 2] 关系性问题 LLM 判断")
    query = "北京和上海的关系"
    result = planner.plan(query)

    if result.success:
        plan = result.plan
        print(f"  查询：{query}")
        print(f"  任务数：{len(plan.tasks)}")
        print(f"  工具类型：{plan.tasks[0].task_type.value}")
        print("  ✓ LLM 判断成功")
    else:
        print(f"  LLM 判断失败（预期：缺少 langchain_openai）：{result.error}")

    # 测试 3: 复杂查询（需要 LLM 分解）
    print("\n[测试 3] 复杂查询 LLM 分解")
    query = "分析北京和上海的经济发展差异"
    result = planner.plan(query)

    if result.success:
        plan = result.plan
        print(f"  查询：{query}")
        print(f"  计划 ID: {plan.plan_id}")
        print(f"  任务数：{len(plan.tasks)}")
        for task in plan.tasks:
            print(f"    - {task.task_id}: {task.task_type.value} - {task.query}")
        print(f"  执行顺序：{plan.execution_order}")
        print("  ✓ LLM 分解成功")
    else:
        print(f"  LLM 分解失败（预期：缺少 langchain_openai）：{result.error}")

    print("\n✓ Planner 测试完成")


def test_executor():
    """测试 Executor"""
    print("\n=== 测试 Executor ===")
    from agents.models.plan import Plan, PlanStatus
    from agents.models.task import Task, TaskType, TaskStatus
    from agents.executor import Executor

    # 创建测试计划
    tasks = [
        Task(
            task_id="task_001",
            task_type=TaskType.RAG,
            query="什么是北京",
            description="检索北京的基本信息",
            depends_on=[]
        ),
        Task(
            task_id="task_002",
            task_type=TaskType.GRAPH_RAG,
            query="北京的经济情况",
            description="检索北京的经济信息",
            depends_on=["task_001"]
        )
    ]

    plan = Plan(
        plan_id="plan_test_001",
        original_query="测试查询",
        tasks=tasks,
        status=PlanStatus.VALID,
        execution_order=["task_001", "task_002"]
    )

    executor = Executor()
    result = executor.execute(plan)

    print(f"执行结果：{result.success}")
    print(f"证据链 ID: {result.evidence_chain.chain_id}")
    print(f"证据数量：{len(result.evidence_chain.evidence_list)}")
    for r in result.task_results:
        print(f"  - {r.task.task_id}: success={r.success}, evidence={len(r.evidence)}")

    print("\n✓ Executor 测试完成")


def test_orchestrator():
    """测试 Orchestrator"""
    print("\n=== 测试 Orchestrator ===")
    from agents.orchestrator import get_orchestrator

    orchestrator = get_orchestrator()

    # 测试简单查询
    query = "什么是人工智能"
    print(f"\n查询：{query}")
    result = orchestrator.process_query(query)
    print(f"成功：{result.success}")
    print(f"类型：{result.query_type}")
    if result.success:
        print(f"答案前 100 字：{result.answer[:100]}...")
    else:
        print(f"错误：{result.error}")

    print("\n✓ Orchestrator 测试完成")


def test_evidence_chain():
    """测试证据链"""
    print("\n=== 测试 EvidenceChain ===")
    from agents.models.evidence import EvidenceChain, Evidence, EvidenceSource, generate_evidence_id

    chain = EvidenceChain(chain_id="chain_test_001", query="测试查询")

    # 添加证据
    evidence1 = Evidence(
        evidence_id=generate_evidence_id(),
        source=EvidenceSource.RAG,
        content="这是 RAG 检索到的证据",
        score=0.9,
        task_id="task_001"
    )
    evidence2 = Evidence(
        evidence_id=generate_evidence_id(),
        source=EvidenceSource.GRAPH_RAG,
        content="这是 GraphRAG 检索到的证据",
        score=0.85,
        task_id="task_002"
    )

    chain.add_evidence(evidence1)
    chain.add_evidence(evidence2)
    chain.add_reasoning_step("步骤 1: 使用 RAG 检索基本信息")
    chain.add_reasoning_step("步骤 2: 使用 GraphRAG 检索关系信息")
    chain.add_graph_path(["北京", "位于", "华北平原"])

    print(f"证据链 ID: {chain.chain_id}")
    print(f"证据数量：{len(chain.evidence_list)}")
    print(f"推理步骤：{len(chain.reasoning_steps)}")
    print(f"图谱路径：{len(chain.graph_paths)}")

    # 测试按任务获取证据
    task_evidence = chain.get_evidence_by_task("task_001")
    print(f"task_001 的证据：{len(task_evidence)}")

    # 测试按来源获取证据
    rag_evidence = chain.get_evidence_by_source("rag")
    print(f"RAG 来源的证据：{len(rag_evidence)}")

    print("\n✓ EvidenceChain 测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Agents 模块测试")
    print("=" * 60)

    # 测试数据模型
    test_evidence_chain()

    # 测试组件
    test_planner()

    # 以下测试需要索引数据
    # test_executor()
    # test_orchestrator()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
