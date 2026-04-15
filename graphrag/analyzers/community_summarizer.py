"""
Community summarization using LangChain.
使用 LangChain 进行社区总结。
"""
import heapq
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..models.graph_models import CommunitySummary


class CommunitySummarizer:
    """
    社区总结器。
    使用 LLM 为每个社区生成文本总结和嵌入。

    Args:
        llm: LangChain 语言模型实例
        embed_model: 嵌入模型，需有 encode 方法
        top_k: 提取前 k 个最重要的关系
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        embed_model,
        top_k: int = 100
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.top_k = top_k
        self._setup_prompt()

    def _setup_prompt(self):
        """设置社区总结的提示模板"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert knowledge graph summarizer."),
            ("human", """You are given relationships extracted from a knowledge graph community.

Each line represents a relation in the format:
- (weight) source -> relation -> target

The weight indicates how frequently the relation appears in the graph.
The relations are sorted by weight in descending order.

Top {top_k} most frequent relations in this community:

{relations}

Task:
Identify the main entities, themes, and types of relationships represented in this community.
Write a short paragraph summarizing the overall topic and the key connections between entities.
Prioritize patterns suggested by higher-weight relations.
""")
        ])

    @staticmethod
    def get_relations(subgraph: nx.Graph, top_k: int = 100) -> List[str]:
        """
        获取子图中最重要的关系。

        Args:
            subgraph: NetworkX 子图
            top_k: 返回前 k 个关系

        Returns:
            格式化的关系字符串列表
        """
        # 构建关系权重字典
        relation_weight_dict = {
            f"{source} -> {relation} -> {target}": int(edge_data.get('weight', 1))
            for source, target, relation, edge_data in subgraph.edges(keys=True, data=True)
        }

        # 获取 top_k 关系
        top_relations = heapq.nlargest(top_k, relation_weight_dict.items(), key=lambda x: x[1])
        relations = [f"- ({weight}) {relation}" for relation, weight in top_relations]

        return relations

    def summarize_community(self, subgraph: nx.Graph) -> Tuple[str, List[str]]:
        """
        总结单个社区。

        Args:
            subgraph: 社区子图

        Returns:
            (总结文本，关系列表)
        """
        relations = self.get_relations(subgraph, top_k=self.top_k)

        # 使用 LangChain 链
        chain = self.prompt | self.llm
        response = chain.invoke({
            "top_k": self.top_k,
            "relations": "\n".join(relations)
        })

        summary = response.content if hasattr(response, 'content') else str(response)
        return summary, relations

    def process_community(
        self,
        subgraph: nx.Graph,
        community_id: int
    ) -> CommunitySummary:
        """
        处理单个社区：生成总结和嵌入。

        Args:
            subgraph: 社区子图
            community_id: 社区 ID

        Returns:
            CommunitySummary 实例
        """
        entities = list(subgraph.nodes)
        summary, relations = self.summarize_community(subgraph)

        # 生成嵌入文本
        embedding_text = f"""
Summary:
{summary}

Entities:
{", ".join(entities)}

Relations:
{'\n'.join(relations)}
"""
        embedding = self.embed_model.encode(embedding_text)

        return CommunitySummary(
            community_id=community_id,
            nodes=entities,
            summary=summary,
            relations=relations,
            embedding=embedding
        )

    def summarize_communities(
        self,
        graph: nx.Graph,
        communities: Dict[int, List[str]],
        max_workers: int = 16
    ) -> List[CommunitySummary]:
        """
        并行总结所有社区。

        Args:
            graph: 完整图
            communities: 社区字典 {community_id: [节点列表]}
            max_workers: 最大并发数

        Returns:
            CommunitySummary 列表
        """
        community_summaries = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for community_id, comm_nodes in communities.items():
                subgraph = graph.subgraph(comm_nodes)
                futures.append(executor.submit(
                    self.process_community, subgraph, community_id
                ))

            for f in tqdm(
                as_completed(futures), total=len(futures),
                desc="Summarizing communities"
            ):
                community_summaries.append(f.result())

            # 按社区 ID 排序
            community_summaries.sort(key=lambda x: x.community_id)

        return community_summaries
