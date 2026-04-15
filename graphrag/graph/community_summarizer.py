"""Community summarization using LangChain."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List

import heapq
import numpy as np
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs.graph_document import GraphDocument
from tqdm import tqdm

try:
    from ..rag.llm import get_llm
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    rag_path = os.path.join(base_dir, "rag")
    sys.path.insert(0, rag_path)
    from llm import get_llm


@dataclass
class CommunitySummary:
    """Summary of a detected community.

    Attributes:
        community_id: Community unique identifier.
        nodes: List of node IDs in this community.
        summary: Text summary of the community.
        relations: List of important relations in the community.
        embedding: Vector embedding of the community summary.
    """
    community_id: int
    nodes: List[str]
    summary: str
    relations: List[str]
    embedding: np.ndarray

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding embedding).

        Returns:
            Dictionary representation.
        """
        return {
            "community_id": self.community_id,
            "nodes": self.nodes,
            "summary": self.summary,
            "relations": self.relations
        }


class CommunitySummarizer:
    """Summarize communities using LLM.

    Args:
        llm: LangChain language model instance.
        embed_model: Embedding model with encode method.
        top_k: Number of top relations to extract.
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

    def _setup_prompt(self) -> None:
        """Set up the community summarization prompt template."""
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

    def summarize_communities(
        self,
        community_subgraphs: Dict[int, GraphDocument],
        max_workers: int = 16
    ) -> List[CommunitySummary]:
        """Parallel summarize all communities.

        Args:
            community_subgraphs: Dictionary mapping community_id to GraphDocument.
            max_workers: Maximum number of concurrent workers.

        Returns:
            List of CommunitySummary instances.
        """
        def process_community(community_id: int, subgraph: GraphDocument) -> CommunitySummary:
            # Build relation weight dictionary
            relation_weight_dict = {
                f"{rel.source.id} -> {rel.type} -> {rel.target.id}":
                rel.properties.get("weight", 1)
                for rel in subgraph.relationships
            }

            # Get top_k relations
            top_relations = heapq.nlargest(
                self.top_k, relation_weight_dict.items(), key=lambda x: x[1]
            )
            relations = [f"- ({weight}) {relation}" for relation, weight in top_relations]

            # Use LangChain chain
            chain = self.prompt | self.llm
            response = chain.invoke({
                "top_k": self.top_k,
                "relations": "\n".join(relations)
            })
            summary = response.content if hasattr(response, 'content') else str(response)

            entities = [str(node.id) for node in subgraph.nodes]

            # Generate embedding text
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

        community_summaries = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_community, comm_id, subgraph)
                for comm_id, subgraph in community_subgraphs.items()
            ]

            for f in tqdm(
                as_completed(futures), total=len(futures),
                desc="Summarizing communities"
            ):
                community_summaries.append(f.result())

        community_summaries.sort(key=lambda x: x.community_id)
        return community_summaries


if __name__ == "__main__":
    """Test cases for CommunitySummarizer."""
    from langchain_community.graphs.graph_document import Node, Relationship
    from langchain_core.documents import Document

    # Test 1: CommunitySummary dataclass
    print("Test 1: CommunitySummary dataclass")
    test_summary = CommunitySummary(
        community_id=0,
        nodes=["node1", "node2"],
        summary="Test summary",
        relations=["rel1", "rel2"],
        embedding=np.array([0.1, 0.2, 0.3])
    )
    print(f"  community_id: {test_summary.community_id}")
    print(f"  nodes: {test_summary.nodes}")
    print(f"  to_dict: {test_summary.to_dict()}")
    assert test_summary.community_id == 0
    assert "embedding" not in test_summary.to_dict()
    print("  PASSED")

    # Test 2: Import and initialize with real LLM and embedder
    print("Test 2: Import get_llm and get_embedder")
    llm = get_llm()
    print(f"  LLM loaded: {type(llm).__name__}")

    # Import embedder
    import sys
    import os
    rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "rag"))
    sys.path.insert(0, rag_path)
    from index.embedder import get_embedder

    embed_model = get_embedder(model="BAAI/bge-small-zh-v1.5", device="cpu")
    print(f"  Embedder loaded: {type(embed_model).__name__}")
    print("  PASSED")

    # Test 3: CommunitySummarizer initialization
    print("Test 3: CommunitySummarizer initialization")
    summarizer = CommunitySummarizer(llm=llm, embed_model=embed_model, top_k=10)
    print(f"  top_k: {summarizer.top_k}")
    assert summarizer.top_k == 10
    print("  PASSED")

    # Test 4: Summarize single community
    print("Test 4: Summarize single community")
    nodes = [
        Node(id="北京", type="city"),
        Node(id="中国", type="country"),
    ]
    relationships = [
        Relationship(source=nodes[0], target=nodes[1], type="位于", properties={"weight": 5}),
    ]
    graph_doc = GraphDocument(
        nodes=nodes,
        relationships=relationships,
        source=Document(page_content="test")
    )
    community_subgraphs = {0: graph_doc}

    summaries = summarizer.summarize_communities(community_subgraphs, max_workers=1)
    print(f"  Generated {len(summaries)} summaries")
    assert len(summaries) == 1
    assert summaries[0].community_id == 0
    assert len(summaries[0].nodes) == 2
    print(f"  Summary preview: {summaries[0].summary[:80]}...")
    print("  PASSED")

    print("\nAll tests passed!")
