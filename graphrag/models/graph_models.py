"""
Data models for knowledge graph.
使用 Pydantic 定义图谱数据结构，与 LangChain GraphDocument 兼容。
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class GraphNode(BaseModel):
    """图节点模型"""
    id: str = Field(..., description="节点唯一标识")
    type: str = Field(default="entity", description="节点类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="节点属性")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            **self.properties
        }


class GraphEdge(BaseModel):
    """图边模型"""
    source: str = Field(..., description="源节点 ID")
    target: str = Field(..., description="目标节点 ID")
    relation: str = Field(..., description="关系类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="边属性")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            **self.properties
        }


class GraphDocumentWrapper(BaseModel):
    """
    LangChain GraphDocument 包装器
    用于表示从单个文档/文本块提取的知识图谱
    """
    nodes: List[GraphNode] = Field(default_factory=list, description="节点列表")
    edges: List[GraphEdge] = Field(default_factory=list, description="边列表")
    source: Optional[str] = Field(default=None, description="源文本或文档标识")

    @classmethod
    def from_extraction_result(cls, result: Dict[str, Any], source: Optional[str] = None) -> "GraphDocumentWrapper":
        """从 LLM 提取结果创建 GraphDocumentWrapper"""
        nodes = []
        for n in result.get("nodes", []):
            if isinstance(n, dict) and "name" in n:
                nodes.append(GraphNode(
                    id=n["name"].strip().lower(),
                    type=n.get("type", "unknown"),
                    properties={}
                ))

        edges = []
        for e in result.get("edges", []):
            if isinstance(e, dict) and "source" in e and "target" in e and "relation" in e:
                edges.append(GraphEdge(
                    source=e["source"].strip().lower(),
                    target=e["target"].strip().lower(),
                    relation=e.get("relation", "related with").strip().lower(),
                    properties={}
                ))

        return cls(nodes=nodes, edges=edges, source=source)


class CommunitySummary(BaseModel):
    """社区总结模型"""
    community_id: int = Field(..., description="社区 ID")
    nodes: List[str] = Field(default_factory=list, description="社区包含的节点列表")
    summary: str = Field(default="", description="社区文本总结")
    relations: List[str] = Field(default_factory=list, description="社区内主要关系")
    embedding: Optional[Any] = Field(default=None, description="社区嵌入向量")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "community_id": self.community_id,
            "nodes": self.nodes,
            "summary": self.summary,
            "relations": self.relations
        }
