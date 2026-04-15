"""Graph builder for building and storing knowledge graphs.

This module provides utilities for building NetworkX graphs from
LangChain GraphDocument and storing them to Neo4j.
"""
import os
import sys
from typing import List, Optional

import networkx as nx
from langchain_community.graphs.graph_document import (
    GraphDocument,
    Node as LCNode,
    Relationship as LCRelationship
)
from tqdm import tqdm

# 支持直接运行和模块导入
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from graph_storage import Neo4jStorage, get_neo4j_storage


class GraphBuilder:
    """Build and store knowledge graphs from LangChain GraphDocument.

    Converts LangChain GraphDocument to NetworkX MultiDiGraph and
    optionally stores to Neo4j.

    Args:
        neo4j_storage: Optional Neo4jStorage instance for persisting graphs.
    """

    def __init__(
        self,
        neo4j_storage: Optional[Neo4jStorage] = None
    ):
        self.graph = nx.MultiDiGraph()
        self.neo4j_storage = neo4j_storage

    def add_document(self, doc: GraphDocument) -> "GraphBuilder":
        """Add a single GraphDocument to the graph.

        Args:
            doc: LangChain GraphDocument to add.

        Returns:
            self for method chaining.
        """
        # Add nodes
        for node in doc.nodes:
            node_id = str(node.id)
            if node_id not in self.graph:
                self.graph.add_node(
                    node_id,
                    node_type=node.type or "Entity",
                    weight=1,
                    **(node.properties or {})
                )
            else:
                self.graph.nodes[node_id]['weight'] += 1

        # Add relationships
        for rel in doc.relationships:
            source_id = str(rel.source.id)
            target_id = str(rel.target.id)

            # Skip self-loops
            if source_id == target_id:
                continue

            self._add_relationship(source_id, target_id, rel)

        return self

    def _add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel: LCRelationship
    ) -> None:
        """Add or update a relationship in the graph.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            rel: Relationship to add.
        """
        # Ensure nodes exist
        for node_id, node_type in [(source_id, "Entity"), (target_id, "Entity")]:
            if node_id not in self.graph:
                self.graph.add_node(node_id, node_type=node_type, weight=1)

        rel_type = rel.type or "RELATED_TO"

        # Check if relationship already exists
        if self.graph.has_edge(source_id, target_id):
            for key, data in self.graph[source_id][target_id].items():
                if key == rel_type:
                    data['weight'] += 1
                    return

        # Add new relationship
        self.graph.add_edge(
            source_id,
            target_id,
            key=rel_type,
            weight=1,
            **(rel.properties or {})
        )

    def add_documents(self, documents: List[GraphDocument]) -> "GraphBuilder":
        """Add multiple GraphDocuments to the graph.

        Args:
            documents: List of GraphDocuments to add.

        Returns:
            self for method chaining.
        """
        for doc in tqdm(documents, desc="Building graph"):
            self.add_document(doc)
        return self

    def build(self, documents: List[GraphDocument]) -> nx.MultiDiGraph:
        """Build graph from a list of GraphDocuments.

        Args:
            documents: List of GraphDocuments to build from.

        Returns:
            The built NetworkX MultiDiGraph.
        """
        self.add_documents(documents)
        return self.graph

    def save_to_neo4j(self) -> None:
        """Save the graph to Neo4j.

        Requires neo4j_storage to be set.
        """
        if self.neo4j_storage is None:
            raise ValueError("Neo4jStorage not configured")

        # Convert NetworkX graph to GraphDocument
        graph_doc = self._to_graph_document()
        self.neo4j_storage.save_graph(graph_doc)

    def _to_graph_document(self) -> GraphDocument:
        """Convert NetworkX graph to LangChain GraphDocument.

        Returns:
            GraphDocument representation of the graph.
        """
        from langchain_core.documents import Document

        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'Entity')
            properties = {k: v for k, v in data.items() if k not in ('node_type', 'weight')}
            nodes.append(LCNode(id=node_id, type=node_type, properties=properties))

        relationships = []
        for source_id, target_id, rel_type, data in self.graph.edges(data=True, keys=True):
            source_node = LCNode(id=source_id, type='Entity')
            target_node = LCNode(id=target_id, type='Entity')
            properties = {k: v for k, v in data.items() if k != 'weight'}
            relationships.append(LCRelationship(
                source=source_node,
                target=target_node,
                type=rel_type,
                properties=properties
            ))

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=Document(page_content="", metadata={"source": "graph_builder"})
        )

    def remove_self_loops(self) -> "GraphBuilder":
        """Remove self-loops from the graph.

        Returns:
            self for method chaining.
        """
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        return self

    def get_graph(self) -> nx.MultiDiGraph:
        """Get the underlying NetworkX graph.

        Returns:
            The NetworkX MultiDiGraph.
        """
        return self.graph

    def stats(self) -> dict:
        """Get graph statistics.

        Returns:
            Dictionary with node and edge counts.
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges()
        }

    def clear(self) -> "GraphBuilder":
        """Clear the graph.

        Returns:
            self for method chaining.
        """
        self.graph.clear()
        return self


def build_graph(
    documents: List[GraphDocument],
    neo4j_storage: Optional[Neo4jStorage] = None
) -> nx.MultiDiGraph:
    """Build a graph from a list of GraphDocuments.

    Args:
        documents: List of GraphDocuments to build from.
        neo4j_storage: Optional Neo4jStorage for persisting.

    Returns:
        The built NetworkX MultiDiGraph.
    """
    builder = GraphBuilder(neo4j_storage=neo4j_storage)
    return builder.build(documents)


def get_graph_builder(
    neo4j_uri: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None
) -> GraphBuilder:
    """Get a GraphBuilder instance with optional Neo4j storage.

    Args:
        neo4j_uri: Neo4j connection URI.
        neo4j_username: Neo4j username.
        neo4j_password: Neo4j password.

    Returns:
        GraphBuilder instance.
    """
    storage = None
    if neo4j_uri is not None:
        storage = get_neo4j_storage(
            uri=neo4j_uri,
            username=neo4j_username or "neo4j",
            password=neo4j_password
        )
    return GraphBuilder(neo4j_storage=storage)


if __name__ == "__main__":
    print("=" * 60)
    print("GraphBuilder Tests")
    print("=" * 60)

    # Test 1: Create GraphBuilder
    print("\n[Test 1] Creating GraphBuilder...")
    builder = GraphBuilder()
    print(f"  ✓ GraphBuilder created")

    # Test 2: Create test GraphDocument
    print("\n[Test 2] Creating test GraphDocument...")
    test_doc = GraphDocument(
        nodes=[
            LCNode(id="Alice", type="Person", properties={"age": 30}),
            LCNode(id="Bob", type="Person", properties={"age": 25}),
            LCNode(id="Acme Corp", type="Organization", properties={"founded": 2020}),
        ],
        relationships=[
            LCRelationship(
                source=LCNode(id="Alice"),
                target=LCNode(id="Bob"),
                type="knows",
                properties={"since": 2015}
            ),
            LCRelationship(
                source=LCNode(id="Alice"),
                target=LCNode(id="Acme Corp"),
                type="works_at",
                properties={"role": "Engineer"}
            ),
        ],
        source=None
    )
    print(f"  GraphDocument created with {len(test_doc.nodes)} nodes, {len(test_doc.relationships)} relationships")

    # Test 3: Add document to graph
    print("\n[Test 3] Adding document to graph...")
    builder.add_document(test_doc)
    stats = builder.stats()
    print(f"  Stats: {stats}")
    assert stats["num_nodes"] == 3, f"Expected 3 nodes, got {stats['num_nodes']}"
    assert stats["num_edges"] == 2, f"Expected 2 edges, got {stats['num_edges']}"
    print(f"  ✓ Document added successfully")

    # Test 4: Build from multiple documents
    print("\n[Test 4] Building from multiple documents...")
    builder2 = GraphBuilder()
    graph = builder2.build([test_doc, test_doc])  # Add twice to test weight increment
    stats = builder2.stats()
    print(f"  Stats after adding twice: {stats}")
    # Check node weight
    alice_weight = builder2.graph.nodes["Alice"]["weight"]
    print(f"  Alice node weight: {alice_weight}")
    assert alice_weight == 2, f"Expected weight 2, got {alice_weight}"
    print(f"  ✓ Multiple documents built successfully")

    # Test 5: Convert to GraphDocument
    print("\n[Test 5] Converting to GraphDocument...")
    converted_doc = builder._to_graph_document()
    print(f"  Converted doc: {len(converted_doc.nodes)} nodes, {len(converted_doc.relationships)} relationships")
    print(f"  ✓ Conversion successful")

    # Test 6: Clear graph
    print("\n[Test 6] Clearing graph...")
    builder.clear()
    stats = builder.stats()
    print(f"  Stats after clear: {stats}")
    assert stats["num_nodes"] == 0, "Expected 0 nodes after clear"
    print(f"  ✓ Graph cleared successfully")

    # Test 7: get_graph_builder factory function
    print("\n[Test 7] Testing get_graph_builder factory...")
    builder3 = get_graph_builder()
    print(f"  ✓ GraphBuilder created via factory")

    # Test 8: build_graph convenience function
    print("\n[Test 8] Testing build_graph convenience function...")
    graph = build_graph([test_doc])
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  ✓ build_graph works")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
