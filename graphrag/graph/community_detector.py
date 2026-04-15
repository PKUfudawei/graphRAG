"""Community detection using Louvain algorithm on GraphDocument."""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import community as community_louvain
import networkx as nx

from langchain_community.graphs.graph_document import GraphDocument


class CommunityDetector:
    """Detect communities in a GraphDocument using Louvain algorithm.

    Uses hierarchical Louvain algorithm for community detection.

    Args:
        max_community_size: Maximum community size. Larger communities are subdivided.
    """

    def __init__(self, max_community_size: int = 50):
        self.max_community_size = max_community_size

    def detect_communities(
        self, graph_doc: GraphDocument
    ) -> Dict[int, List[str]]:
        """Detect communities in the GraphDocument.

        Args:
            graph_doc: Input GraphDocument.

        Returns:
            Dictionary mapping community_id to list of node IDs.
        """
        # Build adjacency for Louvain
        node_ids = [str(node.id) for node in graph_doc.nodes]
        node_id_set = set(node_ids)

        # Build edge list with weights
        edges = []
        for rel in graph_doc.relationships:
            src = str(rel.source.id)
            tgt = str(rel.target.id)
            if src in node_id_set and tgt in node_id_set:
                weight = rel.properties.get("weight", 1)
                edges.append((src, tgt, weight))

        # Create NetworkX graph for Louvain (internal use only)
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(node_ids)
        nx_graph.add_weighted_edges_from(edges)

        # Run hierarchical community detection
        communities = self._detect_hierarchical(nx_graph)

        return communities

    def _detect_hierarchical(
        self, graph: nx.Graph
    ) -> Dict[int, List[str]]:
        """Recursively detect communities using Louvain.

        Args:
            graph: NetworkX graph (internal representation).

        Returns:
            Dictionary mapping community_id to list of node IDs.
        """
        communities = {}
        next_comm_id = self._build_communities_recursive(
            graph, list(graph.nodes()), communities, 0
        )
        return communities

    def _build_communities_recursive(
        self,
        graph: nx.Graph,
        comm_nodes: List[str],
        communities: Dict[int, List[str]],
        next_comm_id: int
    ) -> int:
        """Recursively build communities.

        Args:
            graph: NetworkX graph.
            comm_nodes: Candidate nodes for current community.
            communities: Communities dictionary (modified in place).
            next_comm_id: Next available community ID.

        Returns:
            Next unused community ID.
        """
        # If nodes <= max size, treat as one community
        if len(comm_nodes) <= self.max_community_size:
            communities[next_comm_id] = comm_nodes
            return next_comm_id + 1

        # Build subgraph and run Louvain
        subgraph = graph.subgraph(comm_nodes)
        if subgraph.number_of_edges() == 0:
            # No edges, treat as one community
            communities[next_comm_id] = comm_nodes
            return next_comm_id + 1

        # Run Louvain with weights
        partition = community_louvain.best_partition(
            subgraph, weight="weight"
        )

        # Group by partition
        partition_groups: Dict[int, List[str]] = defaultdict(list)
        for node, comm_id in partition.items():
            partition_groups[comm_id].append(node)

        # If only one partition, don't subdivide further
        if len(partition_groups) == 1:
            communities[next_comm_id] = comm_nodes
            return next_comm_id + 1

        # Recursively process each subgroup
        for group_nodes in partition_groups.values():
            next_comm_id = self._build_communities_recursive(
                graph, group_nodes, communities, next_comm_id
            )

        return next_comm_id

    def get_community_subgraphs(
        self,
        graph_doc: GraphDocument,
        communities: Dict[int, List[str]]
    ) -> Dict[int, GraphDocument]:
        """Extract subgraph GraphDocument for each community.

        Args:
            graph_doc: Original GraphDocument.
            communities: Communities dictionary from detect_communities.

        Returns:
            Dictionary mapping community_id to subgraph GraphDocument.
        """
        from langchain_core.documents import Document

        node_map = {str(node.id): node for node in graph_doc.nodes}

        community_subgraphs = {}
        for comm_id, node_ids in communities.items():
            node_set = set(node_ids)
            comm_nodes = [node_map[nid] for nid in node_ids if nid in node_map]
            comm_rels = [
                rel for rel in graph_doc.relationships
                if str(rel.source.id) in node_set and str(rel.target.id) in node_set
            ]

            community_subgraphs[comm_id] = GraphDocument(
                nodes=comm_nodes,
                relationships=comm_rels,
                source=Document(
                    page_content="",
                    metadata={"community_id": comm_id, "merged": True}
                )
            )

        return community_subgraphs


if __name__ == "__main__":
    """Test cases for CommunityDetector."""
    from langchain_community.graphs.graph_document import Node, Relationship
    from langchain_core.documents import Document

    # Test 1: Basic community detection
    print("Test 1: Basic community detection")
    nodes = [
        Node(id="1", properties={"name": "A"}),
        Node(id="2", properties={"name": "B"}),
        Node(id="3", properties={"name": "C"}),
        Node(id="4", properties={"name": "D"}),
    ]
    relationships = [
        Relationship(source=nodes[0], target=nodes[1], properties={"weight": 1}, type="related"),
        Relationship(source=nodes[1], target=nodes[2], properties={"weight": 1}, type="related"),
        Relationship(source=nodes[2], target=nodes[3], properties={"weight": 1}, type="related"),
    ]
    graph_doc = GraphDocument(
        nodes=nodes,
        relationships=relationships,
        source=Document(page_content="test graph")
    )

    detector = CommunityDetector(max_community_size=50)
    communities = detector.detect_communities(graph_doc)
    print(f"  Detected {len(communities)} communities")
    for comm_id, node_ids in communities.items():
        print(f"  Community {comm_id}: {node_ids}")
    assert len(communities) > 0, "Should detect at least one community"
    print("  PASSED")

    # Test 2: Empty graph
    print("Test 2: Empty graph")
    empty_graph = GraphDocument(
        nodes=[],
        relationships=[],
        source=Document(page_content="empty graph")
    )
    empty_communities = detector.detect_communities(empty_graph)
    print(f"  Detected {len(empty_communities)} communities")
    # Empty graph may produce an empty community, verify all communities are empty
    total_nodes = sum(len(nodes) for nodes in empty_communities.values())
    assert total_nodes == 0, "Empty graph should have no nodes in communities"
    print("  PASSED")

    # Test 3: Large community subdivision
    print("Test 3: Large community subdivision (max_community_size=3)")
    large_nodes = [Node(id=str(i), properties={"name": f"Node{i}"}) for i in range(10)]
    large_rels = [
        Relationship(source=large_nodes[i], target=large_nodes[i + 1], properties={"weight": 1}, type="related")
        for i in range(9)
    ]
    large_graph = GraphDocument(
        nodes=large_nodes,
        relationships=large_rels,
        source=Document(page_content="large graph")
    )

    small_detector = CommunityDetector(max_community_size=3)
    small_communities = small_detector.detect_communities(large_graph)
    print(f"  Detected {len(small_communities)} communities")
    for comm_id, node_ids in small_communities.items():
        print(f"  Community {comm_id}: {node_ids} (size: {len(node_ids)})")
    # Verify all nodes are covered
    all_nodes = set()
    for node_ids in small_communities.values():
        all_nodes.update(node_ids)
    assert len(all_nodes) == 10, "All 10 nodes should be covered"
    print("  PASSED")

    # Test 4: Community subgraphs extraction
    print("Test 4: Community subgraphs extraction")
    subgraphs = detector.get_community_subgraphs(graph_doc, communities)
    print(f"  Extracted {len(subgraphs)} subgraphs")
    for comm_id, subgraph in subgraphs.items():
        print(f"  Subgraph {comm_id}: {len(subgraph.nodes)} nodes, {len(subgraph.relationships)} relationships")
        assert len(subgraph.nodes) > 0, f"Subgraph {comm_id} should have nodes"
    print("  PASSED")

    print("\nAll tests passed!")
