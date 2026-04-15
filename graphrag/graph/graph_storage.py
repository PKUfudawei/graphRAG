"""Graph storage utilities for Neo4j."""
import os
from typing import Any, Dict, List, Optional

from langchain_community.graphs.graph_document import (
    GraphDocument,
    Node as LCNode,
    Relationship as LCRelationship
)
from neo4j import GraphDatabase, Driver


class Neo4jStorage:
    """Neo4j graph storage for LangChain GraphDocument.

    Args:
        uri: Neo4j connection URI. Default is "bolt://localhost:7687".
        username: Neo4j username. Default is "neo4j".
        password: Neo4j password. Default is from NEO4J_PASSWORD env var.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: Optional[str] = None
    ):
        self.uri = uri
        self.username = username
        self.password = password or os.environ.get("NEO4J_PASSWORD", "neo4j")
        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        """Lazy load the Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
        return self._driver

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jStorage":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def save_graph(self, graph_doc: GraphDocument) -> None:
        """Save a GraphDocument to Neo4j.

        Args:
            graph_doc: GraphDocument to save.
        """
        with self.driver.session() as session:
            # Save nodes
            for node in graph_doc.nodes:
                node_id = str(node.id)
                node_type = node.type or "Entity"
                properties = node.properties or {}

                session.run(
                    """
                    MERGE (n:Entity {id: $id})
                    SET n.node_type = $type
                    SET n += $properties
                    """,
                    id=node_id,
                    type=node_type,
                    properties=properties
                )

            # Save relationships (build Cypher dynamically for relationship type)
            for rel in graph_doc.relationships:
                source_id = str(rel.source.id)
                target_id = str(rel.target.id)
                rel_type = rel.type or "RELATED_TO"
                # Sanitize relationship type for Cypher
                safe_rel_type = rel_type.upper().replace(" ", "_").replace("-", "_")
                properties = rel.properties or {}

                cypher = f"""
                    MATCH (source:Entity {{id: $source_id}})
                    MATCH (target:Entity {{id: $target_id}})
                    MERGE (source)-[r:{safe_rel_type}]->(target)
                    SET r += $properties
                """
                session.run(
                    cypher,
                    source_id=source_id,
                    target_id=target_id,
                    properties=properties
                )

        print(f"\tGraph saved to Neo4j: {len(graph_doc.nodes)} nodes, {len(graph_doc.relationships)} relationships")

    def load_graph(self) -> GraphDocument:
        """Load a GraphDocument from Neo4j.

        Returns:
            GraphDocument instance.
        """
        from langchain_core.documents import Document

        nodes = []
        relationships = []

        with self.driver.session() as session:
            # Load nodes
            result = session.run(
                """
                MATCH (n:Entity)
                RETURN n.id as id, n.node_type as type, labels(n) as labels, properties(n) as props
                """
            )
            for record in result:
                props = dict(record["props"])
                node_type = props.pop("node_type", "Entity")
                nodes.append(LCNode(id=record["id"], type=node_type, properties=props))

            # Load relationships
            result = session.run(
                """
                MATCH (source:Entity)-[r]->(target:Entity)
                RETURN source.id as source_id, target.id as target_id, type(r) as rel_type, properties(r) as props
                """
            )
            for record in result:
                source_id = record["source_id"]
                target_id = record["target_id"]
                rel_type = record["rel_type"]
                props = dict(record["props"])

                # Create source and target nodes for relationship
                source_node = LCNode(id=source_id, type="Entity")
                target_node = LCNode(id=target_id, type="Entity")

                relationships.append(LCRelationship(
                    source=source_node,
                    target=target_node,
                    type=rel_type,
                    properties=props
                ))

        graph_doc = GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=Document(page_content="", metadata={"source": "neo4j"})
        )

        print(f"Graph loaded from Neo4j: {len(nodes)} nodes, {len(relationships)} relationships")
        return graph_doc

    def clear_graph(self) -> None:
        """Clear all data from Neo4j."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("\tNeo4j graph cleared")

    def stats(self) -> Dict[str, int]:
        """Get graph statistics from Neo4j.

        Returns:
            Dictionary with node and relationship counts.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WITH count(n) as node_count
                MATCH ()-[r]->()
                RETURN node_count, count(r) as rel_count
                """
            )
            record = result.single()
            return {
                "num_nodes": record["node_count"],
                "num_relationships": record["rel_count"]
            }

    def query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query string.
            parameters: Query parameters.

        Returns:
            List of result records as dictionaries.
        """
        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]


def get_neo4j_storage(
    uri: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: Optional[str] = None
) -> Neo4jStorage:
    """Get a Neo4j storage instance.

    Args:
        uri: Neo4j connection URI.
        username: Neo4j username.
        password: Neo4j password.

    Returns:
        Neo4jStorage instance.
    """
    return Neo4jStorage(uri=uri, username=username, password=password)


def _check_neo4j_connection(
    uri: str, username: str, password: str
) -> bool:
    """Check if Neo4j is accessible.

    Args:
        uri: Neo4j connection URI.
        username: Neo4j username.
        password: Neo4j password.

    Returns:
        True if Neo4j is accessible, False otherwise.
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import os

    print("=" * 60)
    print("Neo4jStorage Tests")
    print("=" * 60)

    # Get Neo4j connection settings from environment or use defaults
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")

    print(f"\nConnection: {NEO4J_URI}")
    print(f"Username: {NEO4J_USERNAME}")

    # Check Neo4j connectivity
    if not _check_neo4j_connection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD):
        print("\n⚠ Neo4j is not accessible. Skipping integration tests.")
        print("\nTo run full tests, start Neo4j:")
        print("  docker run -d --name neo4j-test -p 7687:7687 -p 7474:7474 \\" )
        print("    -e NEO4J_AUTH=neo4j/neo4j neo4j:latest")
        print("\nOr set environment variables:")
        print("  export NEO4J_URI=bolt://your-host:7687")
        print("  export NEO4J_USERNAME=neo4j")
        print("  export NEO4J_PASSWORD=your-password")

        # Run unit tests that don't require Neo4j
        print("\n" + "-" * 60)
        print("Running unit tests (no Neo4j required)")
        print("-" * 60)

        # Test 1: Class instantiation
        print("\n[Test 1] Creating Neo4jStorage...")
        storage = get_neo4j_storage(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        print(f"  ✓ Neo4jStorage created")
        print(f"    URI: {storage.uri}")
        print(f"    Username: {storage.username}")

        # Test 2: Context manager
        print("\n[Test 2] Testing context manager...")
        with get_neo4j_storage(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        ) as ctx_storage:
            print(f"  ✓ Context manager works")
            print(f"    Storage URI: {ctx_storage.uri}")

        # Test 3: GraphDocument structure
        print("\n[Test 3] Testing GraphDocument structure...")
        from langchain_core.documents import Document
        from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

        test_doc = GraphDocument(
            nodes=[
                Node(id="Alice", type="Person", properties={"age": 30}),
                Node(id="Bob", type="Person", properties={"age": 25}),
            ],
            relationships=[
                Relationship(
                    source=Node(id="Alice"),
                    target=Node(id="Bob"),
                    type="knows",
                    properties={"since": 2015}
                ),
            ],
            source=Document(page_content="test data", metadata={"source": "test"})
        )
        print(f"  ✓ GraphDocument created")
        print(f"    Nodes: {len(test_doc.nodes)}")
        print(f"    Relationships: {len(test_doc.relationships)}")

        print("\n" + "=" * 60)
        print("Unit tests passed! (Integration tests skipped)")
        print("=" * 60)
    else:
        # Neo4j is accessible, run full integration tests
        try:
            # Test 1: Create storage instance
            print("\n[Test 1] Creating Neo4jStorage...")
            storage = get_neo4j_storage(
                uri=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD
            )
            print("  ✓ Neo4jStorage created")

            # Test 2: Clear existing data
            print("\n[Test 2] Clearing existing data...")
            storage.clear_graph()
            print("  ✓ Graph cleared")

            # Test 3: Create and save a GraphDocument
            print("\n[Test 3] Saving GraphDocument...")
            from langchain_core.documents import Document
            from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

            test_doc = GraphDocument(
                nodes=[
                    Node(id="Alice", type="Person", properties={"age": 30}),
                    Node(id="Bob", type="Person", properties={"age": 25}),
                    Node(id="Acme Corp", type="Organization", properties={"founded": 2020}),
                ],
                relationships=[
                    Relationship(
                        source=Node(id="Alice"),
                        target=Node(id="Bob"),
                        type="knows",
                        properties={"since": 2015}
                    ),
                    Relationship(
                        source=Node(id="Alice"),
                        target=Node(id="Acme Corp"),
                        type="works_at",
                        properties={"role": "Engineer"}
                    ),
                ],
                source=Document(page_content="test data", metadata={"source": "test"})
            )

            storage.save_graph(test_doc)
            print("  ✓ GraphDocument saved")

            # Test 4: Get stats
            print("\n[Test 4] Getting graph stats...")
            stats = storage.stats()
            print(f"  Nodes: {stats['num_nodes']}")
            print(f"  Relationships: {stats['num_relationships']}")
            assert stats["num_nodes"] == 3, f"Expected 3 nodes, got {stats['num_nodes']}"
            assert stats["num_relationships"] == 2, f"Expected 2 relationships, got {stats['num_relationships']}"
            print("  ✓ Stats correct")

            # Test 5: Load graph
            print("\n[Test 5] Loading GraphDocument...")
            loaded_doc = storage.load_graph()
            print(f"  Loaded {len(loaded_doc.nodes)} nodes")
            print(f"  Loaded {len(loaded_doc.relationships)} relationships")
            assert len(loaded_doc.nodes) == 3, f"Expected 3 nodes, got {len(loaded_doc.nodes)}"
            assert len(loaded_doc.relationships) == 2, f"Expected 2 relationships, got {len(loaded_doc.relationships)}"
            print("  ✓ GraphDocument loaded")

            # Test 6: Custom Cypher query
            print("\n[Test 6] Running custom Cypher query...")
            results = storage.query(
                "MATCH (n:Entity) WHERE n.node_type = 'Person' RETURN n.id as name, n.age as age ORDER BY name"
            )
            print(f"  Query results: {results}")
            assert len(results) == 2, f"Expected 2 results, got {len(results)}"
            print("  ✓ Cypher query executed")

            # Test 7: Context manager
            print("\n[Test 7] Testing context manager...")
            with get_neo4j_storage(
                uri=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD
            ) as ctx_storage:
                stats = ctx_storage.stats()
                print(f"  Stats from context manager: {stats}")
            print("  ✓ Context manager works")

            # Cleanup
            storage.clear_graph()
            storage.close()

            print("\n" + "=" * 60)
            print("All tests passed!")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            raise
