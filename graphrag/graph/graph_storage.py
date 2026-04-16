"""Graph storage utilities for Neo4j."""
import os
from typing import Any, Dict, List, Optional

from langchain_community.graphs.graph_document import (
    GraphDocument,
    Node as LCNode,
    Relationship as LCRelationship
)
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv
load_dotenv()


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
                # Sanitize relationship type for Cypher - remove/replace all special chars
                safe_rel_type = rel_type.upper()
                safe_rel_type = safe_rel_type.replace(" ", "_")
                safe_rel_type = safe_rel_type.replace("-", "_")
                safe_rel_type = safe_rel_type.replace("'", "_")
                safe_rel_type = safe_rel_type.replace('"', "_")
                safe_rel_type = safe_rel_type.replace(".", "_")
                safe_rel_type = safe_rel_type.replace("/", "_")
                safe_rel_type = safe_rel_type.replace("\\", "_")
                safe_rel_type = safe_rel_type.replace("(", "_")
                safe_rel_type = safe_rel_type.replace(")", "_")
                safe_rel_type = safe_rel_type.replace("?", "_")
                safe_rel_type = safe_rel_type.replace("!", "_")
                safe_rel_type = safe_rel_type.replace(",", "_")
                safe_rel_type = safe_rel_type.replace(";", "_")
                safe_rel_type = safe_rel_type.replace(":", "_")
                # Ensure it starts with a letter
                if safe_rel_type and not safe_rel_type[0].isalpha():
                    safe_rel_type = "R_" + safe_rel_type
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
    from langchain_core.documents import Document
    from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")

    print(f"Testing Neo4jStorage: {NEO4J_URI}")

    # Create test data
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
        source=Document(page_content="test", metadata={"source": "test"})
    )

    # Test without Neo4j connection
    print("\n[Test 1] Class instantiation...")
    storage = get_neo4j_storage(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    assert storage.uri == NEO4J_URI
    assert storage.username == NEO4J_USERNAME
    print("  ✓ Passed")

    print("\n[Test 2] Context manager...")
    with get_neo4j_storage(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) as s:
        assert s.uri == NEO4J_URI
    print("  ✓ Passed")

    print("\n[Test 3] GraphDocument structure...")
    assert len(test_doc.nodes) == 2
    assert len(test_doc.relationships) == 1
    print("  ✓ Passed")

    # Integration tests if Neo4j is available
    if not _check_neo4j_connection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD):
        print("\n⚠ Neo4j not accessible. Integration tests skipped.")
        print("  Start Neo4j: ./neo4j start")
        print("\nAll unit tests passed!")
    else:
        try:
            storage = get_neo4j_storage(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

            print("\n[Test 4] Clear graph...")
            storage.clear_graph()
            print("  ✓ Passed")

            print("\n[Test 5] Save graph...")
            storage.save_graph(test_doc)
            print("  ✓ Passed")

            print("\n[Test 6] Get stats...")
            stats = storage.stats()
            assert stats["num_nodes"] == 2
            assert stats["num_relationships"] == 1
            print("  ✓ Passed")

            print("\n[Test 7] Load graph...")
            loaded = storage.load_graph()
            assert len(loaded.nodes) == 2
            assert len(loaded.relationships) == 1
            print("  ✓ Passed")

            print("\n[Test 8] Custom query...")
            results = storage.query("MATCH (n:Entity) RETURN count(n) as cnt")
            assert results[0]["cnt"] == 2
            print("  ✓ Passed")

            storage.clear_graph()
            storage.close()
            print("\nAll tests passed!")

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            raise
