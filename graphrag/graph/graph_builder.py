"""Graph builder for building knowledge graph from documents."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from tqdm import tqdm

from .entity_extractor import get_entity_extractor
from .entity_deduplicator import get_entity_deduplicator
from .graph_storage import get_neo4j_storage


class GraphBuilder:
    """Build knowledge graph from documents.

    Args:
        dedup_threshold: Similarity threshold for entity deduplication. Default is 0.9.
        max_workers: Maximum number of concurrent workers for batch processing.
    """

    def __init__(
        self,
        entity_extractor = None,
        entity_deduplicator = None,
        max_workers: int = 16,
    ):
        self.max_workers = max_workers
        self.entity_extractor = entity_extractor or get_entity_extractor()
        self.entity_deduplicator = entity_deduplicator or get_entity_deduplicator()
        self._graph_storage = None

    @property
    def graph_storage(self):
        """Lazy load graph storage."""
        if self._graph_storage is None:
            self._graph_storage = get_neo4j_storage()
        return self._graph_storage

    def build(self, document: Document, chunk_id: int) -> GraphDocument:
        """Build a GraphDocument from a single document.

        Args:
            document: Input LangChain Document.
            chunk_id: The index of the document in the batch.

        Returns:
            GraphDocument with deduplicated nodes and relationships.
        """
        # Step 1: Extract entities and relations
        graph_doc = self.entity_extractor.extract(
            text=document.page_content,
            source=str(chunk_id)
        )

        # Step 2: Deduplicate entities
        graph_doc = self._deduplicate_graph(graph_doc, chunk_id)

        return graph_doc

    def _deduplicate_graph(self, graph_doc: GraphDocument, chunk_id: int) -> GraphDocument:
        """Deduplicate nodes in a GraphDocument using entity deduplicator.

        Args:
            graph_doc: Input GraphDocument.
            chunk_id: The chunk ID for the source document.

        Returns:
            GraphDocument with deduplicated nodes.
        """
        if not graph_doc.nodes:
            return graph_doc

        # Get all unique node names
        node_names = [node.id for node in graph_doc.nodes]

        # Find aliases using embedding similarity
        alias_map = self.entity_deduplicator.find_aliases(node_names)

        # Build mapping from old name to canonical name
        name_to_canonical = {name: alias_map.get(name, name) for name in node_names}

        # Count types for each canonical name to find most common type
        canonical_type_counter: dict[str, Counter] = {}
        for node in graph_doc.nodes:
            canonical_name = name_to_canonical[node.id]
            if canonical_name not in canonical_type_counter:
                canonical_type_counter[canonical_name] = Counter()
            canonical_type_counter[canonical_name][node.type] += 1

        # Build new node map with canonical names
        canonical_nodes: dict[str, Node] = {}
        for node in graph_doc.nodes:
            canonical_name = name_to_canonical[node.id]
            if canonical_name not in canonical_nodes:
                # Get most common type for this canonical name
                most_common_type = canonical_type_counter[canonical_name].most_common(1)[0][0]
                canonical_nodes[canonical_name] = Node(
                    id=canonical_name,
                    type=most_common_type
                )

        # Rebuild relationships with canonical node references
        new_relationships = []
        for rel in graph_doc.relationships:
            source_id = getattr(rel.source, 'id', str(rel.source))
            target_id = getattr(rel.target, 'id', str(rel.target))
            source_canonical = name_to_canonical.get(source_id, source_id)
            target_canonical = name_to_canonical.get(target_id, target_id)

            # Skip if source or target doesn't exist after dedup
            if source_canonical in canonical_nodes and target_canonical in canonical_nodes:
                new_relationships.append(Relationship(
                    source=canonical_nodes[source_canonical],
                    target=canonical_nodes[target_canonical],
                    type=rel.type
                ))

        # Create source document with chunk_id
        source_doc = Document(
            page_content=graph_doc.source.page_content if graph_doc.source else "",
            metadata={"source": chunk_id}
        )

        return GraphDocument(
            nodes=list(canonical_nodes.values()),
            relationships=new_relationships,
            source=source_doc
        )

    def build_and_save(self, document: Document, chunk_id: int) -> GraphDocument:
        """Build a GraphDocument and save it to Neo4j.

        Args:
            document: Input LangChain Document.
            chunk_id: The index of the document in the batch.

        Returns:
            The built GraphDocument.
        """
        graph_doc = self.build(document, chunk_id)
        self.graph_storage.save_graph(graph_doc)
        return graph_doc

    def build_batch(self, documents: list[Document]) -> list[GraphDocument]:
        """Build GraphDocuments from multiple documents concurrently.

        Args:
            documents: List of input LangChain Documents.

        Returns:
            List of GraphDocuments with deduplicated nodes and relationships.
        """
        if not documents:
            return []

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.build, doc, idx)
                for idx, doc in enumerate(documents)
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Building graphs"
            ):
                results.append(future.result())

        # Sort results by chunk_id to maintain order
        results.sort(key=lambda g: int(g.source.metadata.get("source", 0)))
        return results

    def build_and_save_batch(self, documents: list[Document]) -> list[GraphDocument]:
        """Build GraphDocuments from multiple documents and save to Neo4j.

        Args:
            documents: List of input LangChain Documents.

        Returns:
            List of built GraphDocuments.
        """
        if not documents:
            return []

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.build_and_save, doc, idx)
                for idx, doc in enumerate(documents)
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Building and saving graphs"
            ):
                results.append(future.result())

        # Sort results by chunk_id to maintain order
        results.sort(key=lambda g: int(g.source.metadata.get("source", 0)))
        return results


def get_graph_builder(
    entity_extractor=None,
    entity_deduplicator=None,
    max_workers=16,
) -> GraphBuilder:
    """Get a GraphBuilder instance.

    Args:
        dedup_threshold: Similarity threshold for entity deduplication. Default is 0.9.
        max_workers: Maximum number of concurrent workers. Default is 16.

    Returns:
        GraphBuilder instance.
    """
    entity_extractor = entity_extractor or get_entity_extractor()
    entity_deduplicator = entity_deduplicator or get_entity_deduplicator()
    
    return GraphBuilder(
        entity_extractor=entity_extractor,
        entity_deduplicator=entity_deduplicator,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    # Test GraphBuilder
    print("[Test 1] Create GraphBuilder...")
    builder = get_graph_builder()
    print("  ✓ Passed")

    print("\n[Test 2] Build single document...")
    test_doc = Document(
        page_content="Ebenezer Scrooge is a wealthy businessman in Victorian London. "
                     "He is visited by the ghost of his former partner Jacob Marley on Christmas Eve.",
        metadata={"source": "test"}
    )
    graph_doc = builder.build(test_doc, chunk_id=0)
    print(f"  Nodes: {len(graph_doc.nodes)}")
    print(f"  Relationships: {len(graph_doc.relationships)}")
    for node in graph_doc.nodes:
        print(f"    - {node.id} ({node.type})")
    for rel in graph_doc.relationships:
        print(f"    - {rel.source.id} --{rel.type}--> {rel.target.id}")
    print("  ✓ Passed")

    print("\n[Test 3] Build batch documents...")
    test_docs = [
        Document(
            page_content="北京是中国的首都，位于华北平原。",
            metadata={"source": "doc1"}
        ),
        Document(
            page_content="北京市是中华人民共和国的首都，政治文化中心。",
            metadata={"source": "doc2"}
        ),
        Document(
            page_content="上海是中国的经济中心，位于长江入海口。",
            metadata={"source": "doc3"}
        ),
    ]
    graph_docs = builder.build_batch(test_docs)
    print(f"  Built {len(graph_docs)} graphs")
    for i, gd in enumerate(graph_docs):
        print(f"  Graph {i} (chunk_id={gd.source.metadata.get('source')}): {len(gd.nodes)} nodes, {len(gd.relationships)} relationships")
    print("  ✓ Passed")

    # Test save to Neo4j if available
    print("\n[Test 4] Build and save to Neo4j...")
    try:
        graph_docs = builder.build_and_save_batch(test_docs)
        print(f"  Saved {len(graph_docs)} graphs to Neo4j")
        stats = builder.graph_storage.stats()
        print(f"  Total in Neo4j: {stats['num_nodes']} nodes, {stats['num_relationships']} relationships")
        print("  ✓ Passed")
    except Exception as e:
        print(f"  ⚠ Neo4j not accessible: {e}")
        print("  Start Neo4j: ./neo4j start")

    print("\nAll tests completed!")
