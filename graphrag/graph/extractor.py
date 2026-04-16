"""Entity and relation extractor using LangChain with json_schema structured output."""
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from tqdm import tqdm

# Add project root to path for models import
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.graph import ExtractionResult
from models.llm import get_llm


class Extractor:
    """Extract entities and relations from text using json_schema structured output.

    Uses LangChain's with_structured_output with json_schema method
    for structured output with Pydantic validation.

    Args:
        llm: LangChain language model instance.
        max_workers: Maximum number of concurrent workers for parallel extraction.
    """

    def __init__(self, llm: Optional[BaseLanguageModel] = None, max_workers: int = 16):
        # Suppress Pydantic serialization warnings from LangChain internals
        warnings.filterwarnings(
            "ignore",
            message="Pydantic serializer warnings",
            category=UserWarning,
            module="pydantic"
        )
        self.llm = llm or get_llm()
        self.max_workers = max_workers
        self._setup_prompt()

    def _setup_prompt(self) -> None:
        """Set up the entity extraction prompt template."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an entity and relation extraction assistant.
Extract all entities and their relationships from the given text.

Return entities as a list of strings or objects with name and type fields.
Return relationships as a list of objects with source, target, and relation fields.
All source and target in relationships must exist in entities."""),
            ("human", "Text: {text}")
        ])

    def extract(self, text: str, source: Optional[str] = None) -> GraphDocument:
        """Extract entities and relations using json_schema structured output.

        Args:
            text: Input text.
            source: Optional source identifier.

        Returns:
            LangChain GraphDocument instance.
        """
        try:
            # Use json_schema method for structured output
            structured_llm = self.llm.with_structured_output(
                ExtractionResult,
                method="json_schema",
                include_raw=False,
                strict=True,
            )
            chain = self.prompt | structured_llm
            result: ExtractionResult = chain.invoke({"text": text})

            return self._parse_to_graph_document(result, text, source)
        except Exception as e:
            print(f"\tError in entity extraction: {e}")
            return self._empty_graph_document(text, source)

    def _parse_to_graph_document(
        self, result: ExtractionResult, text: str, source: Optional[str]
    ) -> GraphDocument:
        """Parse ExtractionResult to LangChain GraphDocument with validation.

        First validates that all edges reference existing entities.
        If validation fails, automatically adds missing entities.
        """
        try:
            result.validate_edges_reference_existing_entities()
        except ValueError as e:
            print(f"\tValidation warning: {e}")
            print(f"\tFixing orphan edges...")
            result.fix_orphan_edges()

        nodes_data = result.get_nodes()
        nodes = [
            Node(id=node.name, type=node.type)
            for node in nodes_data
        ]

        node_map = {node.id: node for node in nodes}

        # Ensure all edge nodes exist
        for edge in result.get_edges():
            if edge.source not in node_map:
                node_map[edge.source] = Node(id=edge.source, type="entity")
            if edge.target not in node_map:
                node_map[edge.target] = Node(id=edge.target, type="entity")

        relationships = [
            Relationship(
                source=node_map[edge.source],
                target=node_map[edge.target],
                type=edge.relation
            )
            for edge in result.get_edges()
        ]

        source_doc = Document(
            page_content=text,
            metadata={"source": source}
        ) if text else Document(page_content="", metadata={"source": source})

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=source_doc
        )

    def _empty_graph_document(self, text: str, source: Optional[str]) -> GraphDocument:
        """Return an empty GraphDocument."""
        source_doc = Document(
            page_content=text,
            metadata={"source": source}
        ) if text else Document(page_content="", metadata={"source": source})
        return GraphDocument(nodes=[], relationships=[], source=source_doc)

    def extract_batch(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None,
        parallel: bool = True
    ) -> List[GraphDocument]:
        """Batch extract entities and relations from multiple texts.

        Args:
            texts: List of input texts.
            sources: Optional list of source identifiers.
            parallel: If True, use parallel extraction. Default is True.

        Returns:
            List of LangChain GraphDocument instances.
        """
        if sources is None:
            sources = [None] * len(texts)

        if not parallel:
            return [self.extract(text, source) for text, source in zip(texts, sources)]

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.extract, text, source)
                for text, source in zip(texts, sources)
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Extracting entities and relations"
            ):
                results.append(future.result())

        return results


def get_extractor(
    llm: Optional[BaseLanguageModel] = None,
    max_workers: int = 16
) -> Extractor:
    """Get an entity extractor instance.

    Args:
        llm: LangChain language model instance. If None, loads default.
        max_workers: Maximum number of concurrent workers.

    Returns:
        EntityExtractor instance.
    """
    llm = llm or get_llm()
    return Extractor(llm=llm, max_workers=max_workers)


if __name__ == "__main__":
    extractor = get_extractor()

    text = "Ebenezer Scrooge is a wealthy but miserly businessman in Victorian London. " \
           "He is visited by the ghost of his former partner Jacob Marley on Christmas Eve."

    result = extractor.extract(text, source="test")
    print(f"Nodes: {len(result.nodes)}")
    print(f"Relationships: {len(result.relationships)}")
    for node in result.nodes:
        print(f"  - {node.id} ({node.type})")
    for rel in result.relationships:
        print(f"  - {rel.source.id} --{rel.type}--> {rel.target.id}")
