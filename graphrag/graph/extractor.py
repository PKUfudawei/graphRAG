"""Entity and relation extractor using LangChain with json_schema structured output."""
import os
import sys
import warnings
import asyncio
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

from models.extraction_result import ExtractionResult
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

    def _build_graph_document(self, result: ExtractionResult, document: Document) -> GraphDocument:
        """Build GraphDocument from ExtractionResult."""
        # Validate and fix orphan edges
        try:
            result.validate_edges_reference_existing_entities()
        except ValueError:
            result.fix_orphan_edges()

        # Build nodes
        nodes_data = result.get_nodes()
        nodes = [Node(id=node.name, type=node.type) for node in nodes_data]
        node_map = {node.id: node for node in nodes}

        # Ensure all edge nodes exist
        for edge in result.get_edges():
            if edge.source not in node_map:
                node_map[edge.source] = Node(id=edge.source, type="entity")
            if edge.target not in node_map:
                node_map[edge.target] = Node(id=edge.target, type="entity")

        # Build relationships
        relationships = [
            Relationship(
                source=node_map[edge.source],
                target=node_map[edge.target],
                type=edge.relation
            )
            for edge in result.get_edges()
        ]

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=document,
        )

    def extract(self, document: Document) -> GraphDocument:
        """Extract entities and relations using json_schema structured output.

        Args:
            document: Input LangChain Document.

        Returns:
            LangChain GraphDocument instance.
        """
        text = document.page_content

        try:
            structured_llm = self.llm.with_structured_output(
                ExtractionResult,
                method="json_schema",
                include_raw=False,
                strict=True,
            )
            chain = self.prompt | structured_llm
            result: ExtractionResult = chain.invoke({"text": text})
            return self._build_graph_document(result, document)
        except Exception as e:
            print(f"\tError in entity extraction: {e}")
            return GraphDocument(nodes=[], relationships=[], source=document)

    async def aextract(self, document: Document) -> GraphDocument:
        """Async extract entities and relations.

        Args:
            document: Input LangChain Document.

        Returns:
            LangChain GraphDocument instance.
        """
        text = document.page_content

        try:
            structured_llm = self.llm.with_structured_output(
                ExtractionResult,
                method="json_schema",
                include_raw=False,
                strict=True,
            )
            chain = self.prompt | structured_llm
            result: ExtractionResult = await chain.ainvoke({"text": text})
            return self._build_graph_document(result, document)
        except Exception as e:
            print(f"\tError in entity extraction: {e}")
            return GraphDocument(nodes=[], relationships=[], source=document, metadata=document.metadata)

    def extract_batch(
        self,
        documents: List[Document],
        mode: str = "thread"
    ) -> List[GraphDocument]:
        """Batch extract entities and relations from multiple documents.

        Args:
            documents: List of input LangChain Documents.
            mode: Execution mode. Options:
                  - "async": Async concurrent execution (recommended for I/O bound)
                  - "thread": Thread pool concurrent execution
                  - "sync": Sequential execution

        Returns:
            List of LangChain GraphDocument instances.
        """
        if mode not in ['sync', 'thread', 'async']:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: 'sync', 'thread', 'async'")

        if not documents:
            return []

        if mode == "async":
            return asyncio.run(self._extract_batch_async(documents))
        elif mode == "thread":
            return self._extract_batch_thread(documents)
        else:  # mode == "sync"
            return [self.extract(doc) for doc in documents]

    def _extract_batch_thread(self, documents: List[Document]) -> List[GraphDocument]:
        """Thread pool batch extraction with progress bar."""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.extract, doc) for doc in documents]

            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Extracting (thread)"
            ):
                results.append(future.result())

        # 按原始顺序返回
        results.sort(key=lambda r: r.source.metadata.get("chunk_id", 0))
        return results

    async def _extract_batch_async(self, documents: List[Document]) -> List[GraphDocument]:
        """Async batch extraction with concurrency limit and progress bar."""
        semaphore = asyncio.Semaphore(self.max_workers)

        # 创建进度条
        pbar = tqdm(total=len(documents), desc="Extracting (async)", unit="doc")

        async def extract_with_limit(doc: Document) -> GraphDocument:
            result = await self.aextract(doc)
            pbar.update(1)
            return result

        # 并发执行所有任务
        tasks = [extract_with_limit(doc) for doc in documents]
        results = await asyncio.gather(*tasks)

        pbar.close()
        return list(results)


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
    from langchain_core.documents import Document

    extractor = get_extractor()

    text = "Ebenezer Scrooge is a wealthy but miserly businessman in Victorian London. " \
           "He is visited by the ghost of his former partner Jacob Marley on Christmas Eve."

    doc = Document(page_content=text, metadata={"source": "test"})
    result = extractor.extract(doc)
    print(f"Nodes: {len(result.nodes)}")
    print(f"Relationships: {len(result.relationships)}")
    for node in result.nodes:
        print(f"  - {node.id} ({node.type})")
    for rel in result.relationships:
        print(f"  - {rel.source.id} --{rel.type}--> {rel.target.id}")
