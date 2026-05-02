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
        """Set up the entity extraction prompt template (LightRAG style)."""
        # Build system prompt with escaped braces for JSON examples
        system_prompt = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1. **Entity Extraction & Output:**
   * **Identification:** Identify clearly defined and meaningful entities in the input text.
   * **Entity Details:** For each identified entity, extract the following information:
     - `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
     - `entity_type`: Categorize the entity using one of the following types: Person, Organization, Location, Event, Concept, Method, Content, Data, Artifact, NaturalObject. If none apply, use "Other".
     - `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.

2. **Relationship Extraction & Output:**
   * **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
   * **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities, decompose it into multiple binary (two-entity) relationship pairs.
   * **Relationship Details:** For each binary relationship, extract the following fields:
     - `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction.
     - `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction.
     - `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords separated by comma `,`. **DO NOT use special delimiters.**
     - `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities.
     - `weight`: Importance weight (1.0-10.0, default 1.0).

3. **Output Order & Prioritization:**
   * Output all extracted entities first, followed by all relationships.
   * Within relationships, prioritize those **most significant** to the core meaning of the input text.

4. **Context & Objectivity:**
   * Ensure all entity names and descriptions are written in the **third person**.
   * Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`.

5. **Language:**
   * The entire output must be written in the same language as the input text.
   * Proper nouns should be retained in their original language if a proper translation is not available.

---Examples---
Example 1:
Input: "Apple Inc. was founded by Steve Jobs in 1976. The company is headquartered in Cupertino, California."
Output:
{{
  "entities": [
    {{"entity_name": "Apple Inc.", "entity_type": "Organization", "entity_description": "A technology company founded in 1976 and headquartered in Cupertino, California."}},
    {{"entity_name": "Steve Jobs", "entity_type": "Person", "entity_description": "Co-founder of Apple Inc."}},
    {{"entity_name": "Cupertino", "entity_type": "Location", "entity_description": "A city in California where Apple Inc. is headquartered."}}
  ],
  "relationships": [
    {{"source_entity": "Steve Jobs", "target_entity": "Apple Inc.", "relationship_keywords": "founding, entrepreneurship", "relationship_description": "Steve Jobs founded Apple Inc. in 1976.", "weight": 8.0}},
    {{"source_entity": "Apple Inc.", "target_entity": "Cupertino", "relationship_keywords": "headquarters, location", "relationship_description": "Apple Inc. is headquartered in Cupertino, California.", "weight": 5.0}}
  ]
}}

Example 2:
Input: "北京是中国的首都，位于华北平原。北京市是中华人民共和国的政治文化中心。"
Output:
{{
  "entities": [
    {{"entity_name": "北京", "entity_type": "Location", "entity_description": "中国的首都，位于华北平原，是中华人民共和国的政治文化中心。"}},
    {{"entity_name": "中国", "entity_type": "Location", "entity_description": "一个国家，北京是其首都。"}},
    {{"entity_name": "华北平原", "entity_type": "Location", "entity_description": "一个地理区域，北京位于此平原上。"}}
  ],
  "relationships": [
    {{"source_entity": "北京", "target_entity": "中国", "relationship_keywords": "capital, political center", "relationship_description": "北京是中国的首都和政治文化中心。", "weight": 9.0}},
    {{"source_entity": "北京", "target_entity": "华北平原", "relationship_keywords": "location, geography", "relationship_description": "北京位于华北平原上。", "weight": 5.0}}
  ]
}}"""

        human_prompt = """---Task---
Extract entities and relationships from the input text below.

---Instructions---
1. Output ONLY valid JSON matching the ExtractionResult schema.
2. Do NOT include any introductory or concluding remarks.
3. Ensure all relationship source_entity and target_entity names exactly match entity_name in entities list.
4. Use title case for entity names (capitalize significant words).
5. Keep proper nouns in their original language.

---Input Text---
```
{text}
```

---Output---"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

    def _build_graph_document(self, result: ExtractionResult, document: Document) -> GraphDocument:
        """Build GraphDocument from ExtractionResult (LightRAG style with metadata)."""
        # Validate and fix orphan edges
        try:
            result.validate_edges_reference_existing_entities()
        except ValueError:
            result.fix_orphan_edges()

        # Build nodes with metadata (LightRAG style)
        nodes_data = result.get_nodes()
        chunk_id = document.metadata.get("chunk_id", "unknown_0")
        file_path = document.metadata.get("source", "unknown_source")

        nodes = []
        for entity in nodes_data:
            node = Node(
                id=entity.entity_name,
                type=entity.entity_type,
                properties={
                    "description": entity.entity_description,
                    "chunk_id": chunk_id,
                    "file_path": file_path,
                }
            )
            nodes.append(node)
        node_map = {node.id: node for node in nodes}

        # Ensure all edge nodes exist
        for rel in result.get_edges():
            if rel.source_entity not in node_map:
                node_map[rel.source_entity] = Node(id=rel.source_entity, type="Other", description="")
            if rel.target_entity not in node_map:
                node_map[rel.target_entity] = Node(id=rel.target_entity, type="Other", description="")

        relationships = []
        for rel in result.get_edges():
            relationship = Relationship(
                source=node_map[rel.source_entity],
                target=node_map[rel.target_entity],
                type="RELATED_TO",
                properties={
                    "keywords": rel.relationship_keywords,
                    "description": rel.relationship_description,
                    "weight": rel.weight,
                    "chunk_id": chunk_id,
                    "file_path": file_path,
                }
            )
            relationships.append(relationship)

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
        #results.sort(key=lambda r: r.source.metadata.get("chunk_id", 0))
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
