from pydantic import BaseModel, Field
from typing import List, Optional, Union

class Node(BaseModel):
    """Extracted node from text."""
    name: str = Field(..., description="Entity name")
    type: str = Field(default="entity", description="Entity type (e.g., Person, Location, Organization)")


class Edge(BaseModel):
    """Extracted edge from text."""
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation: str = Field(..., description="Relation type (lowercase, 1-3 words)")


class ExtractionResult(BaseModel):
    """Result of entity and relation extraction."""
    entities: List[Union[Node, str]] = Field(
        ...,
        description="Extracted entities (can be strings or {name, type} objects)"
    )
    relationships: List[Edge] = Field(
        ...,
        description="Extracted relationships as list of {source, target, relation} objects"
    )

    def _get_entity_names(self) -> set:
        """Get set of all entity names."""
        names = set()
        for e in self.entities:
            if isinstance(e, str):
                names.add(e)
            else:
                names.add(e.name)
        return names

    def validate_edges_reference_existing_entities(self) -> bool:
        """Validate that all edge source/target entities exist in entities list.

        Returns:
            True if all edges reference valid entities.
        Raises:
            ValueError if validation fails.
        """
        entity_names = self._get_entity_names()
        for edge in self.relationships:
            if edge.source not in entity_names:
                raise ValueError(f"Edge source '{edge.source}' not found in entities")
            if edge.target not in entity_names:
                raise ValueError(f"Edge target '{edge.target}' not found in entities")
        return True

    def fix_orphan_edges(self) -> None:
        """Add missing entities from edges to entities list.

        This ensures all edge source/target entities exist in the entities list.
        """
        entity_names = self._get_entity_names()
        added_entities = set()

        for edge in self.relationships:
            if edge.source not in entity_names and edge.source not in added_entities:
                self.entities.append(Node(name=edge.source, type="entity"))
                added_entities.add(edge.source)
            if edge.target not in entity_names and edge.target not in added_entities:
                self.entities.append(Node(name=edge.target, type="entity"))
                added_entities.add(edge.target)

    def get_nodes(self) -> List[Node]:
        """Normalize entities to Node list."""
        nodes = []
        for e in self.entities:
            if isinstance(e, str):
                nodes.append(Node(name=e))
            else:
                nodes.append(e)
        return nodes

    def get_edges(self) -> List[Edge]:
        """Get relationships list."""
        return self.relationships