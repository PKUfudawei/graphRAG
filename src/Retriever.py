class GraphRetriever:
    def __init__(self, graph, index, llm):
        self.graph = graph
        self.index = index
        self.llm = llm

    def extract_entities(self, query):
        prompt = f"""
Extract entities from the question.

Return JSON:
{{"entities":[]}}

Question:
{query}
"""

        result = self.llm.generate_json(
            "Entity extraction",
            prompt
        )

        return result.get("entities", [])

    def retrieve(self, query, top_chunks=5):

        entities = self.extract_entities(query)

        chunk_scores = {}

        for entity in entities:

            if entity not in self.index.entity_to_chunks:
                continue

            for chunk in self.index.entity_to_chunks[entity]:

                chunk_scores[chunk] = chunk_scores.get(chunk, 0) + 1

        ranked = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [c for c, _ in ranked[:top_chunks]]


class HybridRetriever:
    def __init__(self, graph_retriever, vector_store):
        self.graph_retriever = graph_retriever
        self.vector_store = vector_store

    def retrieve(self, query):
        graph_chunks = self.graph_retriever.retrieve(query)
        vector_chunks = self.vector_store.search(query)
        merged = list(set(graph_chunks + vector_chunks))

        return merged