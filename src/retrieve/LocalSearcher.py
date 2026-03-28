import faiss, time
from collections import Counter
from functools import lru_cache
from ..index.GraphAnalyzer import GraphAnalyzer


class LocalSearcher:
    def __init__(self, graph, node_embeddings, chunks, community_summaries, embed_model, LLM):
        self.graph = graph
        self.entities = list(self.graph.nodes)
        self.node_embeddings = node_embeddings
        self.chunks = chunks
        self.community_summaries = community_summaries
        self.embed_model = embed_model
        self.LLM = LLM


    @lru_cache(maxsize=128)
    def get_query_embedding(self, query_str):
        t0 = time.time()
        emb = self.embed_model.encode([query_str])
        faiss.normalize_L2(emb)
        print(f"[Timer] embedding: {time.time() - t0:.3f}s")
        return emb


    def retrieve_entities(self, query, top_k=5):
        t0 = time.time()
        query_embedding = self.get_query_embedding(query if isinstance(query, str) else query[0])
        similarities, idx = self.node_embeddings.search(query_embedding, top_k)
        entities = [self.entities[i] for i in idx[0]]
        print(f"[Timer] retrieve_entities: {time.time() - t0:.3f}s")
        return entities


    def graph_traversal(self, seed_entities, hops=2):
        t0 = time.time()

        visited = set(seed_entities)
        frontier = set(seed_entities)

        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
            frontier = next_frontier

        print(f"[Timer] graph_traversal: {time.time() - t0:.3f}s")
        return visited


    def retrieve_chunks(self, nodes, top_k=10):
        t0 = time.time()

        chunk_counter = Counter()
        for node in nodes:
            data = self.graph.nodes[node]
            chunk_counter.update(data.get("chunk_ids", []))

        top_chunks = chunk_counter.most_common(top_k)
        chunks = [
            f"- ({freq}) {self.chunks[chunk_id]['text']}"
            for chunk_id, freq in top_chunks
        ]

        print(f"[Timer] retrieve_chunks: {time.time() - t0:.3f}s")
        return chunks


    def retrieve_communities(self, nodes, top_k=5):
        t0 = time.time()

        comm_counter = Counter()
        for node in nodes:
            comm_counter[self.graph.nodes[node]['community_id']] += 1
        top_comm_ids = comm_counter.most_common(top_k)

        summaries = [
            self.community_summaries[comm_id]["summary"]
            for comm_id, _ in top_comm_ids
        ]

        print(f"[Timer] retrieve_communities: {time.time() - t0:.3f}s")
        return summaries


    def build_context(self, entities, relations, chunks, summaries):
        t0 = time.time()

        context = "### Entities\n"
        context += "\n".join(entities)
        context += "\n\n### Relations\n"
        context += "\n".join(relations)
        context += "\n\n### Community Summaries\n"
        context += "\n".join(summaries)
        context += "\n\n### Source Text\n"
        context += "\n".join(chunks)

        print(f"[Timer] build_context: {time.time() - t0:.3f}s")
        return context


    def build_prompt(self, query, context):
        return f"""
You are answering questions using a knowledge graph.

Question:
{query}

Context:
{context}

Answer the question using the provided information.
"""


    def search(self, query):
        t_total = time.time()

        t0 = time.time()
        seed_entities = self.retrieve_entities(query)
        print(f"[Timer] stage1_entity_retrieval: {time.time() - t0:.3f}s")

        t0 = time.time()
        retrieved_entities = self.graph_traversal(seed_entities)
        print(f"[Timer] stage2_graph: {time.time() - t0:.3f}s")

        t0 = time.time()
        subgraph = self.graph.subgraph(retrieved_entities)
        retrieved_relations = GraphAnalyzer.get_relations(subgraph=subgraph)
        print(f"[Timer] stage3_relations: {time.time() - t0:.3f}s")

        t0 = time.time()
        retrieved_chunks = self.retrieve_chunks(retrieved_entities)
        print(f"[Timer] stage4_chunks: {time.time() - t0:.3f}s")

        t0 = time.time()
        retrieved_summaries = self.retrieve_communities(retrieved_entities)
        print(f"[Timer] stage5_communities: {time.time() - t0:.3f}s")

        t0 = time.time()
        context = self.build_context(
            entities=retrieved_entities,
            relations=retrieved_relations,
            chunks=retrieved_chunks,
            summaries=retrieved_summaries
        )
        prompt = self.build_prompt(query, context)
        print(f"[Timer] stage6_prompt: {time.time() - t0:.3f}s")

        t0 = time.time()
        response = self.LLM.generate_response(prompt)
        print(f"[Timer] stage7_llm: {time.time() - t0:.3f}s")

        print(f"[Timer] TOTAL: {time.time() - t_total:.3f}s")

        return response