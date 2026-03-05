import networkx as nx
import json
from collections import defaultdict


class GraphRAG:
    def __init__(self, client, graph, chunks=[]):
        self.client = client
        self.graph = graph
        self.chunks = chunks


    def extract_entities_from_query(self, query):
        extract_prompt = f"""
        Extract key entities (such as people, places, concepts, etc.) from the following question.
        Return as a JSON object with an 'entities' array.

        Example format: {{"entities": ["entity1", "entity2", "entity3"]}}

        Question: {query}
        """
        try:
            response = self.client.chat.completions.create(
                model="openrouter/auto", temperature=0.1, response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are an expert at entity extraction. Extract only the most relevant entities and return them as a JSON object with an 'entities' array."},
                    {"role": "user", "content": extract_prompt}
                ],
            )

            result = json.loads(response.choices[0].message.content)
            return result.get('entities', []) 
        except Exception as e:
            print(f"Error extracting entities from query: {e}")
            return []
    
    
    def retrieve_relevant_nodes(self, query_entities=set()):
        relevant_entities = set()
        for entity in query_entities:
            if entity in self.graph:
                relevant_entities.add(entity)

                neighbors = set(self.graph.successors(entity)) | set(self.graph.predecessors(entity))
                relevant_entities.update(neighbors)
                
                # 添加二跳关系（可选）
                #if len(relevant_entities) < max_nodes:
                #    for neighbor in neighbors:
                #        second_hop = list(self.graph.neighbors(neighbor)) + list(self.graph.predecessors(neighbor))
                #        relevant_entities.update(second_hop)
        
        relevant_nodes = []
        for entity in list(relevant_entities):
            if entity in self.graph:
                node_data = self.graph.nodes[entity]
                relevant_nodes.append(f"- Entity: {entity}, type: {node_data.get('type', 'unknown')}, occurrences: {node_data.get('occurrences', 0)}")
        
        return relevant_nodes, relevant_entities


    def retrieve_relevant_edges(self, relevant_entities):
        relevant_edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in relevant_entities and v in relevant_entities:
                relevant_edges.append(f"- Relation: {u} {data.get('relation', 'related')} {v}, occurrences: {len(data.get('chunk_ids', []))}")
        return relevant_edges


    def retrieve_relevant_chunks(self, relevant_entities):
        relevant_chunk_ids = set()
        for entity in relevant_entities:
            chunk_ids = self.graph.nodes.get(entity, {}).get('chunk_ids', [])
            relevant_chunk_ids.update(chunk_ids)

        relevant_chunks = []
        for chunk_id in relevant_chunk_ids:
            if chunk_id < len(self.chunks):
                relevant_chunks.append(f"- Text chunk {chunk_id}:\n{self.chunks[chunk_id].text}")
        return relevant_chunks


    def retrieve_graph_contexts(self, query):
        query_entities = self.extract_entities_from_query(query)
        relevant_contexts = []

        relevant_nodes, relevant_entities = self.retrieve_relevant_nodes(set(query_entities))
        if relevant_nodes:
            relevant_contexts.append("Relevant Entities:\n" + "\n".join(relevant_nodes))
            
        relevant_edges = self.retrieve_relevant_edges(relevant_entities)
        if relevant_edges:
            relevant_contexts.append("\nRelevant Entity Relations:\n" + "\n".join(relevant_edges))
        
        relevant_chunks = self.retrieve_relevant_chunks(relevant_entities)
        if relevant_chunks:
            relevant_contexts.append("\nRelevant Original Texts:\n" + "\n\n".join(relevant_chunks))
        
        return "\n\n".join(relevant_contexts)


    def query(self, query):
        graph_contexts = self.retrieve_graph_contexts(query)
        
        prompt = f"""Please answer the question based on the information extracted from the knowledge graph.

        Graph context information:
        {graph_contexts}

        Question: {query}

        Requirements:
        1. Fully utilize the entities, relationships, and original text information provided in the graph
        2. If there are multiple relevant entities or relationships, comprehensively analyze the connections between them
        3. The answer should be well-structured and logical
        4. If the information is insufficient, please clearly indicate this

        Answer:"""
        print("Prompt:\n", prompt)
        response = self.client.chat.completions.create(
            model="openrouter/auto", temperature=0.2, response_format={"type": "text"},
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response.choices[0].message.content