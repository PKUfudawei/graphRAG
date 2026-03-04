import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm
from networkx.readwrite import json_graph


class GraphBuilder:
    def __init__(self, client):
        self.client = client
        self.graph = nx.MultiDiGraph()


    def extract_nodes_and_edges_from_chunk(self, chunk):
        prompt = f"""
        Extract all important entities and their relationships from the following text.
        
        Return a JSON object with two arrays:
        - nodes: [{{"name": "entity name", "type": "entity type (person/location/organization/concept/etc.)"}}]
        - edges: [{{"source": "source entity name", "target": "target entity name", "relation": "relationship description"}}]
        
        Note: The same entity or relationship may appear across multiple chunks. chunk_ids will be added later.
        
        Text: {chunk.text}
        """
        try:
            response = self.client.chat.completions.create(
                model="openrouter/auto", temperature=0.1, response_format={"type": "json_object"}, 
                messages=[
                    {"role": "system", "content": "You are an expert at information extraction. Extract entities and relationships accurately."},
                    {"role": "user", "content": prompt}
                ],
            )

            result = json.loads(response.choices[0].message.content)
            return result.get('nodes', []), result.get('edges', [])
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return [], []


    def update_graph_with_chunk(self, chunk, graph=None):
        self.graph = self.graph if graph is None else graph
        nodes, edges = self.extract_nodes_and_edges_from_chunk(chunk)

        for node in nodes:
            entity = node.get('name', '')
            if self.graph.has_node(entity):
                if chunk.id not in self.graph.nodes[entity]['chunk_ids']:
                    self.graph.nodes[entity]['chunk_ids'].append(chunk.id)
            else:
                self.graph.add_node(
                    node_for_adding=entity, type=node.get('type', 'unknown'),
                    chunk_ids=[chunk.id]
                )
        
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            relation = edge.get('relation', 'related')

            if self.graph.has_edge(source, target):
                found = False
                for key, edge_data in self.graph[source][target].items():
                    if edge_data.get('relation') != relation:
                        continue
                    if chunk.id not in edge_data.get('chunk_ids', []):
                        edge_data['chunk_ids'].append(chunk.id)
                    found = True
                    break
                if not found:
                    self.graph.add_edge(
                        u_for_edge=source, v_for_edge=target, 
                        relation=relation, chunk_ids=[chunk.id]
                    )
            elif source and target:
                self.graph.add_edge(
                    u_for_edge=source, v_for_edge=target, 
                    relation=edge.get('relation', 'related'), chunk_ids=[chunk.id]
                )

        return self.graph


    def build_graph_with_chunks(self, chunks):
        for chunk in tqdm(chunks, desc="Building knowledge graph"):
            self.update_graph_with_chunk(chunk=chunk)

        print(f"Knowledge graph built: {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges in total.")
        return self.graph


    def save_graph(self, file_path, graph=None):
        data = json_graph.node_link_data(self.graph if graph is None else graph)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Graph saved to {file_path}")


    def load_graph(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = json_graph.node_link_graph(data)
        print(f"Graph loaded from {file_path} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self.graph
