import json, os
import networkx as nx
from tqdm import tqdm
from networkx.readwrite import json_graph
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import community


class GraphBuilder:
    def __init__(self, embed_model, LLM, cache_dir='./checkpoints/'):
        self.embed_model = embed_model
        self.LLM = LLM
        self.graph = nx.MultiDiGraph()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    
    def normalize_entity_name(self, name):
        if not name:
            return None
        return name.strip().lower().replace('_', ' ')


    def extract_nodes_and_edges(self, text):
        system_prompt = "You extract entities and relations from text."
        user_prompt = f"""
Extract entities and relationships.

Return ONLY valid JSON.
Schema:
{{
"nodes":[{{"name":"","type":""}}],
"edges":[{{"source":"","target":"","relation":""}}]
}}

Text: 
{text}
"""
        try:
            result = self.LLM.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
            return result.get('nodes', []), result.get('edges', [])
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return [], []


    def update_graph(self, chunk):
        nodes, edges = self.extract_nodes_and_edges(text=chunk.text)

        for node in nodes:
            entity = self.normalize_entity(node.get("name"))
            if not entity:
                continue
            node_type = node.get("type", "unknown")
            if entity not in self.graph:
                self.graph.add_node(
                    node_for_adding=entity, type=node_type, chunk_ids=[chunk.id]
                )
            else:
                if chunk.id not in self.graph.nodes[entity]["chunk_ids"]:
                    self.graph.nodes[entity]["chunk_ids"].append(chunk.id)
                else:
                    print(f"Warning: node '{entity}' in chunk {chunk.id} does not have 'chunk_ids' attribute. Initializing it.")
                    self.graph.nodes[entity]['chunk_ids'] = [chunk.id]

        
        for edge in edges:
            source = self.normalize_entity(edge.get("source"))
            target = self.normalize_entity(edge.get("target"))
            relation = edge.get("relation", "related with")
            
            if not source or not target:
                continue

            merged = False
            if self.graph.has_edge(source, target):
                for key, edge_data in self.graph[source][target].items():
                    if edge_data["relation"] == relation:
                        if chunk.id not in edge_data["chunk_ids"]:
                            edge_data['chunk_ids'].append(chunk.id)
                        merged = True
                        break
            if not merged:
                self.graph.add_edge(
                    u_for_edge=source, v_for_edge=target, 
                    relation=relation, chunk_ids=[chunk.id],
                )

        return self.graph


    def build_graph(self, chunks, workers=8):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(tqdm(
                executor.map(self.update_graph, chunks),
                total=len(chunks), desc="Building graph",
            ))
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    
    def merge_entity(self, threshold=0.9):
        entities = list(self.graph.nodes)
        embeddings = self.embed_model.encode(entities)
        similarity = cosine_similarity(embeddings)
        merged = {}

        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                if similarity[i][j] > threshold:
                    merged[entities[j]] = entities[i]

        for old, new in merged.items():
            if old == new:
                continue
            nx.relabel_nodes(self.graph, {old: new}, copy=False)

        return self.graph


    def cluster_community(self, graph, level=0, max_level=2):
        if level >= max_level:
            return
        undirected = self.graph.to_undirected()
        partition = community.best_partition(undirected)
        for node, community in partition.items():
            self.graph.nodes[node][f"community_{level}"] = community

        communities = {}
        for node, community in partition.items():
            communities.setdefault(community, []).append(node)

        for comm_nodes in communities.values():
            subgraph = self.graph.subgraph(comm_nodes)

            if len(subgraph) > 10:
                self.cluster_community(subgraph, level+1)
        return self.graph


    def build_index(self):
        self.entity_to_chunks, self.community_to_chunks = {}, {}
        for node, node_data in self.graph.nodes(data=True):
            chunk_ids = node_data.get("chunk_ids", [])
            self.entity_to_chunks[node] = set(chunk_ids)

            community = node_data.get("community")
            if community is None:
                continue
            self.community_to_chunks.setdefault(community, set()).update(chunk_ids)

        return self.entity_to_chunks, self.community_to_chunks
        

    def save_graph(self, path, graph=None):
        data = json_graph.node_link_data(self.graph if graph is None else graph, edges="edges")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"Graph saved: {path}")


    def load_graph(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = json_graph.node_link_graph(data, edges="edges")
        print(f"Graph loaded: {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self.graph


class GraphIndex:
    def __init__(self):
        self.entity_to_chunks = {}
        self.community_to_chunks = {}

    def build(self, graph):
        for node, node_data in graph.nodes(data=True):
            chunk_ids = node_data.get("chunk_ids", [])
            self.entity_to_chunks[node] = set(chunk_ids)

            community = node_data.get("community")
            if community is None:
                continue
            self.community_to_chunks.setdefault(community, set()).update(chunk_ids)
