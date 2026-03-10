import json, faiss
import networkx as nx
from networkx.readwrite import json_graph
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import Counter


class GraphBuilder:
    def __init__(self, extract_model=None, embed_model=None, graph=None):
        self.extract_model = extract_model
        self.embed_model = embed_model
        self.graph = nx.MultiDiGraph() if graph is None else graph
        
    
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
            result = self.extract_model.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
            return result.get('nodes', []), result.get('edges', [])
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return [], []
    
    
    def add_nodes(self, nodes):
        for node in tqdm(nodes, desc='Adding nodes'):
            name = node.get("name")
            if not name:
                continue

            if name not in self.graph:
                self.graph.add_node(name, type=node.get("type", "unknown"), weight=1)
            else:
                self.graph.nodes[name]['weight'] += 1
        
        return self.graph
    
    
    def add_edges(self, edges):
        for edge in tqdm(edges, desc="Adding edges"):
            source, target = edge.get("source"), edge.get("target")
            relation = edge.get("relation", "related with")
            if not source or not target:
                continue

            if not self.graph.has_edge(source, target, key=relation):
                self.graph.add_edge(source, target, key=relation, weight=1)
            else:
                self.graph[source][target][relation]['weight'] += 1
        
        return self.graph


    def normalize_names(self, names, threshold=0.9):
        freq = Counter(names)
        unique_names = list(freq.keys())
        embeddings = self.embed_model.encode(unique_names)
        
        faiss.normalize_L2(embeddings)
        index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        index.add(embeddings)

        parent = list(range(len(unique_names)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pb] = pa

        for i, name in enumerate(unique_names):
            D, I = index.search(embeddings[i:i+1], 50)
            for score, idx in zip(D[0], I[0]):
                if score >= threshold:
                    union(i, idx)

        clusters = {}
        for i, name in enumerate(unique_names):
            root = find(i)
            clusters.setdefault(root, []).append(name)
        alias = {}
        for cluster in clusters.values():
            if not cluster:
                continue
            canonical = max(cluster, key=lambda x: (freq[x], x))
            alias.update({n: canonical for n in cluster})

        return alias


    def normalize_nodes(self, nodes, alias):
        new_nodes = []
        for node in nodes:
            name = node.get("name")
            if not name:
                continue
            new_nodes.append({
                "name": alias.get(name, name),
                "type": node.get("type", "unknown"),
            })

        return new_nodes
    
    
    def normalize_edges(self, edges, alias):
        new_edges = []
        for edge in edges:
            source, target = edge.get("source"), edge.get("target")
            if not source or not target:
                continue

            source = alias.get(source, source)
            target = alias.get(target, target)
            new_edges.append({
                "source": source, "target": target,
                "relation": edge.get("relation", "related with").strip().lower()
            })

        return new_edges


    def process_chunks(self, chunks, workers=16):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(
                executor.map(self.extract_nodes_and_edges, [c.text for c in chunks]),
                total=len(chunks)
            ))

        nodes = []
        edges = []
        for n, e in results:
            nodes.extend(n)
            edges.extend(e)
        return nodes, edges
    
    
    def build_graph(self, chunks=None, nodes=None, edges=None):
        if chunks is not None:
            nodes, edges = self.process_chunks(chunks=chunks)
        elif nodes is None or edges is None:
            raise ValueError()

        alias = self.normalize_names([n.get("name") for n in nodes if n.get("name")])
        nodes = self.normalize_nodes(nodes, alias)
        edges = self.normalize_edges(edges, alias)
        self.add_nodes(nodes=nodes)
        self.add_edges(edges=edges)

        return self.graph
        
    
    def save_graph(self, path):
        data = json_graph.node_link_data(self.graph, edges="edges")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"Graph saved: {path}")


    def load_graph(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = json_graph.node_link_graph(data, edges="edges")
        print(f"Graph loaded: {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self.graph