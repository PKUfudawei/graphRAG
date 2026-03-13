import json, faiss, os, re
import networkx as nx
from networkx.readwrite import json_graph
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import Counter


class GraphBuilder:
    def __init__(self, LLM=None, embed_model=None, graph=None):
        self.LLM = LLM
        self.embed_model = embed_model
        self.graph = nx.MultiDiGraph() if graph is None else graph
        
    
    def extract_nodes_and_edges(self, text):
        system_prompt = """
You are a precise entity and relation extraction assistant.
Your task is to extract all relevant entities and the relationships between them from a given text.
- Only return valid JSON.
- Nodes should have fields "name" and "type" (no underscores).
- Edges should have fields "source", "target", and "relation" (no underscores).
- Use lowercase and concise relation names.
- Do not include any explanations, comments, or text outside the JSON.
- Make sure to return the complete JSON, do not truncate or omit any part.
"""
        user_prompt = f"""
Text: 
{text}

Return the JSON following the schema:
{{
"nodes":[{{"name":"","type":""}}],
"edges":[{{"source":"","target":"","relation":""}}]
}}
"""
        try:
            result = self.LLM.generate_response(user_prompt=user_prompt, system_prompt=system_prompt)
            result = result.lower().replace('_', ' ')
            try:
                data = json.loads(result)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', result, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    data = {}
            nodes = [n for n in data.get("nodes", []) if isinstance(n, dict) and "name" in n]
            edges = [
                e for e in data.get("edges", []) 
                if isinstance(e, dict) and "source" in e and "target" in e and "relation" in e
            ]
            return nodes, edges
        except Exception as e:
            print(f"\tError in entity extraction: {e}")
            print(result)
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
    
    
    def process_chunks(self, chunks, workers=16):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(
                executor.map(self.extract_nodes_and_edges, [c.text for c in chunks]),
                total=len(chunks), desc="Extracting nodes and edges from chunks",
            ))

        nodes = []
        edges = []
        for n, e in results:
            nodes.extend(n)
            edges.extend(e)
        if not nodes:
            raise ValueError("Found no node from chunks!")
        return nodes, edges


    def normalize_names(self, names, threshold=0.9):
        freq = Counter(names)
        unique_names = list(freq.keys())
        embeddings = self.embed_model.encode(unique_names)

        index = faiss.IndexHNSWFlat(embeddings.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(embeddings)
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
                if score >= threshold and idx != i:
                    union(i, idx)

        clusters = {}
        for i, name in enumerate(unique_names):
            root = find(i)
            clusters.setdefault(root, []).append(name)
        alias = {}
        for cluster in clusters.values():
            if not cluster:
                continue
            #if len(cluster) > 5:
            #    for n in cluster:
            #        alias[n] = n
            #    continue
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
                "relation": edge.get("relation", "related with").strip().lower().replace('_', ' ')
            })

        return new_edges
    
    
    def build_graph(self, chunks):
        nodes, edges = self.process_chunks(chunks=chunks)
        node_names = set([n.get('name') for n in nodes])
        num_nodes = len(node_names)
        print(f"\tExtracted {num_nodes} unique nodes from chunks")
        for e in edges:
            if e["source"] not in node_names:
                nodes.append({"name": e["source"], "type": "unknown"})
            if e["target"] not in node_names:
                nodes.append({"name": e["target"], "type": "unknown"})
        full_num_nodes = len(set([n.get('name') for n in nodes]))
        print(f"\tAdded {full_num_nodes-num_nodes} nodes referenced in edges but absent from node list")
        print(f"\tBefore merging: {full_num_nodes} nodes")
        alias = self.normalize_names([n.get("name") for n in nodes if n.get("name")])
        merged_nodes = self.normalize_nodes(nodes, alias)
        print(f"\tAfter merging: {len(set([n.get('name') for n in merged_nodes]))} nodes")
        merged_edges = self.normalize_edges(edges, alias)
        
        self.add_nodes(nodes=merged_nodes)
        self.add_edges(edges=merged_edges)
        print(f"\tGraph built: {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

        return self.graph
        
    
    def save_graph(self, path, graph=None):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = json_graph.node_link_data(graph if graph else self.graph, edges="edges")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"\tGraph saved: {path}")


    def load_graph(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = json_graph.node_link_graph(data, edges="edges")
        print(f"Graph loaded: {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self.graph
