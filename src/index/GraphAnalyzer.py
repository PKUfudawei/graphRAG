import heapq, faiss, json, os
import community as community_louvain
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class GraphAnalyzer:
    def __init__(self, LLM, embed_model, max_community_size=50):
        self.LLM = LLM
        self.embed_model = embed_model
        self.max_community_size = max_community_size


    def build_communities(self, graph, comm_nodes=None, next_comm_id=0):
        communities = {}
        comm_nodes = comm_nodes if comm_nodes else list(graph.nodes)

        if len(comm_nodes) <= self.max_community_size:
            for node in comm_nodes:
                graph.nodes[node]['community_id'] = next_comm_id

            communities[next_comm_id] = comm_nodes
            return graph, communities, next_comm_id + 1

        subgraph = graph.subgraph(comm_nodes).to_undirected()
        partition = community_louvain.best_partition(subgraph)
        partition_groups = {}
        for node, comm_id in partition.items():
            partition_groups.setdefault(comm_id, []).append(node)
        
        if len(partition_groups) == 1:
            for node in comm_nodes:
                graph.nodes[node]['community_id'] = next_comm_id
            communities[next_comm_id] = comm_nodes
            return graph, communities, next_comm_id + 1

        for group_nodes in partition_groups.values():
            graph, sub_comms, next_comm_id = self.build_communities(graph, group_nodes, next_comm_id)
            communities.update(sub_comms)

        return graph, communities, next_comm_id


    @staticmethod
    def get_communities(graph):
        communities = {}
        for node, node_data in graph.nodes(data=True):
            comm_id = node_data.get(f"community_id")
            if comm_id is not None:
                communities.setdefault(comm_id, []).append(node)

        return communities


    @staticmethod
    def get_relations(subgraph, top_k=100):
        relation_weight_dict = {
            f"{source} -> {relation} -> {target}": int(edge_data.get('weight', 1))
            for source, target, relation, edge_data in subgraph.edges(keys=True, data=True)
        }

        top_relations = heapq.nlargest(top_k, relation_weight_dict.items(), key=lambda x: x[1])
        relations = [f"- ({weight}) {relation}" for relation, weight in top_relations]
        return relations
    

    def summarize_community(self, subgraph, top_k=100):
        relations = self.get_relations(subgraph, top_k=top_k)
        
        system_prompt = "You are an expert knowledge graph summarizer."
        user_prompt = f"""
        You are given relationships extracted from a knowledge graph community.

        Each line represents a relation in the format:
        - (weight) source -> relation -> target

        The weight indicates how frequently the relation appears in the graph.
        The relations are sorted by weight in descending order.

        Top {top_k} most frequent relations in this community:

        {'\n'.join(relations)}

        Task:
        Identify the main entities, themes, and types of relationships represented in this community.
        Write a short paragraph summarizing the overall topic and the key connections between entities.
        Prioritize patterns suggested by higher-weight relations.
        """
        community_summary = self.LLM.generate_response(system_prompt=system_prompt, user_prompt=user_prompt)
        return community_summary, relations


    def process_community(self, subgraph, community_id):
        entities = subgraph.nodes
        summary, relations = self.summarize_community(subgraph)
        embedding_text = f"""
        Summary:
        {summary}

        Entities:
        {", ".join(entities)}

        Relations:
        {'\n'.join(relations)}
        """
        embedding = self.embed_model.encode(embedding_text)
        community_summary = {
            "community_id": community_id,
            "nodes": list(subgraph.nodes),
            "summary": summary,
            "relations": relations,
            "embedding": embedding
        }
        return community_summary


    def analyze(self, graph, max_workers=16):
        graph, communities, _ = self.build_communities(graph)
        community_summaries = []

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for community_id, comm_nodes in communities.items():
                subgraph = graph.subgraph(comm_nodes)
                futures.append(executor.submit(self.process_community, subgraph, community_id))

            for f in tqdm(as_completed(futures), total=len(futures), desc="Summarizing communities"):
                community_summaries.append(f.result())
            
            community_summaries.sort(key=lambda x: x["community_id"])
        return graph, community_summaries


    @staticmethod
    def save(community_summaries, index_path, meta_path):
        embeddings = np.vstack([c["embedding"] for c in community_summaries])
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
        faiss.write_index(index, index_path)

        os.makedirs(os.path.dirname(meta_path) or '.', exist_ok=True)
        communities_data = [
            {k: v for k, v in comm.items() if k!='embedding'} 
            for comm in community_summaries
        ]
        with open(meta_path, "w") as f:
            json.dump(communities_data, f, indent=2)

        print(f"\tCommunities index saved: {index_path}")
        print(f"\tCommunities meta data saved: {meta_path}")
        

    @staticmethod
    def load(index_path, meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            communities_data = json.load(f)
        return index, communities_data
