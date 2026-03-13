import heapq, faiss, pickle, os
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
                graph.nodes[node]['community'] = next_comm_id

            communities[next_comm_id] = comm_nodes
            return graph, communities, next_comm_id + 1

        subgraph = graph.subgraph(comm_nodes).to_undirected()
        partition = community_louvain.best_partition(subgraph)
        partition_groups = {}
        for node, comm_id in partition.items():
            partition_groups.setdefault(comm_id, []).append(node)
        
        if len(partition_groups) == 1:
            for node in comm_nodes:
                graph.nodes[node]['community'] = next_comm_id
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
            comm_id = node_data.get(f"community")
            if comm_id is not None:
                communities.setdefault(comm_id, []).append(node)

        return communities


    @staticmethod
    def get_relation_text(subgraph, max_relations=100):
        relation_weight_dict = {
            f"{source} {relation} {target}": int(edge_data.get('weight', 1))
            for source, target, relation, edge_data in subgraph.edges(keys=True, data=True)
        }

        top_relations = heapq.nlargest(
            max_relations,
            relation_weight_dict.items(),
            key=lambda x: x[1]
        )
        
        relation_list = [f"- ({weight}) {relation}" for relation, weight in top_relations]
        relation_text = '\n'.join(relation_list) if relation_list else "- No significant relationships in this community."
        return relation_text
    

    def summarize_community(self, subgraph, max_relations=100):
        relation_text = self.get_relation_text(subgraph, max_relations=max_relations)
        
        system_prompt = "You are an expert knowledge graph summarizer."
        user_prompt = f"""
        You are given relationships extracted from a knowledge graph community.

        Each line represents a relation in the format:
        - (weight) source relation target

        The weight indicates how frequently the relation appears in the graph.
        The relations are sorted by weight in descending order.

        Top {max_relations} most frequent relations in this community:

        {relation_text}

        Task:
        Identify the main entities, themes, and types of relationships represented in this community.
        Write a short paragraph summarizing the overall topic and the key connections between entities.
        Prioritize patterns suggested by higher-weight relations.
        """
        community_summary = self.LLM.generate_response(system_prompt=system_prompt, user_prompt=user_prompt)
        return community_summary, relation_text


    def compress_summary(self, community_summaries):
        summaries_text = "\n\n".join(community_summaries)
        system_prompt = "You are an expert knowledge graph analyst."
        user_prompt = f"""
        The following are summaries of different communities extracted from a knowledge graph.

        Each community represents a cluster of related entities and relationships.

        Community summaries:

        {summaries_text}

        Task:
        Write a concise global summary describing the major themes, topics, and relationships
        present across the entire knowledge graph.
        Focus on identifying the overall structure and main domains represented.
        """
        global_summary = self.LLM.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        return global_summary


    def process_community(self, subgraph, comm_id):
        summary, relations = self.summarize_community(subgraph)
        embedding = self.embed_model.encode(summary)
        return {
            "community": comm_id,
            "nodes": list(subgraph.nodes),
            "summary": summary,
            "relations": relations,
            "embedding": embedding,
        }


    def analyze(self, graph, max_workers=16):
        graph, communities, _ = self.build_communities(graph)
        community_summaries = []
        
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for comm_id, nodes in communities.items():
                subgraph = graph.subgraph(nodes)
                futures.append(executor.submit(self.process_community, subgraph, comm_id))

            for f in tqdm(as_completed(futures), total=len(futures), desc="Summarizing communities"):
                community_summaries.append(f.result())
            
        #global_summary = self.compress_summary(community_summaries)
        return graph, communities, community_summaries


    @staticmethod
    def save_faiss_index(community_summaries, index_path, meta_path):
        embeddings = np.vstack([c["embedding"] for c in community_summaries]).astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
        faiss.write_index(index, index_path)

        os.makedirs(os.path.dirname(meta_path) or '.', exist_ok=True)
        with open(meta_path, "wb") as f:
            pickle.dump(community_summaries, f)

        print(f"\tCommunity summaries index saved to {index_path}")
        print(f"\tCommunity metadata saved to {meta_path}")
        
    
    @staticmethod
    def load_faiss_index(index_path, meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            community_summaries = pickle.load(f)
        return index, community_summaries
