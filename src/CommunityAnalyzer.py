import community as community_louvain


class CommunityAnalyzer:
    def __init__(self, LLM, max_level=2, max_nodes=10):
        self.LLM = LLM
        self.max_level = max_level
        self.max_nodes = max_nodes


    def build_level_communities(self, graph, level):
        level_communities = {}
        partition = community_louvain.best_partition(graph.to_undirected())
        for node, comm_id in partition.items():
            graph.nodes[node][f"community_{level}"] = comm_id
            level_communities.setdefault(comm_id, []).append(node)

        return graph, level_communities


    def build_hierarchical_communities(self, graph, level=0):
        if level >= self.max_level:
            return graph

        graph, level_communities = self.build_level_communities(graph=graph, level=level)
        for community in level_communities.values():
            subgraph = graph.subgraph(community)
            if subgraph.number_of_nodes() > self.max_nodes:
                self.build_hierarchical_communities(subgraph, level=level+1)
        
        return graph
    
    
    def get_level_communities(self, graph, level=0):
        level_communities = {}
        for node, node_data in graph.nodes(data=True):
            comm_id = node_data.get(f"community_{level}")
            if comm_id is not None:
                level_communities.setdefault(comm_id, []).append(node)

        return level_communities
    
    
    def collect_relations(self, graph, community, max_relations=50):
        relations = []
        seen_relations = set()
        for node in community:
            for source, target, edge_data in graph.edges(node, data=True):
                relation = f"{source} {edge_data.get('relation','related with')} {target}"
                if relation not in seen_relations:
                    relations.append(relation)
                    seen_relations.add(relation)

                if len(relations) >= max_relations:
                    return relations

        return relations
    
    
    def summarize_community(self, graph, community, system_prompt=None):
        relations = self.collect_relations(graph=graph, community=community)
        text = "\n".join(relations)
        user_prompt = f"""
        Summarize the following knowledge graph relations:
        
        {text}
        
        Provide a concise description of the main concepts and relationships.
        """
        community_summary = self.LLM.generate_text(system_prompt=system_prompt, user_prompt=user_prompt)
        return community_summary
    
    
    def summarize_level(self, graph, level):
        communities = self.get_level_communities(graph=graph, level=level)
        level_summary = {
            comm_id: self.summarize_community(graph=graph, community=community) 
            for comm_id, community in communities.items()
        }
        
        return level_summary

    
    def summarize_hierarchy(self, graph):
        hierarchical_summary = {
            level: self.summarize_level(graph=graph, level=level) for level in range(self.max_level)
        }

        return hierarchical_summary

    
    def compress_summary(self, hierarchical_summary):
        texts = []
        for level, communities in hierarchical_summary.items():
            for comm_id, text in communities.items():
                texts.append(f"Level {level} Community {comm_id}:\n{text}")
        summary_text = "\n\n".join(texts)
        prompt = f"""
        Compress the following hierarchical community summaries into a single coherent description of the knowledge graph.
        
        {summary_text}
        """
        return self.LLM.generate_text(prompt)
    
    
    def analyze(self, graph):
        graph = self.build_hierarchical_communities(graph)
        hierarchy = self.summarize_hierarchy(graph)
        global_summary = self.compress_summary(hierarchy)
        return graph, hierarchy, global_summary
