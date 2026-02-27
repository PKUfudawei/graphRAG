import networkx as nx
import json
from collections import defaultdict


class Chunk:
    def __init__(self, chunk_id=-1, text=""):
        self.id = chunk_id
        self.text = text


class GraphRAG:
    def __init__(self, client):
        self.client = client
        self.graph = nx.MultiDiGraph()
        self.chunks = []
        self.node_from_chunks = defaultdict(list)
    
    
    def load_and_chunk_text(self, file_path, chunk_size=500, overlap=50):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunk_id = 0
        for start in range(0, len(text), chunk_size - overlap):
            end = min(start + chunk_size, len(text))
            self.chunks.append(Chunk(chunk_id=chunk_id, text=text[start:end]))
            chunk_id += 1
        return self.chunks


    def extract_nodes_and_edges_from_chunk(self, chunk):
        prompt = f"""
        从以下文本中提取所有重要的实体和它们之间的关系。
        请以JSON格式返回，包含nodes和edges两个数组。
        
        实体格式：{{"name": "实体名称", "type": "实体类型（如人物、地点、组织、概念等）", "chunk_id": {chunk.id}}}
        关系格式：{{"source": "源实体名称", "target": "目标实体名称", "relation": "关系描述", "chunk_id": {chunk.id}}}
        
        文本: {chunk.text}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", temperature=0.1, response_format={"type": "json_object"}, 
                messages=[
                    {"role": "system", "content": "你是一个信息抽取专家，请准确提取实体和关系。"},
                    {"role": "user", "content": prompt}
                ],
            )

            result = json.loads(response.choices[0].message.content)
            return result.get('nodes', []), result.get('edges', [])
        except:
            return [], []


    def build_graph_from_chunk(self, chunk):
        nodes, edges = self.extract_nodes_and_edges_from_chunk(chunk)

        for node in nodes:
            node_name = node.get('name', '')
            if self.graph.has_node(node_name):
                self.graph.nodes[node_name]['occurrences'] += 1
            else:
                self.graph.add_node(node_name, type=node.get('type', '未知'), occurrences=1)
            self.node_from_chunks[node_name].append(chunk.id)
        
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            if source and target:
                self.graph.add_edge(source, target, relation=edge.get('relation', '相关'), chunk_id=chunk.id)


    def build_graph(self, chunks=None):
        print("正在构建知识图谱...")
        
        for chunk in self.chunks if chunks is None else chunks:
            self.build_graph_from_chunk(chunk=chunk)

        print(f"图谱构建完成！共 {self.graph.number_of_nodes()} 个实体，{self.graph.number_of_edges()} 条关系")


    def retrieve_with_graph(self, query, max_nodes=10, max_chunks=3):        
        # 1. 从问题中提取关键实体
        extract_prompt = f"从以下问题中提取关键实体（如人名、地名、概念等），以列表形式返回：\n{query}"
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.1,
            messages=[{"role": "user", "content": extract_prompt}],
        )
        
        query_nodes = response.choices[0].message.content.split()
        
        # 2. 在图谱中找到相关实体和路径
        relevant_nodes = set()
        
        for node in query_nodes:
            # 直接匹配的实体
            if node in self.graph:
                relevant_nodes.add(node)
                
                # 添加邻居节点（一跳关系）
                neighbors = list(self.graph.neighbors(node)) + list(self.graph.predecessors(node))
                relevant_nodes.update(neighbors[:max_nodes])
                
                # 添加二跳关系（可选）
                #if len(relevant_nodes) < max_nodes:
                #    for neighbor in neighbors[:5]:
                #        second_hop = list(self.graph.neighbors(neighbor)) + list(self.graph.predecessors(neighbor))
                #        relevant_nodes.update(second_hop[:max_nodes//2])

        # 3. 根据相关实体找到对应的文本块
        relevant_chunk_ids = set()
        for node in relevant_nodes:
            chunk_ids = self.node_from_chunks.get(node, [])
            relevant_chunk_ids.update(chunk_ids)
        
        # 4. 构建结构化的上下文
        context_parts = []
        
        # 添加实体信息
        node_info = []
        for node in list(relevant_nodes)[:max_nodes]:
            if node in self.graph:
                node_data = self.graph.nodes[node]
                node_info.append(f"实体: {node} (类型: {node_data.get('type', '未知')})")
        
        if node_info:
            context_parts.append("相关实体:\n" + "\n".join(node_info))
        
        # 添加关系信息
        relation_info = []
        for u, v, data in self.graph.edges(data=True):
            if u in relevant_nodes and v in relevant_nodes:
                relation_info.append(f"{u} {data.get('relation', '相关')} {v}")
        
        if relation_info:
            context_parts.append("\n实体关系:\n" + "\n".join(relation_info[:15]))
        
        # 添加原始文本块
        text_chunks = []
        for chunk_id in list(relevant_chunk_ids)[:max_chunks]:
            if chunk_id < len(self.chunks):
                text_chunks.append(f"[文本片段 {chunk_id}]:\n{self.chunks[chunk_id]['text']}")
        
        if text_chunks:
            context_parts.append("\n相关原文:\n" + "\n\n".join(text_chunks))
        
        return "\n\n".join(context_parts)


    def query(self, question):
        """执行GraphRAG查询"""
        # 1. 基于图谱检索
        graph_context = self.retrieve_with_graph(question)
        
        # 2. 构建增强的prompt
        prompt = f"""请基于以下从知识图谱中提取的信息回答问题。
        图谱上下文信息：
        {graph_context}

        问题：{question}

        要求：
        1. 充分利用图谱中提供的实体、关系和原文信息
        2. 如果有多个相关的实体或关系，综合分析它们之间的关联
        3. 回答要结构清晰，有逻辑性
        4. 如果信息不足，请明确指出

        回答："""
        
        # 3. 生成最终回答
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content