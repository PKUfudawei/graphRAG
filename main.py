import os
from openai import OpenAI
from GraphRAG import GraphRAG

if 'OPENROUTER_API_KEY' not in os.environ or 'OPENAI_API_KEY' not in os.environ:
    from dotenv import load_dotenv
    load_dotenv()


def main():
    # 初始化
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"])
    graph_rag = GraphRAG(client)
    
    # 加载并分块文本
    print("正在加载文本...")
    chunks = graph_rag.load_and_chunk_text("./book.txt")
    print(f"文本已切分为 {len(chunks)} 个块")

    # 构建知识图谱
    graph_rag.build_graph(chunks=chunks)
    
    # 获取用户问题
    query = input('请输入您的问题: ')
    if len(query) == 0:
        query = "这本书主要讲了什么？"
        print(f'使用默认问题: {query}')
    
    # 执行GraphRAG查询
    print("\n正在分析问题并检索相关信息...")
    answer = graph_rag.query(query)
    
    print(f"\n{'='*60}")
    print(f"问题: {query}")
    print(f"{'='*60}")
    print(f"回答:\n{answer}")
    print(f"{'='*60}")
    
    # 显示图谱统计信息
    print(f"\n📊 图谱统计:")
    print(f"\t实体数量: {graph_rag.graph.number_of_nodes()}")
    print(f"\t关系数量: {graph_rag.graph.number_of_edges()}")

if __name__ == "__main__":
    main()
