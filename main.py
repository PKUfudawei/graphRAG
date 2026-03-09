import os
from openai import OpenAI
from GraphRAG import GraphRAG

if 'OPENROUTER_API_KEY' not in os.environ or 'OPENAI_API_KEY' not in os.environ:
    from dotenv import load_dotenv
    load_dotenv()




def main():
    # 初始化
    client = OpenAI(base_url=os.environ['OPENROUTER_BASE_URL'], api_key=os.environ["OPENROUTER_API_KEY"])
    graph_rag = GraphRAG(client)

    # 构建知识图谱
    graph_rag.build_graph(chunks=chunks)
    
    # 获取用户问题
    default_query = "这本书主要讲了什么？"
    query = input('请输入您的问题: ')
    if len(query) == 0:
        print(f'使用默认问题: {default_query}')
        query = default_query

    # 执行GraphRAG查询
    answer = graph_rag.query(query)
    print(answer)

if __name__ == "__main__":
    main()
