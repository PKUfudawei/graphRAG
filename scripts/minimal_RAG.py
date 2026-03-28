import os
from openai import OpenAI
if 'OPENROUTER_API_KEY' not in os.environ or 'OPENAI_API_KEY' not in os.environ:
    from dotenv import load_dotenv
    load_dotenv()

client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"])

documents = [
    {"id": 1, "text": "Python 是一种高级编程语言。"},
    {"id": 2, "text": "RAG 是通过检索文档增强生成模型的技术。"},
    {"id": 3, "text": "OpenRouter 提供 OpenAI API 兼容的 endpoint。"},
]

query = "RAG 是什么？"

retrieved_docs = [doc for doc in documents if "RAG" in doc["text"]]
context = "\n".join([doc["text"] for doc in retrieved_docs])

prompt = f"请根据以下文档回答问题:\n{context}\n\n问题: {query}\n回答:"

response = client.chat.completions.create(
    model="openrouter/auto",  # OpenRouter 支持的模型
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
)

print(response.choices[0].message.content)
