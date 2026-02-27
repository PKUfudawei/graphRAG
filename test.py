import os
import litellm

response = litellm.embedding(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="text-embedding-3-small",   # OpenAI embedding 模型
    input=["Hello world", "Test text"],
    encoding_format="float"           # 必须指定 "float" 或 "base64"
)

print(response)

