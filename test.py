import os
from openai import OpenAI

embed_client = OpenAI(
    base_url=os.environ['OPENROUTER_BASE_URL'],
    api_key=os.environ['OPENROUTER_API_KEY']
)
emb = embed_client.embeddings.create(
    model="google/gemini-embedding-001",
    input=["Hello world", "This is a test of the embedding API."]
)
print(emb.data[0].embedding)
