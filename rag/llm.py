from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# 创建 ChatOpenAI 模型实例 (支持 OpenAI 及兼容 API 的服务如 vLLM, Ollama 等)
llm = ChatOpenAI(
    model="Qwen/Qwen3.5-27B",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0.5
)

json_llm = ChatOpenAI(
    model="Qwen/Qwen3.5-27B",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    extra_body={"response_format": {"type": "json_object"}},
    temperature=0.5
)


if __name__ == "__main__":
    # 基本用法：使用 invoke() 方法
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of China?")
    ])
    print(f"Response: {response.content}")

    # 无 system prompt 的用法
    response = llm.invoke([
        HumanMessage(content="Explain gravity in one sentence.")
    ])
    print(f"Response: {response.content}")

    # 使用 JSON 响应格式
    response = json_llm.invoke([
        SystemMessage(content="You are a JSON API. Always respond with valid JSON."),
        HumanMessage(content="Return the capital of China as JSON with key 'capital'")
    ])
    print(f"JSON Response: {response.content}")
