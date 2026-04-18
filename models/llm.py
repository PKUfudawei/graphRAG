from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# 创建 ChatOpenAI 模型实例 (支持 OpenAI 及兼容 API 的服务如 vLLM, Ollama 等)
def get_llm(model="Qwen/Qwen3.5-27B", base_url="http://localhost:8000/v1", api_key="EMPTY", enable_thinking=True, **kwargs):
    """
    Args:
        model: 模型名称
        base_url: API 地址
        api_key: API 密钥
        enable_thinking: 是否启用思考模式，Qwen3.5 默认开启 (True)，设为 False 关闭
        **kwargs: 其他参数传递给 ChatOpenAI
    """
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        extra_body={"enable_thinking": enable_thinking} if enable_thinking is not None else None,
        **kwargs
    )


def get_json_llm(model="Qwen/Qwen3.5-27B", base_url="http://localhost:8000/v1", api_key="EMPTY", enable_thinking=False, **kwargs):
    extra_body = {"response_format": {"type": "json_object"}}
    if enable_thinking is not None:
        extra_body["enable_thinking"] = enable_thinking
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        extra_body=extra_body,
        **kwargs
    )


if __name__ == "__main__":
    llm = get_llm()
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
    json_llm = get_json_llm()
    response = json_llm.invoke([
        SystemMessage(content="You are a JSON API. Always respond with valid JSON."),
        HumanMessage(content="Return the capital of China as JSON with key 'capital'")
    ])
    print(f"JSON Response: {response.content}")
