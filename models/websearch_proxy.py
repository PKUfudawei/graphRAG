#!/usr/bin/env python3
"""
WebSearch Proxy Server

在 vLLM 前面提供一个代理层，拦截 Anthropic API 请求并处理 web_search 工具调用。
支持 Google Custom Search API。

使用方法:
    1. 设置环境变量 GOOGLE_API_KEY 和 GOOGLE_CX_ID
    2. 启动代理：python websearch_proxy.py
    3. 配置 Claude Code 使用代理：ANTHROPIC_BASE_URL=http://localhost:8080
"""

import os
import json
import httpx
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urljoin
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WebSearch Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID", "")

# HTTP 客户端
http_client = httpx.AsyncClient(timeout=120.0)


async def google_search(query: str, num_results: int = 5) -> list[dict]:
    """使用 Google Custom Search API 进行搜索"""
    if not GOOGLE_API_KEY or not GOOGLE_CX_ID:
        logger.warning("Google API Key 或 CX ID 未配置")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX_ID,
        "q": query,
        "num": min(num_results, 10)
    }

    try:
        response = await http_client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })

        return results
    except Exception as e:
        logger.error(f"Google Search error: {e}")
        return []


def format_search_results(results: list[dict]) -> str:
    """格式化搜索结果"""
    if not results:
        return "抱歉，没有找到相关搜索结果。"

    lines = ["### 搜索结果:"]
    for i, result in enumerate(results, 1):
        lines.append(f"\n**{i}. [{result['title']}]({result['url']})**")
        lines.append(f"   {result['snippet']}")

    return "\n".join(lines)


@app.post("/v1/messages")
async def create_message(raw_body: dict[str, Any]):
    """处理 Anthropic API 的消息请求"""

    messages = raw_body.get("messages", [])
    model = raw_body.get("model", "claude-sonnet-4-6")
    max_tokens = raw_body.get("max_tokens", 4096)
    temperature = raw_body.get("temperature", 0.7)
    tools = raw_body.get("tools", [])
    system = raw_body.get("system")

    # 检查是否有 web_search 工具
    has_web_search = any(tool.get("name") == "web_search" for tool in tools)

    # 处理消息内容，转换为 vLLM 格式
    vllm_messages = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # 处理包含 tool_use/tool_result 的内容
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "tool_use":
                        # 处理工具调用 - 如果是 web_search，执行搜索
                        if part.get("name") == "web_search":
                            input_data = part.get("input", {})
                            query = input_data.get("query", "")
                            if query:
                                logger.info(f"执行搜索：{query}")
                                results = await google_search(query)
                                formatted = format_search_results(results)
                                text_parts.append(f"\n[搜索结果]:\n{formatted}\n")
                        else:
                            text_parts.append(f"[tool_use: {part.get('name')}]")
                    elif part.get("type") == "tool_result":
                        text_parts.append(f"[tool_result]: {part.get('content', '')}")
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)
        vllm_messages.append({
            "role": msg.get("role", "user"),
            "content": content
        })

    # 添加 system prompt
    if system:
        vllm_messages.insert(0, {
            "role": "system",
            "content": system
        })

    # 构建 vLLM 请求
    vllm_body = {
        "model": model,
        "messages": vllm_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # 转发到 vLLM
    try:
        response = await http_client.post(
            urljoin(VLLM_BASE_URL, "/v1/chat/completions"),
            json=vllm_body,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        vllm_response = response.json()
        content_text = vllm_response["choices"][0]["message"]["content"]

        # 转换为 Anthropic 格式
        anthropic_response = {
            "id": f"msg_{uuid.uuid4()}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content_text
                }
            ],
            "model": model,
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": vllm_response.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": vllm_response.get("usage", {}).get("completion_tokens", 0)
            }
        }

        return anthropic_response

    except httpx.HTTPError as e:
        logger.error(f"vLLM request failed: {e}")
        raise HTTPException(status_code=502, detail=f"vLLM request failed: {e}")


@app.post("/v1/websearch")
async def websearch(query: str = "", num_results: int = 5):
    """直接调用 websearch 的端点"""
    results = await google_search(query, num_results)
    return {"query": query, "results": results}


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "vllm_url": VLLM_BASE_URL,
        "google_api_configured": bool(GOOGLE_API_KEY and GOOGLE_CX_ID)
    }


if __name__ == "__main__":
    import uvicorn

    if not GOOGLE_API_KEY or not GOOGLE_CX_ID:
        logger.warning("警告：请设置 GOOGLE_API_KEY 和 GOOGLE_CX_ID 环境变量")

    logger.info(f"启动代理服务器，转发到 vLLM: {VLLM_BASE_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
