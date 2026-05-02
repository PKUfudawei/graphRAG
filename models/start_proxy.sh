#!/bin/bash
# WebSearch Proxy 启动脚本

# 设置环境变量
export GOOGLE_API_KEY="${GOOGLE_API_KEY}"
export GOOGLE_CX_ID="${GOOGLE_CX_ID}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000}"

# 检查环境变量
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "错误：请设置 GOOGLE_API_KEY 环境变量"
    echo "示例：export GOOGLE_API_KEY='your-api-key'"
    exit 1
fi

if [ -z "$GOOGLE_CX_ID" ]; then
    echo "错误：请设置 GOOGLE_CX_ID 环境变量"
    echo "示例：export GOOGLE_CX_ID='your-cx-id'"
    exit 1
fi

echo "启动 WebSearch Proxy..."
echo "  vLLM 地址：$VLLM_BASE_URL"
echo "  Proxy 地址：http://localhost:8080"

# 启动代理
cd "$(dirname "$0")"
python websearch_proxy.py
