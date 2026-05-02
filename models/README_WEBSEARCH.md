# WebSearch Proxy 使用说明

## 概述

这个代理服务器在 vLLM 前面提供一层，拦截 Anthropic API 请求并处理 web_search 工具调用。

## 架构

```
Claude Code -> WebSearch Proxy (8080) -> vLLM (8000)
                     |
                     v
              Google Custom Search API
```

## 配置步骤

### 1. 获取 Google Custom Search API 凭证

1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 创建新项目或选择现有项目
3. 启用 "Custom Search API"
4. 创建 API Key
5. 访问 [Custom Search Engine](https://cse.google.com/cse/all) 创建搜索引擎
6. 获取 CX ID

### 2. 设置环境变量

```bash
export GOOGLE_API_KEY="your-api-key"
export GOOGLE_CX_ID="your-cx-id"
```

### 3. 安装依赖

```bash
cd /data/fudawei/graphRAG
uv add fastapi uvicorn
```

### 4. 启动 vLLM (如果还没启动)

```bash
./models/deploy_vllm.sh
```

### 5. 启动 Proxy

```bash
./models/start_proxy.sh
```

### 6. 配置 Claude Code

修改 `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8080",
    "ANTHROPIC_MODEL": "claude-sonnet-4-6"
  }
}
```

## 测试

```bash
# 检查代理状态
curl http://localhost:8080/health

# 直接测试搜索
curl "http://localhost:8080/v1/websearch?query=python+programming"
```

## 注意事项

1. Google Custom Search API 有配额限制（每天 100 次免费）
2. 代理服务器会记录所有搜索请求到日志
3. 确保 vLLM 在代理之前启动
