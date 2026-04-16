#!/usr/bin/env python3
import os

LLM = "Qwen/Qwen3.5-27B"
GPUs= ['4', '5', '6', '7']
os.system(f"CUDA_VISIBLE_DEVICES={','.join(GPUs)} vllm serve {LLM} --port 8000 --tensor-parallel-size {len(GPUs)} --max-model-len 262144 --enable-prefix-caching --gpu-memory-utilization 0.8 --max-num-seqs 16 --language-model-only --stream-interval 4 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder")
