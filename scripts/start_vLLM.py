#!/usr/bin/env python3
import os

LLM = "Qwen/Qwen3.5-9B"
GPUs= ['2', '3']
os.system(f"CUDA_VISIBLE_DEVICES={','.join(GPUs)} vllm serve {LLM} --port 8000 --tensor-parallel-size {len(GPUs)} --max-model-len 4096 --enable-prefix-caching --gpu-memory-utilization 0.6 --max-num-seqs 128 --language-model-only --stream-interval 4")

