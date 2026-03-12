import os

GPU = [0,1]
os.system(f"CUDA_VISIBLE_DEVICES={','.join(GPUs)} vllm serve Qwen/Qwen3.5-9B --port 8000 --tensor-parallel-size {len(GPUs)} --max-model-len 2048 --enable-prefix-caching --gpu-memory-utilization 0.8 --max-num-seqs 128 --language-model-only")
