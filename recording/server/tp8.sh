#!/bin/bash
# I1 (2026-11-11): DeepSeek V4.1 Flash, tensor-parallel across all 8 GPUs. Flags from runs/c1 (InferenceX H200 arm).
source /root/.twc-setup/vllm-venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH HF_HOME=/workspace/hf
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_USE_RUST_FRONTEND=1 PYTHONUNBUFFERED=1
exec vllm serve deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --tensor-parallel-size 8 \
  --language-model-only \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --engram-config "{\"cpu_offload\":true}" \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}" \
  --max-model-len 1048576 \
  --max-cudagraph-capture-size 2048 \
  --safetensors-load-strategy prefetch \
  --disable-uvicorn-access-log \
  --host 127.0.0.1 --port 8001
