#!/bin/bash
# I2 (2026-11-11): DeepSeek V4.1 Flash, data-parallel 8 + expert-parallel, behind vllm-router. Flags from runs/c3.
source /root/.twc-setup/vllm-venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH HF_HOME=/workspace/hf
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_USE_RUST_FRONTEND=1 PYTHONUNBUFFERED=1
if [ -x /workspace/router-venv/bin/vllm-router ]; then
  setsid nohup /workspace/router-venv/bin/vllm-router --host 127.0.0.1 --port 8010 --policy consistent_hash \
    --intra-node-data-parallel-size 8 --worker-urls http://127.0.0.1:8100 --worker-startup-timeout-secs 3600 \
    > /workspace/runs/router.log 2>&1 < /dev/null &
fi
exec vllm serve deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --language-model-only \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}" \
  --max-model-len 1048576 \
  --safetensors-load-strategy prefetch \
  --disable-uvicorn-access-log \
  --host 127.0.0.1 --port 8100
