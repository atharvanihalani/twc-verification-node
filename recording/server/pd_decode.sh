#!/bin/bash
# I3 (2026-11-11): DeepSeek V4.1 Flash split across two nodes. This is the decode server: tensor-parallel across this
# node's 8 GPUs. The cached context is handed from the prefill node to the decode node by NixlConnector (UCX picks
# the transport; cuda_ipc is excluded as in the one-node trial). Flags: server/tp8.sh plus the hand-off settings
# from the 2026-11-10 one-node trial (kv_both, --max-num-seqs 32).
source /root/.twc-setup/vllm-venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH HF_HOME=/workspace/hf
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_USE_RUST_FRONTEND=1 PYTHONUNBUFFERED=1
export UCX_TLS=^cuda_ipc VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.2.6 VLLM_NIXL_SIDE_CHANNEL_PORT=5700
exec vllm serve deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --tensor-parallel-size 8 \
  --language-model-only \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --engram-config "{\"cpu_offload\":true}" \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}" \
  --max-model-len 1048576 \
  --max-num-seqs 32 \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}" \
  --safetensors-load-strategy prefetch \
  --disable-uvicorn-access-log \
  --host 10.0.2.6 --port 8102
