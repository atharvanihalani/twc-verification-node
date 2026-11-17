#!/usr/bin/env bash
# Recording helpers for the 2026-11-11 session (two 8x H200 nodes). Source it, or call: rec.sh <function> [args].
#   rec_begin <run_id> "<notes>"   start logger.py + extras.py, dump nvidia-smi -q, write the first "idle" mark
#   rec_end                         last nvidia-smi dump, stop both loggers
#   train_run <run_id> <minutes> <env assignments...>   one TorchTitan run with idle before and after
#   serve_run <run_id> <server_script> <health_url> <client_url> <metrics_url> "<concs>" [dur_s]
set -u
K=/workspace/serving-traces
export RUNS_ROOT=/workspace/runs PY=/workspace/logger-venv/bin/python HF_HOME=/workspace/hf
export PATH=/root/.twc-setup/vllm-venv/bin:/usr/local/cuda/bin:$PATH
MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
AGENTX=/workspace/agentx-venv/bin/agentx
DATASET=semianalysis-cc-traces-weka-062126

rec_begin() {
  local rid="$1" notes="${2:-}" rd="$RUNS_ROOT/$1"
  "$K/start.sh" "$rid" "$notes" || return 1
  nohup python3 "$K/extras.py" "$rd" > "$rd/extras.log" 2>&1 < /dev/null &
  echo $! > "$RUNS_ROOT/extras.pid"
  nvidia-smi -q > "$rd/nvidia-smi-q.start.txt" 2>&1
  nvidia-smi topo -m > "$rd/nvidia-smi-topo.txt" 2>&1
  "$K/mark.sh" idle
}

rec_end() {
  local rd="$RUNS_ROOT/$(cat "$RUNS_ROOT/CURRENT")"
  nvidia-smi -q > "$rd/nvidia-smi-q.end.txt" 2>&1
  kill -TERM "$(cat "$RUNS_ROOT/extras.pid")" 2>/dev/null
  "$K/stop.sh"
}

# One TorchTitan run: 120 s idle, launch (cold start included), train until <minutes> after launch, stop, 60 s idle.
train_run() {
  local rid="$1" minutes="$2"; shift 2
  local rd="$RUNS_ROOT/$rid" out="/workspace/tt-run/$rid"
  rm -rf "$out"; mkdir -p "$out"
  VLLM_METRICS_URL=http://127.0.0.1:9/metrics rec_begin "$rid" "torchtitan llama3_8b: $*" || return 1
  local cmd="cd /root/.twc-setup/torchtitan && source .venv/bin/activate && env $* MODULE=twc_recipes CONFIG=llama3_8b_twc PYTHONUNBUFFERED=1 ./run_train.sh --output-dir $out"
  echo "$cmd" > "$rd/command.txt"
  (cd /root/.twc-setup/torchtitan && git rev-parse HEAD > "$rd/torchtitan-commit.txt" 2>&1; cp twc_recipes.py "$rd/")
  sleep 120
  "$K/mark.sh" train_start "$*"
  setsid nohup bash -c "$cmd" > "$rd/train.log" 2>&1 < /dev/null &
  sleep $((minutes * 60))
  "$K/mark.sh" train_stop
  pkill -f "[t]orchtitan.train"; sleep 5; pkill -9 -f "[t]orchtitan.train"
  cp -r "$out"/structured_logs "$rd/" 2>/dev/null
  du -sh "$out"/checkpoint/* > "$rd/checkpoint-sizes.txt" 2>&1
  ls -la --time-style=full-iso "$out"/checkpoint >> "$rd/checkpoint-sizes.txt" 2>&1
  "$K/mark.sh" idle_after
  sleep 60
  rec_end
  rm -rf "$out/checkpoint"
  echo "TRAIN DONE $rid"
}

wait_health() {  # <url> <timeout_s>
  local t0=$SECONDS
  until curl -sf "$1" > /dev/null; do
    [ $((SECONDS - t0)) -gt "$2" ] && return 1
    sleep 5
  done
}

# One serving run: 120 s idle, start the server (cold start recorded), then one AgentX point per concurrency.
serve_run() {
  local rid="$1" server="$2" health="$3" client="$4" metrics="$5" concs="$6" dur="${7:-900}"
  local rd="$RUNS_ROOT/$rid" tok
  tok=$(ls -d $HF_HOME/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/* | head -1)
  VLLM_METRICS_URL="$metrics" rec_begin "$rid" "deepseek $server concs=$concs dur=$dur" || return 1
  mkdir -p "$rd/server" "$rd/agentx"; cp "$server" "$rd/server/"
  sleep 120
  "$K/mark.sh" server_start "$(basename "$server")"
  setsid nohup bash "$server" > "$rd/server/serve.log" 2>&1 < /dev/null &
  if ! wait_health "$health" 2400; then "$K/mark.sh" server_failed; tail -30 "$rd/server/serve.log"; serve_stop; rec_end; return 1; fi
  "$K/mark.sh" server_ready
  # The client address may be a router in front of the engine: give it a minute, else fall back to the engine itself.
  local t0=$SECONDS
  until curl -sf "$client/v1/models" > /dev/null || [ $((SECONDS - t0)) -gt 60 ]; do sleep 5; done
  if ! curl -sf "$client/v1/models" > /dev/null && [ -n "${FALLBACK_URL:-}" ]; then
    "$K/mark.sh" note "client address $client is not answering; using $FALLBACK_URL"; client="$FALLBACK_URL"
  fi
  sleep 30
  for c in $concs; do
    "$K/mark.sh" "conc_$c" "agentx $DATASET dur=${dur}s"
    timeout 2400 "$AGENTX" profile --url "$client" --endpoint /v1/chat/completions \
      --model "$MODEL" --tokenizer "$tok" --tokenizer-trust-remote-code \
      --streaming --concurrency "$c" --public-dataset "$DATASET" \
      --benchmark-duration "$dur" --benchmark-grace-period 30 --random-seed 42 \
      --output-artifact-dir "$rd/agentx/c$c" > "$rd/agentx/c$c.log" 2>&1
    echo "conc $c exit=$?" >> "$rd/agentx/status.txt"
  done
  "$K/mark.sh" cooldown; sleep 60
  serve_stop
  "$K/mark.sh" server_stopped; sleep 30
  rec_end
  echo "SERVE DONE $rid"
}

serve_stop() { pkill -f "[v]llm serve"; pkill -f "[v]llm-router"; pkill -f "[p]d_proxy"; sleep 8; pkill -9 -f "[v]llm serve"; pkill -9 -f "VLLM::"; sleep 3; }

if [ "${BASH_SOURCE[0]}" = "$0" ] && [ $# -gt 0 ]; then "$@"; fi
