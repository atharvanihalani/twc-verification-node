#!/bin/bash
# I3 driver (2026-11-11). Runs on node 1 (decode node); drives node 0 (prefill node) over ssh.
#   usage: pd.sh "<concs>"
# Loggers run on both nodes under run ids <RID>-prefill and <RID>-decode; every mark is written on both.
source /workspace/serving-traces/rec.sh
CONCS="${1:-16}"; DUR="${2:-900}"
N0IP=10.0.2.12; N1IP=10.0.2.6
n0() { ssh -o BatchMode=yes $N0IP "source /workspace/serving-traces/rec.sh; $*"; }
T=$(date -u +%Y%m%dT%H%MZ); RID=i3-pd-2node-$T
RDD=$RUNS_ROOT/$RID-decode
markboth() { "$K/mark.sh" "$1" "${2:-}"; n0 "\$K/mark.sh '$1' '${2:-}'"; }
stopboth() { pkill -f "[p]d_proxy_twc"; serve_stop; n0 serve_stop; }

n0 "VLLM_METRICS_URL=http://$N0IP:8101/metrics rec_begin $RID-prefill 'I3 prefill node (TP8); decode on node 1' && mkdir -p \$RUNS_ROOT/$RID-prefill/server && cp \$K/server/pd_prefill.sh \$RUNS_ROOT/$RID-prefill/server/" || exit 1
VLLM_METRICS_URL=http://$N1IP:8102/metrics rec_begin $RID-decode "I3 decode node (TP8); prefill on node 0; proxy and AgentX run here" || exit 1
mkdir -p $RDD/server $RDD/agentx; cp $K/server/pd_decode.sh $K/pd_proxy_twc.py $RDD/server/
tok=$(ls -d $HF_HOME/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/* | head -1)
sleep 120
markboth server_start "prefill on node 0, decode on node 1"
n0 "setsid nohup bash \$K/server/pd_prefill.sh > \$RUNS_ROOT/$RID-prefill/server/serve.log 2>&1 < /dev/null &"
setsid nohup bash $K/server/pd_decode.sh > $RDD/server/serve.log 2>&1 < /dev/null &
if ! wait_health http://$N0IP:8101/health 2400 || ! wait_health http://$N1IP:8102/health 2400; then
  markboth server_failed; tail -20 $RDD/server/serve.log; stopboth; n0 rec_end; rec_end; exit 1
fi
markboth server_ready
setsid nohup /root/.twc-setup/vllm-venv/bin/python $K/pd_proxy_twc.py --host 127.0.0.1 --port 8192 \
  --prefiller-hosts $N0IP --prefiller-ports 8101 --decoder-hosts $N1IP --decoder-ports 8102 > $RDD/server/proxy.log 2>&1 < /dev/null &
sleep 10
# One small request through the whole path before the real load.
curl -s --max-time 120 http://127.0.0.1:8192/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":24}" > $RDD/server/smoke.json 2>&1
if grep -q '"choices"' $RDD/server/smoke.json; then markboth pd_smoke_ok; else markboth pd_smoke_failed "see server/smoke.json and proxy.log"; fi
sleep 20
for c in $CONCS; do
  markboth "conc_$c" "agentx $DATASET dur=${DUR}s through the prefill/decode proxy"
  timeout 2400 "$AGENTX" profile --url http://127.0.0.1:8192 --endpoint /v1/chat/completions \
    --model "$MODEL" --tokenizer "$tok" --tokenizer-trust-remote-code \
    --streaming --concurrency "$c" --public-dataset "$DATASET" \
    --benchmark-duration "$DUR" --benchmark-grace-period 30 --random-seed 42 \
    --output-artifact-dir "$RDD/agentx/c$c" > "$RDD/agentx/c$c.log" 2>&1
  echo "conc $c exit=$?" >> "$RDD/agentx/status.txt"
done
markboth cooldown; sleep 60
stopboth
markboth server_stopped; sleep 30
n0 rec_end; rec_end
echo "PD DONE $RID"
