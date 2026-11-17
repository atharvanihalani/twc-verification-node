# Recording scripts

The scripts that made the recordings the Replay demo plays. They ran on rented 8×H200 nodes: two nodes on 2026-11-11 and one node on 2026-11-16. They are copied here as they ran, so readers can check how each workload was started and what was logged. The paths in them (`/workspace/...`) are the rented boxes' paths.

| File | What |
|---|---|
| `rec.sh` | Recording helpers: `train_run` (one TorchTitan run, with idle time before and after) and `serve_run` (a vLLM server plus the AgentX load at each concurrency) |
| `server/tp8.sh`, `server/ep8.sh` | The vLLM server for tensor-parallel and expert-parallel serving (DeepSeek V4.1 Flash, vLLM 0.33.0) |
| `server/pd_prefill.sh`, `server/pd_decode.sh`, `pd.sh`, `pd_proxy_twc.py` | The two-node prefill/decode split: one server per node, the driver, and the proxy in front (adapted from vLLM's example proxy, Apache-2.0) |
| `logger.py`, `start.sh`, `stop.sh`, `mark.sh` | The telemetry logger (NVML at 20 Hz, plus the server's own metrics) and the marks that split a recording into phases |
| `extras.py` | Once-a-second side readings: network and RDMA port counters, disk, per-link NVLink, which process holds which GPU |

The TorchTitan training recipe is in [`../setup.py`](../setup.py) (`TT_RECIPE`). [`../make_replay_data.py`](../make_replay_data.py) turns the recordings into the page's data and runs the rules over them.
