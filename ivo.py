#!/usr/bin/env python3
"""IVO CLI: fetch the node's verdicts and verify every signature offline.

  ivo.py                      verify against the key the node reports
  ivo.py --pin <NODE_PUB>     verify against a key you pinned earlier (the honest way)
"""
import argparse
import base64
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import twc_common as tc  # noqa: E402

tc.load_env_file(str(HERE / ".env"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default=os.environ.get("NODE_URL", "http://localhost:8000"))
    ap.add_argument("--pin", default=os.environ.get("NODE_PUBKEY"), help="expected node public key")
    ap.add_argument("-n", type=int, default=20, help="show the last N verdicts")
    a = ap.parse_args()
    h = {}
    if os.environ.get("SITE_PASSWORD"):
        h["Authorization"] = "Basic " + base64.b64encode(
            f":{os.environ['SITE_PASSWORD']}".encode()).decode()
    d = httpx.get(a.node.rstrip("/") + "/api/ivo", headers=h).json()
    pub = a.pin or d["node_pub"]
    if a.pin and a.pin != d["node_pub"]:
        print(f"WARNING: node reports key {d['node_pub'][:16]}…, pinned {a.pin[:16]}…")
    vs = d["verdicts"]
    bad = [v for v in vs if not tc.verify(pub, v)]
    print(f"{len(vs)} verdicts, {len(vs) - len(bad)} signatures valid, {len(bad)} INVALID (node key {pub[:16]}…)")
    for v in vs[-a.n:]:
        t = datetime.fromtimestamp(v["issued_at_ms"] / 1000).strftime("%H:%M:%S")
        ok = "sig ok " if tc.verify(pub, v) else "SIG BAD"
        why = v.get("reason") or ", ".join(v["rules"])
        print(f"{t}  {ok}  #{v['window_id']!s:>4}  {v['verdict']:<10} {why}")


if __name__ == "__main__":
    main()
