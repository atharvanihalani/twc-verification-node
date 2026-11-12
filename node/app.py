"""Verification node (v0).

- WS  /ingest      prover streams signed, hash-chained, nonce-bound windows
- GET /node        parked: redirects to /replay (the page is at ../parked-pages/node.html, not in the image)
- GET /ivo         parked: redirects to /replay (the page is at ../parked-pages/ivo.html, not in the image)
- GET /api/state   JSON that the parked /node page read
- GET /api/ivo     JSON behind /ivo (+ node public key, so the IVO can verify)
- POST /api/reset  clear history (demo convenience)

Everything sits behind a password (no username) when SITE_PASSWORD is set.
State is in memory: a restart forgets everything.
"""
import asyncio
import base64
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # twc_common lives one level up (and in /app in Docker)
sys.path.insert(0, str(HERE))
import twc_common as tc  # noqa: E402
import rules  # noqa: E402

tc.load_env_file(str(HERE.parent / ".env"))

WINDOW_S = float(os.environ.get("WINDOW_S", "5"))
NONCE_MAX_AGE_S = 2.5 * WINDOW_S
BUFFER_S = 120.0
ENERGY_TOL = 0.15          # |dE - integral(P)| / integral(P)
ENERGY_MIN_J = 50.0        # ignore tiny intervals
PROVER_PUBKEY = os.environ.get("PROVER_PUBKEY")  # pinned; if unset, trust on first use
SITE_PASSWORD = os.environ.get("SITE_PASSWORD")

NODE_KEY = tc.key_from_seed_b64(os.environ["NODE_KEY"]) if os.environ.get("NODE_KEY") else tc.new_key()
NODE_PUB = tc.pub_b64(NODE_KEY)


class State:
    def __init__(self):
        self.reset_all()

    def reset_all(self):
        self.verdicts = deque(maxlen=1000)   # signed, IVO-visible
        self.events = deque(maxlen=300)      # node-visible log
        self.next_verdict_seq = 0
        self.reset_session()

    def reset_session(self):
        self.connected = False
        self.prover_id = None
        self.prover_pub = None
        self.last_hash = None
        self.buffers = {}                    # gpu -> deque[(t, p)]
        self.energy = {}                     # gpu -> (t, mJ)
        self.declarations = []
        self.issued = {}                     # window_id -> (nonce, issued_at)
        self.used = set()
        self.last_eval = None
        self.windows_ok = 0
        self.windows_rejected = 0


S = State()
app = FastAPI()
# The Replay demo's 10 Hz recordings are up to 5 MB of JSON each; compressed they are about a quarter of that.
app.add_middleware(GZipMiddleware, minimum_size=20_000)


# ---------------------------------------------------------------- auth (ASGI)
# Password only, no username. Browsers get a login page that sets a cookie;
# the CLIs send HTTP basic auth (username ignored). /healthz and /login are open.
AUTH_COOKIE = "twc_auth"
OPEN_PATHS = {"/healthz", "/login"}


def cookie_token():
    return hashlib.sha256(f"twc-auth:{SITE_PASSWORD}".encode()).hexdigest()


class Auth:
    def __init__(self, inner):
        self.inner = inner

    def _ok(self, scope):
        if not SITE_PASSWORD or scope.get("path") in OPEN_PATHS:
            return True
        for k, v in scope.get("headers", []):
            if k == b"authorization" and v[:6].lower() == b"basic ":
                try:
                    _, _, pw = base64.b64decode(v[6:]).decode().partition(":")
                except Exception:
                    return False
                if secrets.compare_digest(pw, SITE_PASSWORD):
                    return True
            if k == b"cookie":
                for part in v.decode(errors="ignore").split(";"):
                    name, _, val = part.strip().partition("=")
                    if name == AUTH_COOKIE and secrets.compare_digest(val, cookie_token()):
                        return True
        return False

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or self._ok(scope):
            return await self.inner(scope, receive, send)
        if scope["type"] == "http":
            path = scope.get("path", "/")
            if path.startswith("/api/"):
                status, headers, body = 401, [(b"content-type", b"text/plain")], b"auth required"
            else:
                status, body = 303, b""
                headers = [(b"location", ("/login?next=" + quote(path)).encode())]
            await send({"type": "http.response.start", "status": status, "headers": headers})
            await send({"type": "http.response.body", "body": body})
        else:
            await receive()  # websocket.connect
            await send({"type": "websocket.close", "code": 1008})


LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>TWC demo</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light dark;--bg:#fafaf9;--fg:#1c1917;--mute:#78716c;--line:#d6d3d1;--err:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mute:#a8a29e;--line:#44403c;--err:#f87171}}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--fg);
font:15px/1.4 ui-sans-serif,system-ui,-apple-system,sans-serif}
form{display:flex;gap:8px;align-items:center}
input{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:transparent;color:inherit;width:200px}
button{font:inherit;padding:8px 14px;border:1px solid var(--line);border-radius:6px;background:var(--fg);color:var(--bg);cursor:pointer}
p{margin:8px 0 0;color:var(--err);font-size:13px;min-height:1em}
</style></head><body><div>
<form method="post" action="/login"><input type="hidden" name="next" value="__NEXT__">
<input type="password" name="password" placeholder="password" autofocus autocomplete="current-password">
<button>Enter</button></form><p>__ERR__</p></div></body></html>"""


def login_page(next_path, err=""):
    from html import escape
    return HTMLResponse(LOGIN_HTML.replace("__NEXT__", escape(next_path)).replace("__ERR__", escape(err)),
                        status_code=401 if err else 200)


def safe_next(n):
    return n if n.startswith("/") and not n.startswith("//") else "/"


# ---------------------------------------------------------------- helpers
def log(kind, text, **extra):
    S.events.append({"t": time.time(), "kind": kind, "text": text, **extra})


def emit_verdict(window_id, status, rules_hit, reason=None):
    body = {
        "seq": S.next_verdict_seq,
        "issued_at_ms": int(time.time() * 1000),   # int: keeps canonical JSON identical in Python and JS
        "prover_id": S.prover_id,
        "window_id": window_id,
        "verdict": status,                    # compliant | flagged | rejected | warming_up
        "rules": sorted(set(rules_hit)),      # rule names only: no telemetry leaves the node
    }
    if reason:
        body["reason"] = reason
    S.next_verdict_seq += 1
    S.verdicts.append(tc.sign(NODE_KEY, body))


def check_chain_and_sig(msg):
    if not tc.verify(S.prover_pub, msg):
        return "bad signature"
    if msg.get("prev_hash") != S.last_hash:
        return "hash chain broken (prev_hash does not match last accepted message)"
    return None


def check_nonce(msg):
    wid = msg.get("window_id")
    if wid not in S.issued:
        return f"nonce for unknown window {wid}"
    nonce, at = S.issued[wid]
    if msg.get("nonce") != nonce:
        return "nonce mismatch (not the nonce this node issued for this window)"
    if wid in S.used:
        return "nonce already used"
    if time.time() - at > NONCE_MAX_AGE_S:
        return "nonce expired"
    return None


def integrate_power(g, t0, t1):
    pts = [(t, p) for t, p in S.buffers.get(g, ()) if t0 <= t <= t1]
    if len(pts) < 2:
        return None
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))


def ingest_window(msg):
    t_last = None
    for g_str, pts in msg["samples"].items():
        g = int(g_str)
        buf = S.buffers.setdefault(g, deque())
        for t, p in pts:
            buf.append((t, p))
            t_last = t if t_last is None else max(t_last, t)
        while buf and buf[0][0] < buf[-1][0] - BUFFER_S:
            buf.popleft()
    # cumulative energy counter: must be monotonic and match integral(P dt)
    problems = []
    for g_str, (te, e_mj) in (msg.get("energy") or {}).items():
        g = int(g_str)
        prev = S.energy.get(g)
        S.energy[g] = (te, e_mj)
        if prev is None:
            continue
        dE = (e_mj - prev[1]) / 1000.0
        if dE < 0:
            problems.append(f"GPU {g}: energy counter went backwards")
            continue
        ip = integrate_power(g, prev[0], te)
        if ip and ip > ENERGY_MIN_J and abs(dE - ip) / ip > ENERGY_TOL:
            problems.append(f"GPU {g}: counter says {dE:.0f} J, power trace says {ip:.0f} J")
    return t_last, problems


# ---------------------------------------------------------------- ingest
@app.websocket("/ingest")
async def ingest(ws: WebSocket):
    await ws.accept()
    hello = await ws.receive_json()
    pub = hello.get("pub")
    if hello.get("type") != "hello" or not pub or not tc.verify(pub, hello):
        await ws.close(code=1008)
        return
    if PROVER_PUBKEY and pub != PROVER_PUBKEY:
        log("reject", f"unknown prover key {pub[:12]}…")
        await ws.send_json({"type": "error", "error": "prover key not pinned on this node"})
        await ws.close(code=1008)
        return
    S.reset_session()
    S.connected, S.prover_id, S.prover_pub = True, hello.get("prover_id", "?"), pub
    S.last_hash = tc.msg_hash(hello)
    log("session", f"prover '{S.prover_id}' connected (key {pub[:12]}…)")
    await ws.send_json({"type": "welcome", "node_pub": NODE_PUB, "window_s": WINDOW_S})

    async def nonce_pump():
        wid = 0
        while True:
            nonce = secrets.token_hex(16)
            S.issued[wid] = (nonce, time.time())
            for old in [w for w in S.issued if w < wid - 10]:
                S.issued.pop(old, None)
            await ws.send_json({"type": "nonce", "window_id": wid, "nonce": nonce})
            wid += 1
            await asyncio.sleep(WINDOW_S)

    pump = asyncio.create_task(nonce_pump())
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            errs = [e for e in (check_chain_and_sig(msg),
                                check_nonce(msg) if kind == "window" else None) if e]
            err = "; ".join(errs)
            if err:
                S.windows_rejected += 1
                log("reject", f"{kind} {msg.get('window_id', '')}: {err}")
                emit_verdict(msg.get("window_id"), "rejected", [], reason=err)
                continue
            S.last_hash = tc.msg_hash(msg)
            if kind == "declare":
                S.declarations = msg["declarations"]
                pretty = "; ".join(f"{d['job_id']}={d['type']}@GPU{min(d['gpus'])}-{max(d['gpus'])}"
                                   for d in S.declarations)
                log("declare", f"declarations: {pretty}")
            elif kind == "window":
                S.used.add(msg["window_id"])
                t_last, problems = ingest_window(msg)
                S.windows_ok += 1
                if t_last is None:  # empty window (e.g. nonce arrived before any samples)
                    emit_verdict(msg["window_id"], "warming_up", [])
                    continue
                gpus = sorted(S.buffers)
                try:
                    ev = rules.evaluate(S.buffers, S.declarations, gpus, t_last)
                except Exception as e:  # never let one bad window kill the session
                    log("integrity", f"rule evaluation failed: {e!r}")
                    emit_verdict(msg["window_id"], "rejected", [], reason="evaluation error")
                    continue
                S.last_eval = {"window_id": msg["window_id"], **ev}
                hit_names = [h["rule"] for h in ev["hits"]]
                if problems:
                    hit_names.append("energy_mismatch")
                    for p in problems:
                        log("integrity", p)
                for h in ev["hits"]:
                    log("rule", f"{h['rule']}: {h['detail']}")
                status = "flagged" if hit_names else ev["status"]
                emit_verdict(msg["window_id"], status, hit_names)
    except WebSocketDisconnect:
        pass
    finally:
        pump.cancel()
        S.connected = False
        log("session", f"prover '{S.prover_id}' disconnected")


# ---------------------------------------------------------------- views + API
@app.get("/")
def root():
    return RedirectResponse("/overview")   # overview essay = front door (2026-11-08)


@app.get("/overview")
def overview_view():
    """Overview: short essay on what the demo is and why, with a link into the demo."""
    return page("overview.html")


@app.get("/login")
def login_get(next: str = "/"):
    return login_page(safe_next(next))


@app.post("/login")
async def login_post(request: Request):
    form = parse_qs((await request.body()).decode())
    pw = (form.get("password") or [""])[0]
    nxt = safe_next((form.get("next") or ["/"])[0])
    if not SITE_PASSWORD or not secrets.compare_digest(pw, SITE_PASSWORD):
        return login_page(nxt, "Wrong password.")
    r = RedirectResponse(nxt, status_code=303)
    r.set_cookie(AUTH_COOKIE, cookie_token(), max_age=30 * 86400, httponly=True, samesite="lax",
                 secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https")
    return r


@app.get("/healthz")
def healthz():
    return {"ok": True}


NO_CACHE = {"Cache-Control": "no-store"}  # prototyping: always serve the latest page


def page(name):
    return FileResponse(HERE / "static" / name, headers=NO_CACHE)


@app.get("/walkthrough")
def walkthrough_view():
    """Walkthrough: install, run workloads, connect the scheduler, try to evade the checker."""
    return page("walkthrough.html")


@app.get("/setup")
def setup_redirect():
    return RedirectResponse("/walkthrough")


@app.get("/node")
def node_view():
    # Parked 2026-11-11 (see TODOS.md): the old signed-prover page is at ../parked-pages/node.html, outside the
    # image, so it is not reachable online. To bring it back: move it to static/node.html and return page("node.html").
    return RedirectResponse("/replay", status_code=303)


@app.get("/ivo")
def ivo_view():
    # Parked (see TODOS.md): the page is at ../parked-pages/ivo.html, outside the image, so it is not reachable online.
    # It still showed the old signed-prover path, not live data. To bring it back: move it to static/ivo.html,
    # return page("ivo.html") here and restore the tab in static/nav.js.
    return RedirectResponse("/replay", status_code=303)


@app.get("/replay")
def replay_view():
    """Replay demo. For now this is the settled MOCK-UP (made-up data, nothing wired), his call 2026-11-11.
    It lives in node/pages/, not node/static/, so /replay is its only address. The page that drew live
    declarations + rule results is parked at ../parked-pages/replay-live.html (not in the image). To bring that
    back: move it to static/replay.html and return page("replay.html")."""
    return FileResponse(HERE / "pages" / "replay.html", headers=NO_CACHE)


@app.get("/api/scenario")
def api_scenario():
    # built offline by make_scenario.py (the real rules run over recorded traces); the page plays it back
    return FileResponse(HERE.parent / "replays" / "scenario-1.json", media_type="application/json", headers=NO_CACHE)


@app.get("/api/state")
def api_state():
    now_t = max((b[-1][0] for b in S.buffers.values() if b), default=None)
    gpus = {}
    for g, buf in sorted(S.buffers.items()):
        pts = [(round(t, 2), round(p, 1)) for t, p in buf if now_t is None or t > now_t - 60]
        gpus[g] = pts[::2]  # 5 Hz is plenty for the sparkline
    return JSONResponse({
        "connected": S.connected, "prover_id": S.prover_id, "now_t": now_t,
        "window_s": WINDOW_S, "windows_ok": S.windows_ok, "windows_rejected": S.windows_rejected,
        "declarations": S.declarations, "gpus": gpus, "last_eval": S.last_eval,
        "events": list(S.events)[-40:], "rule_text": rules.RULE_TEXT,
    })


@app.get("/api/ivo")
def api_ivo():
    return {"node_pub": NODE_PUB, "verdicts": list(S.verdicts)}


@app.post("/api/reset")
def api_reset():
    S.reset_all()
    return {"ok": True}


# ---------------------------------------------------------------- live collector (v0: unsigned)
# collector.py POSTs a batch of raw samples every second. No signatures, nonces or
# declarations yet: this is just the telemetry pipe. Kept separate from the prover session.
LIVE_KEEP_S = 120
LIVE = {}   # host -> {"slot", "gpus": {idx: name}, "last_seen": t, "batches": n, "rows": {gpu: deque[row]}, "hello"}
MAX_BOXES = 8          # boxes shown at once; each gets a fixed slot (= its colour on the page). 8 so a
                      # 4-way --split box still leaves room for other boxes / the replay
EVICT_SILENT_S = 60    # a new box may take the slot of one that has been silent this long
EXPIRE_SILENT_S = 300  # …and a box silent this long drops off the page by itself
LIVE_EVENTS = deque(maxlen=200)   # connection log for the Raw data page: hellos, refusals, evictions
_REFUSED_AT = {}


def live_event(kind, host, text, **extra):
    LIVE_EVENTS.append({"t": time.time(), "kind": kind, "host": host, "text": text, **extra})


def short_gpu(name):
    """'NVIDIA H100 80GB HBM3' -> 'H100', 'NVIDIA H100 NVL' -> 'H100 NVL'."""
    return re.sub(r"\s+\d+GB(\s+HBM\w*)?$", "", (name or "GPU").replace("NVIDIA ", "")).strip()


def box_label(h):
    """'8× H200', or '4× H100 + 2× A100' for a mixed box."""
    names = [short_gpu(n) for _, n in sorted(h["gpus"].items(), key=lambda kv: int(kv[0]))]
    counts = {n: names.count(n) for n in dict.fromkeys(names)}
    label = " + ".join(f"{c}× {n}" for n, c in counts.items()) or "no GPUs yet"
    sp = (h.get("hello") or {}).get("split")
    if isinstance(sp, dict):   # collector.py --split: one physical box pretending to be several nodes
        label += f" · part {sp.get('part')} of {sp.get('of')} on one box"
    return label


def uuids(h):
    gs = (h.get("hello") or {}).get("gpus")
    return {g["uuid"] for g in gs if isinstance(g, dict) and g.get("uuid")} if isinstance(gs, list) else set()


def double_counted(host):
    """Other live hosts reporting any of the same GPUs (by UUID): {other_host: n_shared}.
    In a split-box prototype this is a config slip; in a real deployment it's a GPU counted twice."""
    mine = uuids(LIVE.get(host, {}))
    return {o: len(mine & uuids(oh)) for o, oh in LIVE.items() if o != host and mine & uuids(oh)}


def machine(host, h):
    """The physical box an entry belongs to: a split part names its box in the hello."""
    hello = h.get("hello") or {}
    sp = hello.get("split") if isinstance(hello.get("split"), dict) else {}
    return sp.get("physical_host") or hello.get("hostname") or host


def retire_replaced(host):
    """`collector.py split 4+4` (or back to one) re-registers the same box under new names. The old entries
    cover the same GPUs on the same machine, so drop them at once: the page should not show a box twice, and
    this is not the 'GPU counted twice' case, which is two different machines claiming one GPU."""
    me = LIVE.get(host)
    if not me:
        return
    for other in list(double_counted(host)):
        if machine(other, LIVE[other]) == machine(host, me):
            live_event("replaced", other, f"this box now reports as '{host}'; old entry dropped")
            LIVE.pop(other)


def admit(host, now):
    """The box's entry in LIVE, creating it if there's a free slot. Returns (entry, error)."""
    if host in LIVE:
        return LIVE[host], None
    if len(LIVE) >= MAX_BOXES:
        stale = [x for x in LIVE if now - LIVE[x]["last_seen"] > EVICT_SILENT_S]
        if not stale:
            err = f"this node already shows {MAX_BOXES} boxes ({', '.join(LIVE)}); stop one of them first"
            if now - _REFUSED_AT.get(host, 0) > 60:   # log a refused box once a minute, not every second
                _REFUSED_AT[host] = now
                live_event("refused", host, err)
            return None, err
        gone = min(stale, key=lambda x: LIVE[x]["last_seen"])
        live_event("evicted", gone, f"silent for {now - LIVE[gone]['last_seen']:.0f} s; slot given to '{host}'")
        LIVE.pop(gone)
    slot = min(set(range(MAX_BOXES)) - {x["slot"] for x in LIVE.values()})
    LIVE[host] = {"slot": slot, "gpus": {}, "rows": {}, "batches": 0, "recv": deque(maxlen=600),
                  "last_seen": now, "hello": None}
    return LIVE[host], None


@app.post("/api/hello")
async def api_hello(request: Request):
    """A box introduces itself before its first batch: GPUs, driver, CPU, OS…
    collector.py sends it again whenever the node says it doesn't know the box (e.g. after a restart)."""
    d = json.loads(await request.body())
    host, now = str(d.get("host", "?")), time.time()
    h, err = admit(host, now)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=429)
    ip = request.headers.get("fly-client-ip") or (request.client.host if request.client else None)
    h["hello"] = {**d, "received_at": now, "from_ip": ip}
    if isinstance(d.get("gpus"), list):
        h["gpus"] = {str(g.get("index", i)): g.get("name") for i, g in enumerate(d["gpus"])}
    bits = [box_label(h)] + [f"{k} {d[k]}" for k in ("driver", "cuda") if d.get(k)] + ([d["collector"]] if d.get("collector") else [])
    live_event("hello", host, " · ".join(bits), info=h["hello"])
    retire_replaced(host)
    for other, n in double_counted(host).items():
        live_event("conflict", host, f"{n} GPU(s) also reported by '{other}' (same UUID): counted twice")
    return {"ok": True, "slot": h["slot"]}


@app.post("/api/collect")
async def api_collect(request: Request):
    raw = await request.body()
    d = json.loads(raw)
    now = time.time()
    h, err = admit(str(d.get("host", "?")), now)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=429)
    h["gpus"], h["last_seen"] = d.get("gpus", {}), now
    h["batches"] += 1
    # drop anything that isn't a finite number (NaN/inf would break every JSON response later)
    samples = [{k: v for k, v in r.items() if not (isinstance(v, float) and not math.isfinite(v))}
               for r in d.get("samples", []) if isinstance(r, dict) and "t" in r and "gpu" in r]
    newest = max((r["t"] for r in samples), default=now)
    # ingest bookkeeping for the debug page: arrival time, size, how far behind the box's clock we are
    h["recv"].append((now, len(raw), len(samples), now - newest))
    h["last_batch"] = {**d, "samples": samples[:12], "n_samples": len(samples), "bytes": len(raw)}
    for r in samples:
        q = h["rows"].setdefault(int(r["gpu"]), deque())
        q.append(r)
        while q and q[0]["t"] < r["t"] - LIVE_KEEP_S:
            q.popleft()
    return {"ok": True, "need_hello": h["hello"] is None}   # e.g. the node restarted: please introduce yourself again


# ---------------------------------------------------------------- declarations (Slurm)
# slurm_reporter.py POSTs the cluster's whole job table every few seconds. Each job's declaration
# is its comment (declared:<type>). The node remembers when it FIRST saw each job (commitment
# timing) and every version of its declaration (users can rewrite a running job's comment).
# The reports are the operator's claims; checking them against telemetry is the rules' job.
SLURM = {}            # cluster -> {"reporter", "received_at", "seq", "missed", "raw", "error", "jobs": {id: job}}
SLURM_KEEP_S = 7200   # forget jobs this long after they left Slurm's table


@app.post("/api/declare")
async def api_declare(request: Request):
    d = json.loads(await request.body())
    now = time.time()
    cluster = str(d.get("cluster") or "slurm")
    c = SLURM.setdefault(cluster, {"jobs": {}, "seq": 0, "missed": 0, "reports": 0, "first_report_at": now})
    seq = int(d.get("seq") or 0)
    if c["seq"] and seq > c["seq"] + 1:
        c["missed"] += seq - c["seq"] - 1
    if c["seq"] and seq == 1:
        live_event("reporter", cluster, "Slurm reporter restarted")
    ip = request.headers.get("fly-client-ip") or (request.client.host if request.client else None)
    c.update(seq=seq, received_at=now, sent_at=d.get("sent_at"), poll_s=d.get("poll_s"), raw=d.get("raw", ""),
             error=d.get("error"), reporter=d.get("reporter"), reporter_host=d.get("reporter_host"), from_ip=ip,
             reports=c["reports"] + 1)
    seen = set()
    for j in d.get("jobs") or []:
        if not isinstance(j, dict) or not j.get("id"):
            continue
        jid = str(j["id"]); seen.add(jid)
        old = c["jobs"].get(jid)
        if old is None:
            j.update(first_seen=now, history=[{"t": now, "comment": j.get("comment", "")}], gone_at=None)
            what = f"declared {j['declared']}" if j.get("declared") else "NOT declared"
            live_event("job", cluster, f"job {jid} '{j.get('name')}' first seen ({j.get('state', '').lower()}): {what}")
        else:
            j.update(first_seen=old["first_seen"], history=old["history"], gone_at=None)
            if j.get("comment", "") != old.get("comment", ""):
                j["history"].append({"t": now, "comment": j.get("comment", "")})
                live_event("job", cluster, f"job {jid}: declaration changed '{old.get('comment')}' -> '{j.get('comment')}'")
            if j.get("state") != old.get("state"):
                live_event("job", cluster, f"job {jid}: {old.get('state', '?').lower()} -> {j.get('state', '?').lower()}")
            if not j.get("gpus") and old.get("gpus"):   # Slurm drops the GPU indices once a job ends: keep the last ones
                j["gpus"], j["gpus_from_earlier"] = old["gpus"], True
        c["jobs"][jid] = j
    for jid, j in list(c["jobs"].items()):
        if jid not in seen:
            if j.get("gone_at") is None:
                j["gone_at"] = now
            elif now - j["gone_at"] > SLURM_KEEP_S:
                c["jobs"].pop(jid)
    return {"ok": True, "jobs": len(seen)}


def gpu_lookup():
    """(slurm node name, gres index) -> the streaming GPU it is. Slurm counts GPUs by /dev/nvidiaN (the
    'minor number'); collector.py v0.4+ reports each GPU's minor, older ones only their NVML index."""
    out = {}
    for host, h in LIVE.items():
        hello = h.get("hello") or {}
        sp = hello.get("split") if isinstance(hello.get("split"), dict) else {}
        machine = sp.get("physical_host") or hello.get("hostname") or host
        for g in hello.get("gpus") or []:
            if not isinstance(g, dict):
                continue
            n = g.get("minor")
            if n is None:
                n = g.get("physical_index", g.get("index"))
            out[(machine, n)] = {"host": host, "gpu": g.get("index"), "uuid": g.get("uuid"), "by": "minor" if g.get("minor") is not None else "index"}
    return out


# ---------------------------------------------------------------- rules on live data
# Every RULES_EVERY_S the node runs the rules (node/rules/) on the trailing 60 s of every GPU that is streaming,
# against the Slurm declarations that overlap that window. GPU names = "<host>:<local gpu>".
# A running job with no declared type covers nothing, so its load counts as undeclared.
# Hits are merged into violation intervals (rule + subject, start/end) for the Replay page.
RULES_EVERY_S = 10
FRESH_S = 10                       # a GPU whose newest sample is older than this is left out
LIVE_RULES = {"last": None, "open": {}, "closed": deque(maxlen=200)}
SHOWN_RULES = tuple(rules.RULE_TEXT)


def job_running_at(j, t, window=0.0):
    """Did job j hold its GPUs at any time in [t - window, t]?"""
    if j.get("start") is None or j.get("start_is_estimate"):
        return False
    done = j.get("gone_at") or j.get("state") not in ("PENDING", "RUNNING", "COMPLETING", "SUSPENDED", "CONFIGURING")
    end = (j.get("end") or j.get("gone_at")) if done else None
    return j["start"] <= t and (end is None or end > t - window)


def live_job_gpus(j, look):
    return [f"{m['host']}:{m['gpu']}" for a in j.get("gpus") or [] for i in a.get("idx") or []
            if (m := look.get((a.get("node"), i)))]


def evaluate_live():
    now, look = time.time(), gpu_lookup()
    buffers, newest, models = {}, [], {}
    readings = {"pcie_sent": {}, "tensor_util": {}}   # the once-a-second counters the rules read
    for host, h in LIVE.items():
        for g, q in h["rows"].items():
            pts = [(r["t"], r["power_w"]) for r in q if r.get("power_w") is not None]
            if pts and now - pts[-1][0] < FRESH_S + 5:      # box clocks can trail the node by a few s
                buffers[f"{host}:{g}"] = pts
                models[f"{host}:{g}"] = h["gpus"].get(str(g))
                for name, field in (("pcie_sent", "pcie_tx_gb_s"), ("tensor_util", "tensor_util_pct")):
                    readings[name][f"{host}:{g}"] = [(r["t"], r[field]) for r in q if r.get(field) is not None]
                newest.append(pts[-1][0])
    if not buffers:
        return None
    t_end = min(newest)                                    # every GPU has data up to here
    decls, undeclared_jobs = [], {}
    for c in SLURM.values():
        for j in c["jobs"].values():
            if not job_running_at(j, t_end, rules.WINDOW_S):
                continue
            keys = live_job_gpus(j, look)
            if j.get("declared"):
                decls.append({"job_id": j["id"], "type": j["declared"], "gpus": keys})
            else:
                undeclared_jobs.update({k: j["id"] for k in keys})
    gpus = sorted(buffers)
    ev = rules.evaluate(buffers, decls, gpus, t_end, readings, models)
    hits = []
    for hit in ev["hits"]:
        if hit["rule"] == "undeclared_load":      # one GPU; the page groups these by box
            g = hit["gpus"][0]
            hit = {**hit, "gpu": g, "host": g.rsplit(":", 1)[0], "job": undeclared_jobs.get(g)}
        elif hit["rule"] == "training_shaped":    # one job
            hit = {**hit, "job": hit["jobs"][0]}
        hits.append(hit)                          # boundary_traffic: hit["jobs"] is already the pair
    jobs_gpus = {d["job_id"]: d["gpus"] for d in decls}
    return {"t": t_end, "evaluated_at": now, "status": ev["status"], "hits": hits, "gpus": gpus,
            "declarations": [{k: d[k] for k in ("job_id", "type")} | {"n_gpus": len(d["gpus"])} for d in decls],
            "features": ev.get("features", {}), "jobs_gpus": jobs_gpus}


def track_violations(ev):
    """Merge this evaluation's hits into open intervals; close the ones that stopped."""
    op, seen = LIVE_RULES["open"], set()
    for hit in ev["hits"]:
        if hit["rule"] == "undeclared_load":
            key, gpus, jobs = ("undeclared_load", hit["host"]), [hit["gpu"]], [hit["job"]] if hit.get("job") else []
        elif hit["rule"] == "training_shaped":
            key, gpus, jobs = ("training_shaped", hit["job"]), ev["jobs_gpus"].get(hit["job"], []), [hit["job"]]
        else:
            key, gpus, jobs = ("boundary_traffic", "~".join(hit["jobs"])), [], hit["jobs"]
        seen.add(key)
        v = op.get(key)
        if v is None:
            v = op[key] = {"rule": key[0], "subject": key[1], "start": ev["t"] - RULES_EVERY_S, "gpus": [], "jobs": [],
                           "details": []}
            live_event("rule", key[1], f"{key[0]} broken: {hit['detail']}")
        v["end"], v["last_detail"] = ev["t"], hit["detail"]
        v["gpus"] = sorted(set(v["gpus"]) | set(gpus))
        v["jobs"] = sorted(set(v["jobs"]) | set(jobs))
    for key in [k for k in op if k not in seen]:
        v = op.pop(key)
        LIVE_RULES["closed"].append(v)
        live_event("rule", key[1], f"{key[0]} holding again (broken {v['end'] - v['start']:.0f} s)")


async def rules_loop():
    while True:
        await asyncio.sleep(RULES_EVERY_S)
        try:
            ev = await asyncio.to_thread(evaluate_live)
            if ev is None:
                LIVE_RULES["last"] = None
                for key in list(LIVE_RULES["open"]):           # no telemetry: nothing can stay "broken"
                    LIVE_RULES["closed"].append(LIVE_RULES["open"].pop(key))
                continue
            LIVE_RULES["last"] = ev
            if ev["status"] != "warming_up":
                track_violations(ev)
        except Exception as e:                                  # never let one bad evaluation stop the loop
            live_event("rule", "node", f"rule evaluation failed: {e!r}")


@app.on_event("startup")
async def start_rules_loop():
    asyncio.create_task(rules_loop())


def rules_view():
    last = LIVE_RULES["last"]
    return {"every_s": RULES_EVERY_S, "window_s": rules.WINDOW_S, "min_coverage_s": rules.MIN_COVERAGE_S,
            "text": {k: rules.RULE_TEXT[k] for k in SHOWN_RULES},
            "thresholds": {"idle_w": rules.undeclared_load.BUSY_W, "idle_w_blackwell": rules.undeclared_load.BUSY_W_BLACKWELL, "tensor_util_pct": rules.training_shaped.TENSOR_UTIL_PCT,
                           "pcie_sent_gb_s": rules.boundary_traffic.PCIE_SENT_GB_S},
            "last": {k: v for k, v in last.items() if k != "jobs_gpus"} if last else None,
            "violations": list(LIVE_RULES["closed"]) + list(LIVE_RULES["open"].values()),
            "open": [f"{k[0]}|{k[1]}" for k in LIVE_RULES["open"]]}


@app.get("/api/slurm")
def api_slurm():
    now, look = time.time(), gpu_lookup()
    clusters = []
    for name, c in SLURM.items():
        jobs = []
        for j in c["jobs"].values():
            gl = []
            for a in j.get("gpus") or []:
                for i in a.get("idx") or []:
                    gl.append({"node": a.get("node"), "idx": i, **(look.get((a.get("node"), i)) or {})})
            jobs.append({**j, "gpu_list": gl, "streaming": sum(1 for g in gl if g.get("host"))})
        jobs.sort(key=lambda j: -(j.get("submit") or j.get("first_seen") or 0))
        clusters.append({"cluster": name, "reporter": c.get("reporter"), "reporter_host": c.get("reporter_host"),
                         "from_ip": c.get("from_ip"), "age_s": round(now - c.get("received_at", now), 1),
                         "poll_s": c.get("poll_s"), "seq": c.get("seq"), "missed": c.get("missed"),
                         "reports": c.get("reports"), "error": c.get("error"), "raw": c.get("raw", ""),
                         "received_at": c.get("received_at"), "jobs": jobs})
    def phys(h):   # the box's own GPU numbers: a --split part's GPUs are renumbered from 0, physical_index keeps 4-7
        gs = (h.get("hello") or {}).get("gpus") or []
        return {int(g.get("index", i)): g.get("physical_index", g.get("index", i)) for i, g in enumerate(gs) if isinstance(g, dict)}
    boxes = [{"host": host, "label": box_label(h), "age_s": round(now - h["last_seen"], 1),
              "gpus": sorted(int(g) for g in h["gpus"]), "phys": phys(h)} for host, h in sorted(LIVE.items())]   # light list for the Replay page
    return {"now": now, "clusters": clusters, "boxes": boxes, "rules": rules_view()}


# ---------------------------------------------------------------- replay feed
# A recorded run played into /api/collect by replay_feed.py, started and stopped from the
# Debug page. It runs as a child process and POSTs over localhost, so it takes exactly the
# path a real collector takes. Replay hosts are named "replay · <name>" so the page can tag them.
REPLAY_FILE = HERE.parent / "replays" / "c1-tp8-c64.jsonl.gz"
REPLAY_PREFIX = "replay · "
REPLAY = {"proc": None, "host": None}


_REPLAY_HDR = {}


def replay_header():
    if not _REPLAY_HDR:
        import gzip
        with gzip.open(REPLAY_FILE, "rt") as f:
            _REPLAY_HDR.update(json.loads(f.readline()))
    return _REPLAY_HDR


@app.post("/api/replay/start")
def api_replay_start(request: Request):
    p = REPLAY["proc"]
    if p and p.poll() is None:
        return {"ok": True, "already": True, "host": REPLAY["host"]}
    hdr = replay_header()
    host, now = REPLAY_PREFIX + hdr["name"], time.time()
    if host not in LIVE and len(LIVE) >= MAX_BOXES and all(now - x["last_seen"] <= EVICT_SILENT_S for x in LIVE.values()):
        return JSONResponse({"ok": False, "error": f"already {MAX_BOXES} boxes on the page"}, status_code=409)
    port = request.scope.get("server", (None, 8000))[1]   # our own port: 8000 locally, 8080 on Fly
    REPLAY["host"] = host
    REPLAY["proc"] = subprocess.Popen(
        [sys.executable, str(HERE.parent / "replay_feed.py"), "--node", f"http://127.0.0.1:{port}",
         "--file", str(REPLAY_FILE), "--host", REPLAY["host"], "--exit-with-parent"],
        env={**os.environ, "TWC_PASSWORD": SITE_PASSWORD or ""})
    return {"ok": True, "host": REPLAY["host"]}


@app.post("/api/replay/stop")
def api_replay_stop():
    p = REPLAY["proc"]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
    REPLAY["proc"] = None
    for host in [h for h in LIVE if h.startswith(REPLAY_PREFIX)]:   # drop its data so the page goes back to empty
        LIVE.pop(host, None)
        live_event("stopped", host, "replay stopped from the page; slot freed")
    return {"ok": True}


@app.on_event("shutdown")
def stop_replay_on_shutdown():
    if REPLAY["proc"] and REPLAY["proc"].poll() is None:
        REPLAY["proc"].terminate()


def ingest_stats(h, now):
    recent = [x for x in h["recv"] if x[0] > now - 60]
    times = [x[0] for x in h["recv"] if x[0] > now - 300]
    gaps = sum(1 for a, b in zip(times, times[1:]) if b - a > 3)
    fields = sorted({k for q in h["rows"].values() for r in list(q)[-20:] for k in r} - {"t", "gpu"})
    return {
        "batches_last_min": len(recent),
        "samples_per_s": round(sum(x[2] for x in recent) / 60, 1),
        "kb_per_s": round(sum(x[1] for x in recent) / 60 / 1024, 2),
        "lag_ms": round(recent[-1][3] * 1000) if recent else None,   # node clock minus newest sample time
        "gaps_5min": gaps,                                           # pauses > 3 s between batches
        "fields": fields,
    }


def mean_series(per_gpu):
    """Mean across GPUs at each timestamp (a box stamps all its GPUs with the same time each tick)."""
    acc = {}
    for pts in per_gpu:
        for t, v in pts:
            s = acc.setdefault(round(t, 1), [0.0, 0])
            s[0] += v
            s[1] += 1
    return [(t, round(s / n, 2)) for t, (s, n) in sorted(acc.items())]


@app.get("/api/live")
def api_live():
    out, now = {}, time.time()
    for host in [x for x, h in LIVE.items() if now - h["last_seen"] > EXPIRE_SILENT_S]:
        live_event("expired", host, f"silent for {now - LIVE[host]['last_seen']:.0f} s; dropped from the page")
        LIVE.pop(host)
    for host, h in LIVE.items():
        gpus, full = {}, {"power": [], "hbm_bw": [], "nvlink": [], "pcie_tx": [], "pcie_rx": []}
        for g, q in sorted(h["rows"].items()):
            if not q:
                continue
            t_end = q[-1]["t"]
            win = [r for r in q if r["t"] > t_end - 60]
            slow = [r for r in win if "energy_mj" in r]                                     # the 1 Hz rows
            series = {
                "power": [(r["t"], r["power_w"]) for r in win if r.get("power_w") is not None][::2],  # 5 Hz
                "hbm_bw": [(r["t"], r["dram_bw_util_pct"]) for r in slow if r.get("dram_bw_util_pct") is not None],
                "nvlink": [(r["t"], (r.get("nvlink_tx_gb_s") or 0) + (r.get("nvlink_rx_gb_s") or 0))
                           for r in slow if r.get("nvlink_tx_gb_s") is not None],
                # PCIe is the GPU's link to the host (and, on a real cluster, to the network card). Kept as two
                # series, not a sum: serving already receives GB/s from host memory, so "sent" has the quiet floor.
                "pcie_tx": [(r["t"], r["pcie_tx_gb_s"]) for r in slow if r.get("pcie_tx_gb_s") is not None],
                "pcie_rx": [(r["t"], r["pcie_rx_gb_s"]) for r in slow if r.get("pcie_rx_gb_s") is not None],
            }
            for k in full:
                full[k].append(series[k])
            # the faint per-GPU lines don't need 5 Hz: halve power again to keep 4 boxes x 8 GPUs light
            gpus[g] = {"name": h["gpus"].get(str(g)), **series, "power": series["power"][::2],
                       "latest": slow[-1] if slow else {}}                               # newest 1 Hz row
        hello = h.get("hello") or {}
        out[host] = {"slot": h["slot"], "label": box_label(h), "age_s": round(now - h["last_seen"], 1),
                     "batches": h["batches"], "hello_at": hello.get("received_at"),
                     "gpus": gpus, "mean": {k: mean_series(v) for k, v in full.items()},
                     "ingest": ingest_stats(h, now), "double_counted": double_counted(host)}
        rep = hello.get("replay") or ({"label": replay_header().get("label", "recorded"), "about": replay_header().get("about", "")}
                                      if host.startswith(REPLAY_PREFIX) else None)
        if rep:
            out[host]["replay"], out[host]["about"] = rep.get("label", "recorded"), rep.get("about", "")
        # only the replay this node started itself can be stopped from the page
        out[host]["stoppable"] = host == REPLAY["host"] and REPLAY["proc"] is not None and REPLAY["proc"].poll() is None
    return out


@app.get("/api/live/events")
def api_live_events():
    """Connection log, newest first: hellos (with everything the box said), refusals, evictions."""
    return list(reversed(LIVE_EVENTS))


@app.get("/api/live/raw")
def api_live_raw(host: str, stream: int = 0):
    """For the Raw tab: per GPU, the newest value of every field and when it was read,
    the merged 'inbound this instant' record, and (optionally) the last 30 s of samples."""
    h = LIVE.get(host)
    if not h:
        return JSONResponse({"error": "unknown host"}, status_code=404)
    newest_t = max((q[-1]["t"] for q in h["rows"].values() if q), default=None)
    gpus = {}
    for g, q in sorted(h["rows"].items()):
        if not q:
            continue
        fast, slow = q[-1], next((r for r in reversed(q) if "energy_mj" in r), {})
        fields = {k: {"v": v, "t": slow["t"]} for k, v in slow.items() if k not in ("t", "gpu", "power_w")}
        if fast.get("power_w") is not None:
            fields["power_w"] = {"v": fast["power_w"], "t": fast["t"]}
        gpus[g] = {"name": h["gpus"].get(str(g)), "fields": fields, "instant": {**slow, **fast}}
    lb = h.get("last_batch", {})
    out = {"host": host, "label": box_label(h), "hello": h.get("hello"), "newest_t": newest_t, "received_at": h.get("last_seen"),
           "batch": {"n_samples": lb.get("n_samples"), "bytes": lb.get("bytes")}, "gpus": gpus}
    if stream and newest_t is not None:
        rows = [r for q in h["rows"].values() for r in q if r["t"] > newest_t - 30]
        out["stream"] = sorted(rows, key=lambda r: (-r["t"], r["gpu"]))
    return out


@app.get("/collector.py")
def collector_script():
    # the node hands out the collector, so every box gets the same version with one curl
    return FileResponse(HERE.parent / "collector.py", media_type="text/x-python", headers=NO_CACHE)


@app.get("/setup.py")
def setup_script():
    return FileResponse(HERE.parent / "setup.py", media_type="text/x-python", headers=NO_CACHE)


@app.get("/slurm_reporter.py")
def reporter_script():
    return FileResponse(HERE.parent / "slurm_reporter.py", media_type="text/x-python", headers=NO_CACHE)


@app.get("/live")
def live_redirect():
    return RedirectResponse("/debug")


@app.get("/debug")
def debug_view():
    return page("debug.html")


class NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers.update(NO_CACHE)
        return r


app.mount("/static", NoCacheStatic(directory=HERE / "static"), name="static")
# Local only: mock-up stages live in ../mock-ups/, which .dockerignore keeps out of the image, so this never exists online.
if (HERE.parent / "mock-ups").is_dir():
    app.mount("/mock", NoCacheStatic(directory=HERE.parent / "mock-ups"), name="mock")


app = Auth(app)  # noqa: F811  (wrap last so it covers HTTP + websockets)
