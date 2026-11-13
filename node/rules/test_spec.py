# A second implementation, written from rules.md alone, compared with the real rules on random cases.
# Run from demo-classifier/:  .venv/bin/python node/rules/test_spec.py
import sys, math, random, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # node/
import rules
def spec(power, readings, decls, gpus, t_end, models):
    k_end = round(t_end / 0.1); grid = {g: {} for g in gpus}
    for g in gpus:
        for t, w in power.get(g, []):
            i = 599 - (k_end - round(t / 0.1))
            if 0 <= i < 600: grid[g][i] = w
    kept = [i for i in range(600) if all(i in grid[g] for g in gpus)]
    if len(kept) * 0.1 < 30: return "warming_up", set()
    mean = {g: sum(grid[g][i] for i in kept) / len(kept) for g in gpus}
    hits = set(); covered = {g for d in decls for g in d["gpus"]}
    floor = {g: 300 if any(x in (models.get(g) or "").upper() for x in ("B200", "B300")) else 200 for g in gpus}
    for g in gpus:                                                            # rule 1
        if g not in covered and mean[g] > floor[g]: hits.add(("undeclared_load", g))
    def value(counter, gs):
        meds = []
        for g in gs:
            v = [x for t, x in readings.get(counter, {}).get(g, []) if t_end - 60 < t <= t_end + 1]
            if len(v) >= 15: meds.append(st.median(v))
        return sum(meds) / len(meds) if meds else None
    jobs = []
    for d in decls:
        gs = [g for g in d["gpus"] if g in gpus]
        if not gs: continue
        jm = sum(grid[g][i] for g in gs for i in kept) / (len(gs) * len(kept))
        jobs.append((d["job_id"], d["type"], jm > max(floor[g] for g in gs), value("pcie_sent", gs)))
        if d["type"] != "inference": continue
        boxes = {str(g).rsplit(":", 1)[0] for g in gs}                        # rule 3, test 1
        if len(boxes) == 1:
            v = [x for t, x in readings.get("net_sent", {}).get(boxes.pop(), []) if t_end - 60 < t <= t_end + 1]
            if len(v) >= 15 and sum(v) / len(v) > 0.001: hits.add(("boundary_traffic", d["job_id"]))
                                                                              # rule 2
        busy = value("tensor_util", gs)
        if busy is not None and busy > 40:
            hits.add(("training_shaped", d["job_id"]))
    for i in range(0):                                                        # rule 3, test 2: switched off (rules.md); range(len(jobs)) when on
        for j in range(i + 1, len(jobs)):
            a, b = jobs[i], jobs[j]
            if a[2] and b[2] and any(x[1] == "inference" and x[3] is not None and x[3] > 0.35 for x in (a, b)):
                hits.add(("boundary_traffic", f"{a[0]}~{b[0]}"))
    return ("flagged" if hits else "compliant"), hits
def code_hits(r):
    return {(h["rule"], h["gpus"][0] if h["rule"] == "undeclared_load" else "~".join(h["jobs"])) for h in r["hits"]}
random.seed(11); agree = 0; kinds = {}; N = 600
for case in range(N):
    t_end = 1794340000 + random.random() * 1000; n = random.choice([2, 4, 8]); gpus = [f"b{i}" for i in range(n)] if random.random() < 0.3 else [f"box{i * 2 // n}:{i}" for i in range(n)]
    power, readings = {}, {"pcie_sent": {}, "tensor_util": {}}
    dur = random.choice([25, 40, 70, 70, 70]); drop = random.choice([0, 0, 0.01, 0.05])
    order = gpus[:]; random.shuffle(order) if random.random() < 0.5 else None; cut = random.choice([0, n // 2, n // 2, n])
    jobA, jobB = order[:cut], order[cut:]
    for grp in (jobA, jobB):
        level = random.choice([90, 199, 201, 299, 301, 399, 401, 499, 501, 650]); swing = random.choice([0, 45, 55, 150]); period = random.choice([200, 300, 450])
        base = random.choice([0.0, 0.15, 0.34, 0.36, 0.45, 1.2]); busy = random.choice([3, 30, 39.5, 40.5, 70]); nread = random.choice([10, 14, 15, 30, 64])
        for g in grp:
            power[g] = [(t_end - dur + k * 0.1 + random.uniform(-0.03, 0.03), level + swing * math.sin(6.2832 * k / period) + random.gauss(0, 30))
                        for k in range(int(dur * 10) + 3) if random.random() > drop]
            if random.random() < 0.9:
                readings["pcie_sent"][g] = [(t_end - 61.5 + k * 63 / nread, base + random.gauss(0, 0.004) + (25 if random.random() < 0.05 else 0)) for k in range(nread)]
            if random.random() < 0.9:
                readings["tensor_util"][g] = [(t_end - 61.5 + k * 63 / nread, busy + random.gauss(0, 0.2)) for k in range(nread)]
    if random.random() < 0.6:   # network readings for some boxes: none, silent, just under, just over, heavy; sometimes too few to count
        readings["net_sent"] = {}
        for b in sorted({g.rsplit(":", 1)[0] for g in gpus}):
            if random.random() < 0.8:
                lvl = random.choice([0.0, 0.0, 0.0004, 0.0012, 0.05]); nread = random.choice([10, 14, 15, 30, 64])
                readings["net_sent"][b] = [(t_end - 61.5 + k * 63 / nread, lvl * random.choice([0, 0, 1, 3])) for k in range(nread)]
    kind = random.choice(["NVIDIA H200", "NVIDIA B200", "nvidia b300", None, "mixed"])
    models = {g: (random.choice(["NVIDIA H200", "NVIDIA B200"]) if kind == "mixed" else kind) for g in gpus if kind}
    decls = []
    if jobA and random.random() < 0.9: decls.append({"job_id": "1", "type": random.choice(["inference", "inference", "pretraining"]), "gpus": jobA})
    if jobB and random.random() < 0.9: decls.append({"job_id": "2", "type": random.choice(["inference", "inference", "pretraining"]), "gpus": jobB + (["ghost"] if random.random() < 0.2 else [])})
    if random.random() < 0.1: decls.append({"job_id": "3", "type": "inference", "gpus": ["not-streaming"]})
    r = rules.evaluate(power, decls, gpus, t_end, readings, models); s = spec(power, readings, decls, gpus, t_end, models)
    same = (r["status"], code_hits(r)) == s; agree += same
    for h in s[1]: kinds[h[0]] = kinds.get(h[0], 0) + 1
    kinds[s[0]] = kinds.get(s[0], 0) + 1
    if not same: print("MISMATCH", case, r["status"], code_hits(r), s)
print(f"agreement: {agree}/{N}; outcomes: {kinds}")
