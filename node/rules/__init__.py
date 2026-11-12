"""The rules: one file each, plus this orchestrator.

    undeclared_load.py    1. no busy GPU outside a declared job
    training_shaped.py    2. a job declared inference does not look like training
    boundary_traffic.py   3. separate jobs do not send bulk data to each other

Every rule file has the same shape: NAME, TEXT (one sentence for the page), its thresholds, and
check(window), which returns a list of hits {rule, gpus, jobs, detail}.
The rules read two kinds of data: power (about ten readings a second, laid on a shared grid; it decides
which GPUs and jobs are busy) and once-a-second readings of other counters ("pcie_sent", "tensor_util"),
used as a median per job.

This file does only what the rules share. It lays the trailing minute of every GPU's power readings
on one grid, refuses to judge on too little data, runs the three rules and reports. rules.md says
all of it in plain language, exactly enough to write another implementation from.
"""
from dataclasses import dataclass

import numpy as np

from . import boundary_traffic, training_shaped, undeclared_load

RULES = (undeclared_load, training_shaped, boundary_traffic)
RULE_TEXT = {r.NAME: r.TEXT for r in RULES}

WINDOW_S = 60.0        # the rules look at the trailing minute
GRID_S = 0.1           # power readings are laid on a grid of this step
MIN_COVERAGE_S = 30.0  # with less usable data than this, nothing is judged
MIN_READINGS = 15      # a GPU needs this many once-a-second readings in the window for its median to count


@dataclass
class Job:
    id: str
    type: str    # "inference" or "pretraining"
    gpus: list   # the GPUs it was given that are streaming


@dataclass
class Window:
    t_end: float
    gpus: list        # every streaming GPU
    power: dict       # gpu -> watts over the kept grid slots; the same slots for every GPU
    jobs: list        # declared jobs with at least one streaming GPU
    covered: set      # every GPU a declared job was given
    readings: dict    # counter name -> gpu -> [(time, value)], about one a second; any of it may be missing.
                      # "net_sent" is per box, not per GPU: box -> [(time, GB/s sent to other boxes)]
    models: dict      # gpu -> the model name its driver reports, e.g. "NVIDIA B200"; may be missing

    def gpu_mean_w(self, gpu):
        return float(self.power[gpu].mean())

    def job_mean_w(self, job):
        return float(np.mean([self.power[g] for g in job.gpus]))

    def gpu_busy_floor_w(self, gpu):
        return undeclared_load.busy_floor_w(self.models.get(gpu))

    def job_is_busy(self, job):
        """Mean power over the busy floor; a job on mixed GPU generations uses the highest of their floors."""
        return self.job_mean_w(job) > max(self.gpu_busy_floor_w(g) for g in job.gpus)

    def reading_value(self, name, job):
        """A job's value for a once-a-second counter: each GPU's median over the window, averaged over
        the GPUs that have at least MIN_READINGS readings in it. None if no GPU does."""
        medians = []
        for g in job.gpus:
            values = [v for t, v in self.readings.get(name, {}).get(g, ()) if self.t_end - WINDOW_S < t <= self.t_end + 1.0]
            if len(values) >= MIN_READINGS:
                medians.append(float(np.median(values)))
        return float(np.mean(medians)) if medians else None


def kept_slots(buffers, gpus, t_end):
    """Each GPU's readings on the shared grid, keeping only the slots in which every GPU has one.
    A reading goes to the nearest slot (round, not floor, so timestamp jitter can't put two readings
    in one slot and leave the next empty); the last slot is t_end. Returns [slots, gpus] in time order."""
    n = int(WINDOW_S / GRID_S)
    last = round(t_end / GRID_S)
    grid = np.full((n, len(gpus)), np.nan)
    for j, g in enumerate(gpus):
        for t, watts in buffers.get(g, ()):
            i = n - 1 - (last - round(t / GRID_S))
            if 0 <= i < n:
                grid[i, j] = watts
    return grid[~np.isnan(grid).any(axis=1)]


def evaluate(buffers, declarations, gpus, t_end, readings=None, models=None):
    """buffers: gpu -> [(time, watts)]. declarations: [{job_id, type, gpus}]. gpus: the streaming GPUs.
    readings: {"pcie_sent": gpu -> [(time, GB/s)], "tensor_util": gpu -> [(time, %)],
               "net_sent": box -> [(time, GB/s)]}, all optional. A GPU named "<box>:<index>" is on that box.
    models: gpu -> model name, optional; it sets each GPU's "busy" floor.
    Returns {status, hits, features}; status is warming_up, compliant or flagged."""
    kept = kept_slots(buffers, gpus, t_end)
    coverage = len(kept) * GRID_S
    if coverage < MIN_COVERAGE_S:
        return {"status": "warming_up", "hits": [], "features": {"coverage_s": coverage}}

    streaming = set(gpus)
    jobs = [Job(d["job_id"], d["type"], [g for g in d["gpus"] if g in streaming]) for d in declarations]
    w = Window(t_end=t_end, gpus=list(gpus), power={g: kept[:, j] for j, g in enumerate(gpus)},
               jobs=[job for job in jobs if job.gpus], covered={g for d in declarations for g in d["gpus"]},
               readings=readings or {}, models=models or {})

    hits = [hit for rule in RULES for hit in rule.check(w)]
    return {"status": "flagged" if hits else "compliant", "hits": hits, "features": features(w, coverage)}


def features(w, coverage):
    """The numbers behind the verdict, for the page."""
    jobs = {}
    for job in w.jobs:
        jobs[job.id] = {"type": job.type, "mean_w": round(w.job_mean_w(job))}
        for key, value, digits in (("tensor_util", training_shaped.tensor_util(w, job), 1),
                                   ("pcie_sent", boundary_traffic.pcie_sent(w, job), 2),
                                   ("net_sent", boundary_traffic.net_sent(w, job), 4)):
            if value is not None:
                jobs[job.id][key] = round(value, digits)
    return {"coverage_s": round(coverage, 1), "gpu_mean_w": {str(g): round(w.gpu_mean_w(g)) for g in w.gpus},
            "jobs": jobs, "busy_pairs": [f"{a.id}~{b.id}" for a, b in boundary_traffic.busy_pairs(w)]}
