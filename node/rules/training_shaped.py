"""
Rule 2: a job declared inference does not look like training.

Exact definition: rules.md, "Rule 2".

One test, tensor cores. Training is limited by raw matrix arithmetic, so it keeps the GPU's tensor cores
busy. Generating tokens is limited by memory speed, so they sit mostly idle.

Measured, one box each: TorchTitan training keeps tensor cores 56 to 79% busy (B200, H200). Honest DeepSeek
serving reached at most about 32% (H200). Known gaps: for about a minute after a checkpoint save, and during
a cold start, a training job's 60 s median drops under the limit.

A second test, "power spread" (running hot while the 5 s average power stays within a 100 W band), was
removed on 2026-11-11: on H200 it fired on honest serving under a steady load (64 of 88 checks at 64 AgentX
sessions, 12 of 88 at 16), while tensor cores caught every training run without it.
"""
NAME = "training_shaped"
TENSOR_UTIL_PCT = 40.0   # tensor cores busier than this (median of the trailing minute)
TEXT = f"A job declared inference keeps its tensor cores > {TENSOR_UTIL_PCT:.0f}% busy (60 s median)."


def tensor_util(w, job):
    return w.reading_value("tensor_util", job)


def check(w):
    hits = []
    for job in w.jobs:
        busy = tensor_util(w, job) if job.type == "inference" else None
        if busy is not None and busy > TENSOR_UTIL_PCT:
            hits.append({"rule": NAME, "gpus": list(job.gpus), "jobs": [job.id],
                         "detail": f"{job.id}: tensor cores {busy:.0f}% busy"})
    return hits
