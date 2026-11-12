"""Rule 3: separate jobs do not send bulk data to each other.

Exact definition: rules.md, "Rule 3".

One test is on: network. A second one, PCIe, is switched off for now (Atharva's call, 2026-11-11): set
PCIE_TEST_ON to bring it back, and put test 2 back in rules.md and test_spec.py. With it off, a box whose
collector reports no network reading can never break this rule.

  network   A job declared inference that sits on one box, while that box sends to other boxes over its
            high-speed network ports (the RDMA ports that join the nodes of a cluster). Judged on the mean
            over the trailing minute, not the median: a prefill-to-decode hand-off moves in bursts, so its
            median is 0. No "busy" condition: the prefill half of a split idles under the busy floor most
            of the time, and the traffic is the evidence. This is the boundary itself, and it is only
            available when the collector reports it ("net_sent"). A job spread over several boxes is not
            judged by this test, because the counter cannot tell traffic between its own boxes from
            traffic that leaves them.
  PCIe      (off) Two busy jobs, one of which has GPUs sending a lot over PCIe (to the host and from there
            to the network card). It was the stand-in when there is no network reading: one box, or a
            collector that does not report it. It missed the real two-node split (0.24 against 0.35).

Why only jobs declared inference are checked: honest training can send more over PCIe than a cheating
inference job does, and a training job spread over several boxes uses the network by design.

Measured on the network test, one rental of two 8x H200 nodes, 2026-11-11: four honest single-node
recordings (serving and training) sent exactly 0 on these ports. A real prefill/decode split across the two
nodes, under 16 AgentX sessions, sent 0.011 to 0.236 GB/s from the prefill node (60 s means); the decode
node sent at most 0.0002 back. Not measured: clusters that also carry storage over these ports.

Measured on the PCIe test, per GPU, one box each: honest serving 0.15 to 0.31 GB/s on H200 (0.45 for
expert-parallel on 8 GPUs); a one-box prefill-to-decode hand-off 0.42 to 0.48 with 20,000-token prompts;
the two-node split above only 0.24; honest training 1.1 to 1.5 on H200 and 0.00 on a B200.
"""
import numpy as np

NAME = "boundary_traffic"
NET_SENT_GB_S = 0.001
PCIE_TEST_ON = False
PCIE_SENT_GB_S = 0.35
TEXT = f"A job declared inference on one node sends > {NET_SENT_GB_S} GB/s to other nodes over the network (60 s mean)."


def pcie_sent(w, job):
    return w.reading_value("pcie_sent", job)


def box_of(gpu):
    """A GPU is named "<box>:<index>"; a name with no colon is its own box."""
    return str(gpu).rsplit(":", 1)[0]


def net_sent(w, job):
    """What the job's box sends to other boxes: the mean of the box's readings over the window. None if the job
    is on more than one box, or the box has fewer than MIN_READINGS readings in the window.
    A box's traffic counts for every single-box job on it."""
    from . import MIN_READINGS, WINDOW_S
    boxes = {box_of(g) for g in job.gpus}
    if len(boxes) != 1:
        return None
    values = [v for t, v in w.readings.get("net_sent", {}).get(boxes.pop(), ()) if w.t_end - WINDOW_S < t <= w.t_end + 1.0]
    return float(np.mean(values)) if len(values) >= MIN_READINGS else None


def busy_pairs(w):
    busy = [job for job in w.jobs if w.job_is_busy(job)]   # the same floor as rule 1
    return [(a, b) for i, a in enumerate(busy) for b in busy[i + 1:]]


def check(w):
    hits = []
    for job in w.jobs:                                     # test 1, network: a job by itself
        net = net_sent(w, job) if job.type == "inference" else None
        if net is not None and net > NET_SENT_GB_S:
            hits.append({"rule": NAME, "gpus": list(job.gpus), "jobs": [job.id],
                         "detail": f"{job.id} sits on one node and sends {net:.3f} GB/s to other nodes over the network"})
    for a, b in busy_pairs(w) if PCIE_TEST_ON else ():     # test 2, PCIe: a pair of busy jobs (switched off)
        offenders = [(job, sent) for job in (a, b) if job.type == "inference"
                     and (sent := pcie_sent(w, job)) is not None and sent > PCIE_SENT_GB_S]
        if offenders:
            who = ", ".join(f"{job.id} sends {sent:.2f} GB/s per GPU over PCIe" for job, sent in offenders)
            hits.append({"rule": NAME, "gpus": [], "jobs": [a.id, b.id],
                         "detail": f"{a.id} and {b.id} are both busy; {who}"})
    return hits
