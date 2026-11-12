# One box split into "nodes": what's fake

`collector.py --split` makes one GPU box report as several nodes. That's fine for pipeline and type checks. Don't use it as evidence for anything below.

- **Cross-node traffic.**
  - Real nodes talk over InfiniBand/RoCE, ~50 GB/s per GPU. NVML can't see that traffic.
  - Split nodes talk over NVLink, ~450 GB/s per direction, and NVML's NVLink counters show it.
  - So collectives run ~9× faster: smaller lockstep dips.
  - Coupling also looks easier to detect than it really is.
- **Clocks.**
  - Real nodes are skewed by NTP (milliseconds) and sample on independent schedules.
  - Split nodes share one clock, so synchrony checks get perfect alignment for free.
- **Physical sharing.** Split nodes share PSU, cooling, host CPU and power cap.
  - Correlated throttling can fake synchrony between them.
  - The c2 TP4×2 run showed ~0 for serving; it's untested for training at the power cap.
  - The box dying takes every "node" down at once.
- **Topology.**
  - The box is one fully connected NVSwitch domain, so there's nothing to re-derive.
  - Rewiring attacks can't be staged.
- **Box identity.**
  - All split nodes share the same hostname, CPU, driver and BMC.
  - The verifier could tell they're one box. The hello is honest about this on purpose, and rules must not read it.
