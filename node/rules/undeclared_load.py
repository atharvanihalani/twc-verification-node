"""
Rule 1: no busy GPU outside a declared job.

Exact definition: rules.md, "Rule 1".

"Busy" is a floor in watts that depends on the GPU's generation, because Blackwell cards idle higher.
Measured, one box each: an H200 draws about 80 W with nothing loaded and about 120 W with a server
loaded and waiting. A B200 draws about 190 W with nothing loaded and about 250 W with a server loaded
and waiting; an idle B300 about 180 W.
"""
NAME = "undeclared_load"
BUSY_W = 200.0             # Hopper and anything unrecognised
BUSY_W_BLACKWELL = 300.0   # B200, B300 (and GB200, GB300)
TEXT = (f"A GPU averages > {BUSY_W:.0f} W ({BUSY_W_BLACKWELL:.0f} W on Blackwell) "
        f"while no declared job covers it.")


def is_blackwell(model):
    """`model` is the name the driver reports, e.g. "NVIDIA B200"."""
    return any(x in (model or "").upper() for x in ("B200", "B300"))


def busy_floor_w(model):
    """The power above which a GPU of this model counts as busy."""
    return BUSY_W_BLACKWELL if is_blackwell(model) else BUSY_W


def check(w):
    return [{"rule": NAME, "gpus": [g], "jobs": [], "detail": f"GPU {g} at {w.gpu_mean_w(g):.0f} W, no job"}
            for g in w.gpus if g not in w.covered and w.gpu_mean_w(g) > w.gpu_busy_floor_w(g)]
