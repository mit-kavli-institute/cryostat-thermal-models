# cryotherm/visualize.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def visualize_model(
    model,
    *,
    cmap={"cond": "#0072B2", "rad": "#D55E00", "ext": "#009E73"},
    scale=50,  # linewidth (pt) when arrow = 100% of stage load
    dx_cond=0.12,
    dx_rad=0.12,
    dx_ext=0.12,
):
    # ---------- helpers -------------------------------------------------
    def _fmt_power(P_W: float) -> str:
        """Pretty power label: use mW below 1 W, otherwise W."""
        a = abs(P_W)
        if a < 1.0:
            val = P_W * 1e3
            if abs(val) < 10:
                return f"{val:.2f} mW"
            elif abs(val) < 100:
                return f"{val:.1f} mW"
            else:
                return f"{val:.0f} mW"
        else:
            val = P_W
            if abs(val) < 10:
                return f"{val:.2f} W"
            elif abs(val) < 100:
                return f"{val:.1f} W"
            else:
                return f"{val:.0f} W"

    # ---------- stage ordering (cold→hot) -------------------------------
    stages = sorted(model.stages, key=lambda s: s.temperature)
    y = {s: i for i, s in enumerate(stages)}

    # ---------- per-stage TOTAL absolute load (W) for scaling -----------
    stage_total_W = {s: 0.0 for s in stages}

    # Conduction contributes to the *colder* stage
    for c in model.conductors:
        hot, cold = (
            (c.stage1, c.stage2)
            if c.stage1.temperature > c.stage2.temperature
            else (c.stage2, c.stage1)
        )
        qW = abs(c.heat_flow(hot.temperature, cold.temperature))
        stage_total_W[cold] += qW

    # Radiation contributes to the colder stage too
    for r in model.radiators:
        if r.stage2 is None:
            continue
        hot, cold = (
            (r.stage1, r.stage2)
            if r.stage1.temperature > r.stage2.temperature
            else (r.stage2, r.stage1)
        )
        qW = abs(r.heat_flow(hot.temperature, cold.temperature))
        stage_total_W[cold] += qW

    # External loads heat their own stage
    for s in stages:
        qW = abs(s.external_load(s.temperature))
        stage_total_W[s] += qW

    # Avoid 0 division; also keep a copy for display before padding
    stage_total_display_W = stage_total_W.copy()
    for s in stages:
        if stage_total_W[s] == 0.0:
            stage_total_W[s] = 1e-12  # tiny epsilon

    # ---------- plot setup ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    # Stage bars & labels
    for s in stages:
        yy = y[s]
        ax.hlines(yy, 0, 1, lw=1.4, color="black")
        ax.text(
            0.02,
            yy + 0.06,
            f"{s.name}\n{round(s.temperature, 2)} K",
            ha="left",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            0.98,
            yy - 0.06,
            _fmt_power(stage_total_display_W[s]),
            ha="right",
            va="top",
            fontsize=9,
        )

    # Arrow helper
    def _arrow(x, y1, y2, qW, totalW, col, label_prefix):
        pct = 100.0 * abs(qW) / max(totalW, 1e-12)
        lw = max(1.0, min(scale, (pct / 100.0) * scale))  # cap at `scale`
        patch = FancyArrowPatch(
            (x, y1),
            (x, y2),
            mutation_scale=max(lw, 6),  # visible tip even for tiny arrows
            color=col,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            0.5 * (y1 + y2),
            f"{label_prefix}\n{_fmt_power(qW)}\n{pct:.0f}%",
            ha="center",
            va="center",
            fontsize=8,
            color=col,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    # ---------- conduction arrows --------------------------------------
    x0_cond = 0.20
    for idx, c in enumerate(model.conductors):
        hot, cold = (
            (c.stage1, c.stage2)
            if c.stage1.temperature > c.stage2.temperature
            else (c.stage2, c.stage1)
        )
        qW = c.heat_flow(hot.temperature, cold.temperature)
        if abs(qW) < 1e-12:
            continue
        total = stage_total_W[cold]
        lab = getattr(c, "name", getattr(c, "material", "Cond"))
        _arrow(x0_cond + idx * dx_cond, y[hot], y[cold], qW, total, cmap["cond"], lab)

    # ---------- radiation arrows ---------------------------------------
    x0_rad = 0.65
    r_idx = 0
    for r in model.radiators:
        if r.stage2 is None:
            continue
        hot, cold = (
            (r.stage1, r.stage2)
            if r.stage1.temperature > r.stage2.temperature
            else (r.stage2, r.stage1)
        )
        qW = r.heat_flow(hot.temperature, cold.temperature)
        if abs(qW) < 1e-12:
            continue
        total = stage_total_W[cold]
        _arrow(x0_rad + r_idx * dx_rad, y[hot], y[cold], qW, total, cmap["rad"], "Rad")
        r_idx += 1

    # ---------- external loads (downward onto stage) --------------------
    x_ext = 0.92
    for idx, s in enumerate(stages):
        qW = s.external_load(s.temperature)
        if abs(qW) < 1e-12:
            continue
        total = stage_total_W[s]
        _arrow(x_ext + idx * 0.0, y[s] + 0.15, y[s], qW, total, cmap["ext"], "Ext")

    # Cosmetics
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(stages))
    ax.axis("off")
    ax.set_title("Thermal-model heat-flow breakdown", pad=20)
    plt.tight_layout()
