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
    # order stages cold→hot
    stages = sorted(model.stages, key=lambda s: s.temperature)
    y = {s: i for i, s in enumerate(stages)}

    # compute each stage’s TOTAL absolute load (mW)
    stage_total = {s: 0.0 for s in stages}
    for c in model.conductors:
        q = abs(c.heat_flow(c.stage1.temperature, c.stage2.temperature)) * 1e3
        # heat enters the colder stage
        _, cold = (
            (c.stage1, c.stage2)
            if c.stage1.temperature > c.stage2.temperature
            else (c.stage2, c.stage1)
        )
        stage_total[cold] += q
    for r in model.radiators:
        if r.stage2 is not None:
            q = abs(r.heat_flow(r.stage1.temperature, r.stage2.temperature)) * 1e3
            _, cold = (
                (r.stage1, r.stage2)
                if r.stage1.temperature > r.stage2.temperature
                else (r.stage2, r.stage1)
            )
            stage_total[cold] += q
    for s in stages:
        qext = abs(s.external_load(s.temperature)) * 1e3
        stage_total[s] += qext
    # avoid div zero
    for s in stages:
        if stage_total[s] == 0:
            stage_total[s] = 1e-9

    # set up plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in stages:
        yy = y[s]
        ax.hlines(yy, 0, 1, lw=1.4, color="black")
        ax.text(
            0.02,
            yy + 0.06,
            f"{s.name}\n{round(s.temperature,2)} K",
            ha="left",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            0.98,
            yy - 0.06,
            f"{stage_total[s]:.1f} mW",
            ha="right",
            va="top",
            fontsize=9,
        )

    # arrow helper
    def _arrow(x, y1, y2, q_mW, total_mW, col, label):
        pct = q_mW / total_mW * 100
        lw = max(1.0, pct / 100 * scale)
        patch = FancyArrowPatch(
            (x, y1),
            (x, y2),
            # arrowstyle="-|>",
            mutation_scale=max(lw, 6),
            # linewidth=lw,
            color=col,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            0.5 * (y1 + y2),
            f"{label}\n{pct:.0f}%",
            ha="center",
            va="center",
            fontsize=8,
            color=col,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    # conduction arrows
    x0_cond = 0.20
    for idx, c in enumerate(model.conductors):
        hot, cold = (
            (c.stage1, c.stage2)
            if c.stage1.temperature > c.stage2.temperature
            else (c.stage2, c.stage1)
        )
        q = c.heat_flow(hot.temperature, cold.temperature) * 1e3
        if abs(q) < 1e-6:
            continue
        total = stage_total[cold]
        lab = getattr(c, "name", c.material)
        _arrow(
            x0_cond + idx * dx_cond,
            y[hot],
            y[cold],
            q,
            total,
            cmap["cond"],
            f"{lab}\n{q:.1f} mW",
        )

    # radiation arrows
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
        q = r.heat_flow(hot.temperature, cold.temperature) * 1e3
        if abs(q) < 1e-6:
            continue
        total = stage_total[cold]
        _arrow(
            x0_rad + r_idx * dx_rad,
            y[hot],
            y[cold],
            q,
            total,
            cmap["rad"],
            f"Rad\n{q:.1f} mW",
        )
        r_idx += 1

    # external loads (always downward onto stage)
    x_ext = 0.92
    for s in stages:
        q = s.external_load(s.temperature) * 1e3
        if abs(q) < 1e-6:
            continue
        total = stage_total[s]
        _arrow(x_ext, y[s] + 0.15, y[s], q, total, cmap["ext"], f"Ext\n{q:.1f} mW")

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(stages))
    ax.axis("off")
    ax.set_title("Thermal-model heat-flow breakdown", pad=20)
    plt.tight_layout()
