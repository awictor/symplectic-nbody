"""Gillespie SSA demo: exact stochastic reaction trajectories -- noisy decay vs the ODE, and predator-prey oscillation."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import gillespie_ssa as ssa


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Gillespie SSA -- exact stochastic simulation of reaction networks")
    lines.append("=" * 66)
    lines.append("")

    # decay A -> 0: stochastic trajectories bracket the ODE, noisier at small counts
    k = 0.4
    lines.append(f"Decay A -> 0 (k={k}): stochastic mean vs deterministic A0 exp(-kt).")
    lines.append("   A0     t=2 mean    ODE       rel. noise (sd/mean)")
    lines.append("   " + "-" * 50)
    for A0 in (20, 100, 1000):
        react = [ssa.Reaction({0: 1}, {}, k)]
        vals = []
        for r in range(300):
            times, states = ssa.simulate(react, [A0], t_max=2.0, seed=r + 1)
            sampled = ssa.sample_at(times, states, [2.0])
            vals.append(sampled[0][0])
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        ode = A0 * math.exp(-k * 2.0)
        lines.append(f"   {A0:4d}   {m:8.2f}   {ode:8.2f}    {sd/m if m else 0:.3f}")
    lines.append("")
    lines.append("Small counts -> large relative noise (ODEs miss it); large counts -> ODE limit.")
    lines.append("")

    # Lotka-Volterra oscillation
    lv = [
        ssa.Reaction({0: 1}, {0: 2}, 1.0),        # prey birth
        ssa.Reaction({0: 1, 1: 1}, {1: 2}, 0.01), # predation
        ssa.Reaction({1: 1}, {}, 1.0),            # predator death
    ]
    times, states = ssa.simulate(lv, [50, 50], t_max=12.0, seed=7)
    prey = [s[0] for s in states]
    pred = [s[1] for s in states]
    lines.append("Lotka-Volterra predator-prey (stochastic):")
    lines.append(f"  prey range  {min(prey)}..{max(prey)},  predator range {min(pred)}..{max(pred)}")
    lines.append(f"  {len(times)} reaction events in t=[0,12] -- sustained noisy oscillation")
    lines.append("")

    # waiting-time / exactness note
    st = [100, 50]
    a0 = ssa.total_propensity(lv, st)
    lines.append(f"At state (prey=100, pred=50) total propensity a0 = {a0:.2f};")
    lines.append(f"  next reaction waits an exponential time, mean 1/a0 = {1/a0:.4f}. Statistically exact.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(times, prey, pred)
    return text, svg


def _svg(times, prey, pred):
    W, H = 640, 450
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Stochastic Lotka-Volterra: time series (top) and phase orbit (bottom)</text>')

    tmax = times[-1]
    vmax = max(max(prey), max(pred))

    # TOP: time series (step functions)
    tx0, tx1, ty0, ty1 = 55, 610, 50, 200

    def tx(t):
        return tx0 + t / tmax * (tx1 - tx0)

    def ty(v):
        return ty1 - v / vmax * (ty1 - ty0)

    def step_poly(vals):
        pts = []
        for i in range(len(times)):
            pts.append(f"{tx(times[i]):.1f},{ty(vals[i]):.1f}")
            if i + 1 < len(times):
                pts.append(f"{tx(times[i + 1]):.1f},{ty(vals[i]):.1f}")  # hold
        return " ".join(pts)

    parts.append(f'<polyline points="{step_poly(prey)}" fill="none" stroke="{P["green"]}" stroke-width="1.2"/>')
    parts.append(f'<polyline points="{step_poly(pred)}" fill="none" stroke="{P["red"]}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{tx0}" y1="{ty1}" x2="{tx1}" y2="{ty1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{tx0}" y="{ty0 - 2:.1f}" fill="{P["green"]}" font-size="11">prey</text>')
    parts.append(f'<text x="{tx0 + 50}" y="{ty0 - 2:.1f}" fill="{P["red"]}" font-size="11">predator</text>')
    parts.append(f'<text x="{tx1 - 40}" y="{ty1 + 14:.1f}" fill="{P["gray"]}" font-size="10">time</text>')

    # BOTTOM: phase plane (prey vs predator) -- the noisy orbit
    px0, px1, py0, py1 = 200, 460, 250, 430
    pmax_x = max(prey)
    pmax_y = max(pred)

    def phx(v):
        return px0 + v / pmax_x * (px1 - px0)

    def phy(v):
        return py1 - v / pmax_y * (py1 - py0)

    orbit = " ".join(f"{phx(prey[i]):.1f},{phy(pred[i]):.1f}" for i in range(len(prey)))
    parts.append(f'<polyline points="{orbit}" fill="none" stroke="{P["yellow"]}" '
                 f'stroke-width="0.8" opacity="0.7"/>')
    parts.append(f'<circle cx="{phx(prey[0]):.1f}" cy="{phy(pred[0]):.1f}" r="4" fill="{P["green"]}"/>')
    parts.append(f'<rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" '
                 f'fill="none" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{px0}" y="{py0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'phase orbit: prey (x) vs predator (y)</text>')
    parts.append(f'<text x="{(px0+px1)/2:.1f}" y="{py1 + 14:.1f}" fill="{P["gray"]}" '
                 f'font-size="10" text-anchor="middle">prey</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'each step is one exact reaction event; the orbit wanders (stochastic), unlike the ODE '
                 f'closed loop.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
