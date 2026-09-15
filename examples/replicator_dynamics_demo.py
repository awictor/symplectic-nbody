"""Replicator dynamics demo: rock-paper-scissors cycling on the simplex, and a dominant strategy sweeping."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import replicator_dynamics as rd


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Replicator dynamics -- evolutionary game theory on the simplex")
    lines.append("=" * 62)
    lines.append("")

    # RPS
    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    x0 = [0.5, 0.3, 0.2]
    times, traj = rd.integrate(rps, x0, t_max=40.0, dt=0.005, record_every=200)
    lines.append("Rock-Paper-Scissors (zero-sum cyclic game):")
    lines.append(f"  start (R,P,S) = ({x0[0]}, {x0[1]}, {x0[2]})")
    lines.append("   t      R       P       S      product x1x2x3")
    lines.append("   " + "-" * 46)
    for k in range(0, len(traj), max(1, len(traj) // 6)):
        x = traj[k]
        lines.append(f"   {times[k]:4.0f}   {x[0]:.3f}   {x[1]:.3f}   {x[2]:.3f}   {rd.rps_conserved(x):.5f}")
    prods = [rd.rps_conserved(x) for x in traj]
    lines.append(f"  product conserved: range {min(prods):.5f}..{max(prods):.5f} (orbits, never settles)")
    lines.append("")

    # dominant strategy
    dom = [[3, 3, 3], [1, 2, 1], [0, 1, 2]]
    times, traj = rd.integrate(dom, [1 / 3, 1 / 3, 1 / 3], t_max=30.0, dt=0.01, record_every=100)
    lines.append("Strictly dominant strategy (strategy 1 earns most vs everyone):")
    lines.append("   t      x1      x2      x3")
    lines.append("   " + "-" * 34)
    for k in range(0, len(traj), max(1, len(traj) // 6)):
        x = traj[k]
        lines.append(f"   {times[k]:4.0f}   {x[0]:.3f}   {x[1]:.3f}   {x[2]:.3f}")
    lines.append("  -> sweeps to fixation (x1 -> 1); dominance always wins under replicator dynamics.")
    lines.append("")

    # hawk-dove interior equilibrium
    hd = [[0, 3], [1, 2]]
    _, traj_hd = rd.integrate(hd, [0.9, 0.1], t_max=50.0, dt=0.01, record_every=200)
    lines.append("Hawk-Dove (interior Nash equilibrium at x=0.5):")
    lines.append(f"  from x1=0.9 -> {traj_hd[-1][0]:.3f}; a polymorphic ESS, not fixation.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(rps)
    return text, svg


def _svg(rps):
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Rock-Paper-Scissors orbits on the strategy simplex</text>')

    # simplex triangle vertices (R top, P bottom-left, S bottom-right)
    cx, cy = 320, 240
    R = 150
    verts = [(cx, cy - R), (cx - R * 0.866, cy + R * 0.5), (cx + R * 0.866, cy + R * 0.5)]

    def bary(x):
        # x = (x1,x2,x3) barycentric -> cartesian
        return (x[0] * verts[0][0] + x[1] * verts[1][0] + x[2] * verts[2][0],
                x[0] * verts[0][1] + x[1] * verts[1][1] + x[2] * verts[2][1])

    # triangle edges
    for a in range(3):
        b = (a + 1) % 3
        parts.append(f'<line x1="{verts[a][0]:.1f}" y1="{verts[a][1]:.1f}" '
                     f'x2="{verts[b][0]:.1f}" y2="{verts[b][1]:.1f}" stroke="{P["gray"]}" stroke-width="1"/>')
    labels = ["Rock", "Paper", "Scissors"]
    for a in range(3):
        parts.append(f'<text x="{verts[a][0]:.1f}" y="{verts[a][1] + (- 8 if a == 0 else 16):.1f}" '
                     f'fill="{P["text"]}" font-size="11" text-anchor="middle">{labels[a]}</text>')

    # center equilibrium
    ex, ey = bary([1 / 3, 1 / 3, 1 / 3])
    parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{P["red"]}"/>')
    parts.append(f'<text x="{ex + 8:.1f}" y="{ey + 4:.1f}" fill="{P["red"]}" font-size="10">Nash eq</text>')

    # several orbits from different starts, colored
    colors = [P["yellow"], P["green"], P["blue"], P["purple"]]
    starts = [[0.6, 0.3, 0.1], [0.15, 0.7, 0.15], [0.2, 0.2, 0.6], [0.45, 0.45, 0.1]]
    for si, x0 in enumerate(starts):
        _, traj = rd.integrate(rps, x0, t_max=60.0, dt=0.005, record_every=20)
        pts = " ".join(f"{bary(x)[0]:.1f},{bary(x)[1]:.1f}" for x in traj)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[si % 4]}" '
                     f'stroke-width="1" opacity="0.8"/>')
        sx, sy = bary(x0)
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="{colors[si % 4]}"/>')

    parts.append(f'<text x="20" y="{H - 24}" fill="{P["gray"]}" font-size="11">'
                 f'each colored orbit is a starting mix cycling around the Nash equilibrium (red) -- '
                 f'closed loops, never converging.</text>')
    parts.append(f'<text x="20" y="{H - 8}" fill="{P["gray"]}" font-size="11">'
                 f'the conserved product x1 x2 x3 pins each orbit to a level curve, like energy in a '
                 f'Hamiltonian system.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
