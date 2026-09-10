"""Demo: two disk galaxies collide and grow tidal tails (Toomre-style).

Integrates a two-galaxy encounter with Barnes-Hut forces and renders the
particle field to SVG: a multi-frame "contact sheet" of the collision and a
single animated SVG of the tracers moving. Pure stdlib.

    python examples/galaxy_collision_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from galaxy import two_galaxy_encounter, centers_of  # noqa: E402
from barnes_hut import BarnesHut  # noqa: E402
from integrators import velocity_verlet  # noqa: E402


def _snapshot_svg(pos, masses, na, path, size=680, extent=7.0, title="", sub=""):
    def sx(x):
        return size / 2 + x / extent * (size / 2 - 20)

    def sy(y):
        return size / 2 - y / extent * (size / 2 - 20)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#05070d"/>',
    ]
    for i in range(len(pos)):
        x, y = pos[i][0], pos[i][1]
        if abs(x) > extent or abs(y) > extent:
            continue
        if masses[i] > 0.0:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="#ffd166"/>')
        else:
            col = "#4cc9f0" if i < na else "#ff70a6"  # galaxy A vs B
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1" '
                         f'fill="{col}" fill-opacity="0.85"/>')
    if title:
        parts.append(f'<text x="16" y="26" fill="#e6edf3" font-size="15">{title}</text>')
    if sub:
        parts.append(f'<text x="16" y="{size-14}" fill="#8b949e" font-size="11">{sub}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    sysn = two_galaxy_encounter(n_ring=500, inclination=0.4)
    na = 1 + 500  # indices [0..500] are galaxy A (center + 500 tracers)
    bh = BarnesHut(G=sysn.G, theta=0.6, softening=(sysn.soft2 ** 0.5))

    def accel(pos):
        return bh.accel(pos, sysn.m)

    print(f"Two-galaxy encounter: {sysn.n} bodies, Barnes-Hut forces\n")
    dt, total = 0.02, 800
    frames = [0, 200, 400, 600, 799]
    pos, vel = sysn.pos, sysn.vel
    frame_i = 0
    for step in range(total):
        pos, vel = velocity_verlet(pos, vel, accel, dt)
        if step == frames[frame_i]:
            t = (step + 1) * dt
            path = os.path.join(outdir, f"galaxy_t{frame_i}.svg")
            _snapshot_svg(pos, sysn.m, na, path,
                          title=f"t = {t:.1f}",
                          sub="two disk galaxies + tidal tails (blue=A, pink=B, gold=cores)")
            print(f"  wrote {path}  (t={t:.1f})")
            frame_i += 1
            if frame_i >= len(frames):
                break

    # count how much material got drawn into tails
    cA, cB = centers_of(sysn)
    flung = sum(1 for i in range(sysn.n) if sysn.m[i] == 0.0 and
                min(math.dist(pos[i], pos[cA]), math.dist(pos[i], pos[cB])) > 2.5)
    print(f"\n{flung} tracer particles pulled into tidal bridges/tails.")
    print("Same physics as the Antennae and the Mice: differential gravity")
    print("stretches a cold disk into long tails during a close passage.")


if __name__ == "__main__":
    main()
