"""Demo: Boids -- flocking from three local rules.

Prints how the flock's alignment (polarization) rises from a random scatter as separation,
alignment, and cohesion act, then draws the flock as heading arrows at the start and after it
has organized, plus polarization over time.

    python examples/boids_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from boids import (random_flock, step, evolve, polarization,  # noqa: E402
                   mean_nearest_distance)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Boids: separation + alignment + cohesion -> emergent flocking, no leader\n")
    flock = random_flock(40, size=100.0, seed=1)
    print(f"  {'step':>8}{'polarization':>16}{'mean spacing':>16}")
    f = flock
    for target in (0, 30, 80, 200):
        f = evolve(flock, target, size=100.0) if target else flock
        print(f"  {target:>8}{polarization(f):>16.3f}{mean_nearest_distance(f):>16.2f}")

    print("\n  From a random scatter (polarization near 0) the flock aligns into coherent motion")
    print("  (polarization toward 1) while separation keeps the birds from colliding -- the")
    print("  murmuration of starlings, the bait ball of sardines, all bottom-up from local rules.")

    _svg(os.path.join(outdir, "boids.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'boids.svg')}")


def _draw_flock(parts, flock, x0, y0, panel, label):
    sc = panel / 100.0
    parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{panel:.1f}" height="{panel:.1f}" '
                 f'fill="none" stroke="#21262d" stroke-width="1"/>')
    for b in flock:
        px, py = x0 + b[0] * sc, y0 + b[1] * sc
        sp = math.hypot(b[2], b[3]) or 1.0
        dx, dy = b[2] / sp * 7, b[3] / sp * 7
        parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px+dx:.1f}" y2="{py+dy:.1f}" '
                     f'stroke="#4dabf7" stroke-width="1.4"/>')
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.6" fill="#4dabf7"/>')
    parts.append(f'<text x="{x0 + panel/2:.1f}" y="{y0 + panel + 16:.1f}" fill="#8b949e" '
                 f'font-size="10" text-anchor="middle">{label}</text>')


def _svg(path, size=720, pad=50):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">Boids: emergent flocking</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'random scatter (left) organizes into aligned motion (middle); polarization over time (right)</text>',
    ]

    flock = random_flock(50, size=100.0, seed=2)
    aligned = evolve(flock, 200, size=100.0)

    panel = 200
    _draw_flock(parts, flock, pad, 70, panel, "t=0: random headings")
    _draw_flock(parts, aligned, pad + panel + 30, 70, panel, "t=200: flocking")

    # polarization over time
    rx0, rx1 = pad + 2 * (panel + 30), size - pad
    ry0, ry1 = 70 + panel, 90
    def TX(t, tmax):
        return rx0 + t / tmax * (rx1 - rx0)
    def PY(p):
        return ry0 - p * (ry0 - ry1)
    tmax = 200
    f = flock
    pts = [f"{TX(0, tmax):.1f},{PY(polarization(f)):.1f}"]
    for t in range(1, tmax + 1):
        f = step(f, size=100.0)
        if t % 4 == 0:
            pts.append(f"{TX(t, tmax):.1f},{PY(polarization(f)):.1f}")
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    for p in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{rx0-6:.1f}" y="{PY(p)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{p:.1f}</text>')
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">time -> polarization</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
