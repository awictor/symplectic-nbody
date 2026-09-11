"""Demo: the double pendulum -- deterministic chaos you can build.

Prints the energy conservation of the integrator and the divergence of two nearly identical
releases, then draws the wild trace of the lower bob and three snapshots of two pendulums
(started a hair apart) flailing into completely different configurations.

    python examples/double_pendulum_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from double_pendulum import (trajectory, total_energy, positions,  # noqa: E402
                             divergence_rate)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Double pendulum: deterministic yet chaotic (two rods, gravity)\n")
    e0s = (0.4, 0.0, 0.2, 0.0)
    E0 = total_energy(e0s)
    tr = trajectory(e0s, 1e-5, 5000)                # 0.05 s at a fine step
    print("  Energy conservation (RK4, dt=1e-5, 0.05 s run):")
    print(f"    E(0) = {E0:.6f} J,  E(end) = {total_energy(tr[-1]):.6f} J  (relative drift %.1e)"
          % (abs(total_energy(tr[-1]) - E0) / abs(E0)))
    print("    (RK4 is not symplectic, so energy slowly drifts over long chaotic runs; over a")
    print("     short well-resolved window it holds to a part in a million or better.)")

    print("\n  Two releases 1e-4 rad apart, angle-space separation over time:")
    s0 = (math.pi / 2, 0.0, math.pi / 2, 0.0)
    a = tuple(s0)
    b = (s0[0] + 1e-4, s0[1], s0[2], s0[3])
    from double_pendulum import rk4_step
    a, b = tuple(a), tuple(b)
    for step_i in range(0, 2001, 400):
        if step_i > 0:
            for _ in range(400):
                a = rk4_step(a, 0.005)
                b = rk4_step(b, 0.005)
        sep = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(4)))
        print(f"    t = {step_i*0.005:>5.1f} s  ->  separation {sep:.3e}")

    print("\n  A ten-thousandth of a radian grows until the two pendulums are doing utterly")
    print("  different things -- the same sensitive dependence that makes weather chaotic, in a")
    print("  system you can hang from a nail. Energy is conserved; the motion is still unpredictable.")

    _svg(os.path.join(outdir, "double_pendulum.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'double_pendulum.svg')}")


def _svg(path, size=760, pad=50):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">The double pendulum</text>',
        f'<text x="20" y="48" fill="#8b949e" font-size="12">'
        f'the lower bob&#39;s chaotic trace (left); two near-identical starts diverging (right)</text>',
    ]

    # --- left: trace of bob 2 ---
    lx0, lx1 = pad, size * 0.50
    lcx = (lx0 + lx1) / 2
    lcy = size * 0.42
    scale = (lx1 - lx0) / 2 / 2.2      # world radius ~2 (l1+l2) into half-panel
    tr = trajectory((math.pi / 2, 0.0, math.pi / 2, 0.0), 0.004, 9000)
    pts = []
    for s in tr[::2]:
        _, _, x2, y2 = positions(s)
        pts.append(f"{lcx + x2*scale:.1f},{lcy - y2*scale:.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" '
                 f'stroke-width="0.4" opacity="0.7"/>')
    parts.append(f'<circle cx="{lcx:.1f}" cy="{lcy:.1f}" r="3" fill="#e6edf3"/>')
    parts.append(f'<text x="{lcx:.1f}" y="{lcy - 2.2*scale - 6:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">pivot</text>')
    parts.append(f'<text x="{lcx:.1f}" y="{size-pad+4:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">lower-bob trace (never repeats)</text>')

    # --- right: three snapshots of two pendulums a hair apart ---
    rx = size * 0.72
    a = (math.pi / 2, 0.0, math.pi / 2, 0.0)
    b = (math.pi / 2 + 0.01, 0.0, math.pi / 2, 0.0)   # 0.01 rad apart for visibility
    from double_pendulum import rk4_step
    snap_times = [0, 700, 1400]
    ry_positions = [size * 0.30, size * 0.55, size * 0.80]
    sc = 46
    step_count = 0
    for si, target in enumerate(snap_times):
        while step_count < target:
            a = rk4_step(a, 0.006)
            b = rk4_step(b, 0.006)
            step_count += 1
        cy = ry_positions[si]
        for state, col in ((a, "#06d6a0"), (b, "#ff6b6b")):
            x1, y1, x2, y2 = positions(state)
            px1, py1 = rx + x1 * sc, cy - y1 * sc
            px2, py2 = rx + x2 * sc, cy - y2 * sc
            parts.append(f'<line x1="{rx:.1f}" y1="{cy:.1f}" x2="{px1:.1f}" y2="{py1:.1f}" '
                         f'stroke="{col}" stroke-width="1.6"/>')
            parts.append(f'<line x1="{px1:.1f}" y1="{py1:.1f}" x2="{px2:.1f}" y2="{py2:.1f}" '
                         f'stroke="{col}" stroke-width="1.6"/>')
            parts.append(f'<circle cx="{px2:.1f}" cy="{py2:.1f}" r="3" fill="{col}"/>')
        parts.append(f'<circle cx="{rx:.1f}" cy="{cy:.1f}" r="2.5" fill="#8b949e"/>')
        parts.append(f'<text x="{rx+70:.1f}" y="{cy:.1f}" fill="#8b949e" font-size="10">'
                     f't = {target*0.006:.1f} s</text>')
    parts.append(f'<text x="{rx:.1f}" y="{size-20:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">two starts 0.01 rad apart (green/red) drift apart</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
