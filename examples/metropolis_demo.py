"""Demo: Metropolis Monte Carlo on the 2D Ising model.

Prints the average magnetization measured by simulation as temperature crosses the Onsager
critical point, then draws the simulated m(T) curve dropping to zero at T_c (compared to the
exact 2.269) alongside a snapshot of the spin configuration near criticality.

    python examples/metropolis_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metropolis import run, sweep, magnetization, _Rng, ONSAGER_TC  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Metropolis MCMC: accept a flip with prob min(1, exp(-dE/T)) -- samples exp(-E/T)\n")
    print("  2D Ising ferromagnet, exact Onsager T_c = %.3f J/k_B\n" % ONSAGER_TC)
    print(f"  {'T (J/kB)':>10}{'avg |m|':>12}{'phase':>16}")
    for T in (1.0, 1.8, 2.27, 2.6, 3.5):
        m = run(20, T, sweeps=160, seed=1)
        phase = "ordered" if m > 0.5 else "disordered"
        print(f"  {T:>10.2f}{m:>12.3f}{phase:>16}")

    print("\n  The simulation reproduces the real phase transition mean-field theory only")
    print("  approximates: spins order below ~2.27 J/k_B and disorder above it, with the true")
    print("  Onsager critical temperature and genuine critical fluctuations near T_c.")

    _svg(os.path.join(outdir, "metropolis.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'metropolis.svg')}")


def _svg(path, size=720, pad=76):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Metropolis Monte Carlo: the 2D Ising transition</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'simulated magnetization drops to zero at the Onsager T_c (left); a near-critical spin snapshot (right)</text>',
    ]

    # --- left: m(T) curve from simulation ---
    lx0, lx1 = pad, size * 0.54
    ly0, ly1 = size - pad, pad + 44
    T_min, T_max = 1.0, 3.6
    def TX(T):
        return lx0 + (T - T_min) / (T_max - T_min) * (lx1 - lx0)
    def MY(m):
        return ly0 - m * (ly0 - ly1)
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.4"/>')
    for m in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{lx0-6:.1f}" y="{MY(m)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{m:.1f}</text>')
    # T_c marker
    parts.append(f'<line x1="{TX(ONSAGER_TC):.1f}" y1="{ly0:.1f}" x2="{TX(ONSAGER_TC):.1f}" y2="{ly1:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1.2" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{TX(ONSAGER_TC):.1f}" y="{ly1-4:.1f}" fill="#ff6b6b" font-size="10" '
                 f'text-anchor="middle">T_c=2.27</text>')
    # simulated points
    pts = []
    for i in range(14):
        T = T_min + (T_max - T_min) * i / 13
        m = run(16, T, sweeps=100, seed=1)
        pts.append((T, m))
        parts.append(f'<circle cx="{TX(T):.1f}" cy="{MY(m):.1f}" r="3.5" fill="#4dabf7"/>')
    poly = " ".join(f"{TX(T):.1f},{MY(m):.1f}" for T, m in pts)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#4dabf7" stroke-width="1.6" opacity="0.6"/>')
    for T in (1.0, 2.0, 3.0):
        parts.append(f'<text x="{TX(T):.1f}" y="{ly0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{T:.0f}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">temperature (J/k_B) -- avg |m|</text>')

    # --- right: spin snapshot near T_c ---
    rx0 = size * 0.60
    ry0 = pad + 60
    m = 34
    rng = _Rng(seed=2)
    grid = [[1] * m for _ in range(m)]
    for _ in range(120):
        sweep(grid, 2.3, rng)
    cell = (size - pad - rx0) / m
    for r in range(m):
        for c in range(m):
            col = "#ffd43b" if grid[r][c] > 0 else "#1b3a6b"
            parts.append(f'<rect x="{rx0 + c*cell:.1f}" y="{ry0 + r*cell:.1f}" '
                         f'width="{cell+0.5:.1f}" height="{cell+0.5:.1f}" fill="{col}"/>')
    parts.append(f'<text x="{rx0 + m*cell/2:.1f}" y="{ry0 + m*cell + 18:.1f}" fill="#8b949e" '
                 f'font-size="10" text-anchor="middle">spins near T_c (up=yellow, down=blue): domains at every scale</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
