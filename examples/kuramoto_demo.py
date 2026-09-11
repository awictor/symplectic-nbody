"""Demo: the Kuramoto model -- oscillators falling into sync.

Prints the steady-state synchrony versus coupling strength, then draws the synchronization
transition (order parameter r rising past the critical coupling) alongside phase-circle
snapshots at weak and strong coupling.

    python examples/kuramoto_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kuramoto import (simulate, order_parameter, derivatives,  # noqa: E402
                      critical_coupling_lorentzian, _Rng)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Kuramoto: dtheta_i/dt = omega_i + (K/N) sum sin(theta_j - theta_i)\n")
    print("  Steady-state synchrony r vs coupling K (N=50, freq spread +/-1):\n")
    print(f"  {'K':>6}{'order r':>10}{'state':>18}")
    for k in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        r = simulate(50, k, omega_spread=1.0, seed=1)
        state = "incoherent" if r < 0.3 else ("partial" if r < 0.8 else "synchronized")
        print(f"  {k:>6.1f}{r:>10.3f}{state:>18}")

    print("\n  Below a critical coupling the oscillators drift independently (r ~ 0); above it a")
    print("  synchronized cluster spontaneously forms and r climbs toward 1 -- a phase")
    print("  transition to collective order. Same mechanism: fireflies flashing in unison,")
    print("  pacemaker cells, applause locking into rhythm, and generators on a power grid.")

    _svg(os.path.join(outdir, "kuramoto.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'kuramoto.svg')}")


def _evolve(n, k, seed, steps=500, dt=0.05):
    """Return the final phase set for a snapshot."""
    rng = _Rng(seed)
    omegas = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    phases = [rng.uniform(-math.pi, math.pi) for _ in range(n)]
    for _ in range(steps):
        d = derivatives(phases, omegas, k)
        phases = [phases[i] + dt * d[i] for i in range(n)]
    return phases


def _svg(path, size=720, pad=72):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">The Kuramoto synchronization transition</text>',
        f'<text x="20" y="48" fill="#8b949e" font-size="12">'
        f'order parameter r vs coupling (left); phase circles, scattered vs clustered (right)</text>',
    ]

    x0, x1 = pad, size - pad

    # --- left: r vs K ---
    lx0, lx1 = pad, size * 0.52
    ly0, ly1 = size - pad, pad + 40
    k_max = 8.0
    def KX(k):
        return lx0 + k / k_max * (lx1 - lx0)
    def RY(r):
        return ly0 - r * (ly0 - ly1)
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.4"/>')
    for r in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{lx0-6:.1f}" y="{RY(r)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{r:.1f}</text>')
    # critical coupling marker (Lorentzian-ish estimate for a spread ~1; use K_c ~ 1.6 for uniform)
    kc = critical_coupling_lorentzian(0.8)
    parts.append(f'<line x1="{KX(kc):.1f}" y1="{ly0:.1f}" x2="{KX(kc):.1f}" y2="{ly1:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>')
    parts.append(f'<text x="{KX(kc):.1f}" y="{ly1-4:.1f}" fill="#ff6b6b" font-size="10" '
                 f'text-anchor="middle">~K_c</text>')
    pts = []
    k = 0.0
    while k <= k_max:
        pts.append(f"{KX(k):.1f},{RY(simulate(40, k, 1.0, seed=1)):.1f}")
        k += 0.4
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')
    for k in (0, 2, 4, 6, 8):
        parts.append(f'<text x="{KX(k):.1f}" y="{ly0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">coupling K (order parameter r)</text>')

    # --- right: two phase-circle snapshots ---
    def circle(cx, cy, rad, phases, r_val, label):
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rad:.1f}" fill="none" '
                     f'stroke="#30363d" stroke-width="1.5"/>')
        # oscillators as dots on the circle
        for p in phases:
            px = cx + rad * math.cos(p)
            py = cy - rad * math.sin(p)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="#4dabf7"/>')
        # order-parameter vector
        c = sum(math.cos(p) for p in phases) / len(phases)
        s = sum(math.sin(p) for p in phases) / len(phases)
        parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx + rad*c:.1f}" y2="{cy - rad*s:.1f}" '
                     f'stroke="#ffd43b" stroke-width="2.4"/>')
        parts.append(f'<text x="{cx:.1f}" y="{cy + rad + 20:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{label} (r = {r_val:.2f})</text>')

    rcx = size * 0.77
    rad = 70
    weak = _evolve(40, 0.5, 1)
    strong = _evolve(40, 6.0, 1)
    circle(rcx, size * 0.36, rad, weak, order_parameter(weak), "weak K: scattered")
    circle(rcx, size * 0.68, rad, strong, order_parameter(strong), "strong K: clustered")

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
