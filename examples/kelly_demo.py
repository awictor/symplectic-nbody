"""Demo: the Kelly criterion -- how much to bet to grow fastest.

Prints the optimal fraction and growth rate against a Monte-Carlo run for a favorable bet, then
draws the growth-rate-versus-bet-fraction curve (peaking at f*, crossing zero where overbetting
begins) and sample compounding bankroll trajectories at under-, Kelly-, and over-betting.

    python examples/kelly_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kelly import (optimal_fraction, optimal_growth_rate, growth_rate,  # noqa: E402
                   break_even_fraction, doubling_time, edge, simulate_growth, _Rng)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    p, b = 0.6, 1.0
    fstar = optimal_fraction(p, b)
    print(f"Kelly criterion: even-money bet, win probability p={p}, edge {edge(p, b):.0%}\n")
    print(f"  optimal fraction f*     = {fstar:.3f}  (bet {fstar:.0%} of bankroll each time)")
    print(f"  growth rate g(f*)       = {optimal_growth_rate(p, b):.4f} per bet")
    print(f"  Monte-Carlo growth      = {simulate_growth(fstar, p, b, bets=500, trials=3000, seed=7):.4f}")
    print(f"  doubling time           = {doubling_time(p, b):.1f} bets")
    print(f"  break-even fraction     = {break_even_fraction(p, b):.3f}  (overbetting past this loses)")

    print(f"\n  {'fraction':>10}{'growth':>10}{'vs f*':>8}")
    for f in (0.05, 0.10, fstar, 0.30, 0.40, 0.50):
        g = growth_rate(f, p, b)
        tag = "  <- Kelly" if abs(f - fstar) < 1e-9 else ""
        print(f"  {f:>10.2f}{g:>10.4f}{g / optimal_growth_rate(p, b):>8.2f}{tag}")

    print("\n  Bet the Kelly fraction and the bankroll compounds fastest in the long run. Bet")
    print("  more and volatility eats the growth -- past 2f* the growth rate goes negative and")
    print("  you go broke despite a winning edge. Half-Kelly keeps ~3/4 the growth at far less")
    print("  risk, which is why real traders and gamblers bet fractional Kelly.")

    _svg(os.path.join(outdir, "kelly.svg"), p, b)
    print(f"\n  wrote {os.path.join(outdir, 'kelly.svg')}")


def _svg(path, p, b, w=760, h=380):
    fstar = optimal_fraction(p, b)
    gstar = optimal_growth_rate(p, b)
    be = break_even_fraction(p, b)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'The Kelly criterion: bet f* to grow fastest</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'growth rate vs bet fraction, peaking at f* (left); compounding bankrolls (right)</text>',
    ]

    # left: growth rate g(f) vs f
    lx0, lx1 = 58, w // 2 - 20
    ly0, ly1 = h - 55, 62
    fmax = min(be * 1.15, 0.99)
    gmin = growth_rate(fmax, p, b)
    gtop = gstar * 1.15

    def LX(f):
        return lx0 + f / fmax * (lx1 - lx0)

    def LY(g):
        return ly0 - (g - gmin) / (gtop - gmin) * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    # zero-growth line
    parts.append(f'<line x1="{lx0}" y1="{LY(0):.1f}" x2="{lx1}" y2="{LY(0):.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    pts = []
    for i in range(0, 201):
        f = fmax * i / 200
        pts.append(f"{LX(f):.1f},{LY(growth_rate(f, p, b)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    # mark f* (peak) and break-even
    parts.append(f'<line x1="{LX(fstar):.1f}" y1="{ly1}" x2="{LX(fstar):.1f}" y2="{ly0}" '
                 f'stroke="#06d6a0" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<circle cx="{LX(fstar):.1f}" cy="{LY(gstar):.1f}" r="3.5" fill="#06d6a0"/>')
    parts.append(f'<text x="{LX(fstar):.1f}" y="{LY(gstar)-7:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="middle">f* = {fstar:.2f}</text>')
    parts.append(f'<circle cx="{LX(be):.1f}" cy="{LY(0):.1f}" r="3" fill="#ff6b6b"/>')
    parts.append(f'<text x="{LX(be):.1f}" y="{LY(0)-7:.1f}" fill="#ff6b6b" font-size="9" '
                 f'text-anchor="middle">break-even</text>')
    parts.append(f'<text x="{lx0+4:.1f}" y="{ly1-2:.1f}" fill="#8b949e" font-size="9">growth g(f)</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+22:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">bet fraction f</text>')

    # right: sample bankroll trajectories (log scale) at 3 fractions
    rx0, rx1 = w // 2 + 45, w - 25
    ry0, ry1 = h - 55, 62
    bets = 300
    fracs = [(0.1, "#ffd43b", "0.10 (under)"),
             (fstar, "#06d6a0", f"{fstar:.2f} (Kelly)"),
             (0.45, "#ff6b6b", "0.45 (over)")]
    # compute log-bankroll paths with a shared RNG seed per fraction
    paths = []
    logmin, logmax = 0.0, 0.0
    for f, col, lab in fracs:
        rng = _Rng(seed=3)
        logw = 0.0
        walk = [0.0]
        for _ in range(bets):
            if rng.random() < p:
                logw += math.log(1 + b * f)
            else:
                logw += math.log(1 - f)
            walk.append(logw)
        paths.append((walk, col, lab))
        logmin = min(logmin, min(walk))
        logmax = max(logmax, max(walk))
    span = (logmax - logmin) or 1

    def RX(t):
        return rx0 + t / bets * (rx1 - rx0)

    def RY(lw):
        return ry0 - (lw - logmin) / span * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{RY(0):.1f}" x2="{rx1}" y2="{RY(0):.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    ly = ry1
    for walk, col, lab in paths:
        pts = " ".join(f"{RX(t):.1f},{RY(v):.1f}" for t, v in enumerate(walk))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        parts.append(f'<rect x="{rx0+8}" y="{ly}" width="9" height="9" fill="{col}"/>'
                     f'<text x="{rx0+21}" y="{ly+8}" fill="#e6edf3" font-size="9">{lab}</text>')
        ly += 14
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+22:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">bets -> log bankroll</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
