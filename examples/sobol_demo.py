"""Sobol demo: quasi-random vs random point clouds, and QMC vs Monte Carlo convergence (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sobol as S
import low_discrepancy as LD


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _cloud(s, pts, ox, oy, side, title, color):
    s.append(f'<text x="{ox+side/2:.0f}" y="{oy-8}" fill="{TEXT}" font-size="12" '
             f'text-anchor="middle">{title}</text>')
    s.append(f'<rect x="{ox}" y="{oy}" width="{side}" height="{side}" fill="#161b22" '
             f'stroke="{GRAY}" stroke-width="0.6"/>')
    # dyadic grid to show stratification
    for g in (4,):
        for i in range(1, g):
            t = i / g
            s.append(f'<line x1="{ox+t*side:.1f}" y1="{oy}" x2="{ox+t*side:.1f}" y2="{oy+side}" '
                     f'stroke="#21262d"/>')
            s.append(f'<line x1="{ox}" y1="{oy+t*side:.1f}" x2="{ox+side}" y2="{oy+t*side:.1f}" '
                     f'stroke="#21262d"/>')
    for (x, y) in pts:
        s.append(f'<circle cx="{ox+x*side:.1f}" cy="{oy+side-y*side:.1f}" r="2.2" fill="{color}"/>')


def _convergence(s, ox, oy, w, h):
    f = lambda x: math.exp(x[0] + x[1])
    true = (math.e - 1) ** 2
    Ns = [64, 128, 256, 512, 1024, 2048, 4096]
    sob_e = []
    mc_e = []
    for N in Ns:
        sob_e.append(max(abs(S.qmc_integrate(f, 2, N) - true), 1e-9))
        rnd = _lcg(123 + N)
        trials = [abs(sum(f([rnd(), rnd()]) for _ in range(N)) / N - true) for _ in range(10)]
        mc_e.append(max(sum(trials) / len(trials), 1e-9))
    import math as _m
    lx = [_m.log10(N) for N in Ns]
    ally = [_m.log10(v) for v in sob_e + mc_e]
    xmin, xmax = min(lx), max(lx)
    ymin, ymax = min(ally), max(ally)
    yr = ymax - ymin or 1

    def sx(v):
        return ox + (v - xmin) / (xmax - xmin) * w

    def sy(v):
        return oy + (ymax - v) / yr * h

    s.append(f'<text x="{ox+w/2:.0f}" y="{oy-8}" fill="{TEXT}" font-size="12" '
             f'text-anchor="middle">integration error vs N (log-log)</text>')
    s.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" fill="none" stroke="{GRAY}" '
             f'stroke-width="0.6"/>')
    for series, col, lab in ((sob_e, GREEN, "Sobol QMC ~1/N"), (mc_e, RED, "random MC ~1/sqrt(N)")):
        d = " ".join(f"{sx(lx[i]):.1f},{sy(_m.log10(series[i])):.1f}" for i in range(len(Ns)))
        s.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for i in range(len(Ns)):
            s.append(f'<circle cx="{sx(lx[i]):.1f}" cy="{sy(_m.log10(series[i])):.1f}" r="2.5" '
                     f'fill="{col}"/>')
    s.append(f'<text x="{ox+10}" y="{oy+16}" fill="{GREEN}" font-size="10">Sobol QMC (~1/N)</text>')
    s.append(f'<text x="{ox+10}" y="{oy+30}" fill="{RED}" font-size="10">random MC (~1/sqrt N)</text>')
    for i in (0, len(Ns) - 1):
        s.append(f'<text x="{sx(lx[i]):.1f}" y="{oy+h+14}" fill="{GRAY}" font-size="9" '
                 f'text-anchor="middle">{Ns[i]}</text>')


def main(outdir=None):
    N = 256
    sob = S.sobol(N, 2)
    hal = LD.halton(N, 2)
    rnd = _lcg(2026)
    rand = [[rnd(), rnd()] for _ in range(N)]

    lines = []
    lines.append("Sobol low-discrepancy sequence")
    lines.append("=" * 50)
    lines.append(f"star discrepancy at N={N} (lower = more uniform):")
    lines.append(f"  Sobol   {LD.star_discrepancy(sob, 3000):.5f}")
    lines.append(f"  Halton  {LD.star_discrepancy(hal, 3000):.5f}")
    lines.append(f"  random  {LD.star_discrepancy(rand, 3000):.5f}")
    lines.append("")
    lines.append("quasi-Monte Carlo vs Monte Carlo: integral of exp(x+y) on [0,1]^2")
    f = lambda x: math.exp(x[0] + x[1])
    true = (math.e - 1) ** 2
    lines.append(f"  true value = (e-1)^2 = {true:.8f}")
    lines.append(f"{'N':>8}{'Sobol QMC err':>16}{'random MC err':>16}{'speedup':>10}")
    for Nq in (256, 1024, 4096, 16384):
        q = abs(S.qmc_integrate(f, 2, Nq) - true)
        r = _lcg(7 + Nq)
        trials = [abs(sum(f([r(), r()]) for _ in range(Nq)) / Nq - true) for _ in range(10)]
        mc = sum(trials) / len(trials)
        lines.append(f"{Nq:>8}{q:>16.2e}{mc:>16.2e}{mc/q:>9.1f}x")
    lines.append("")
    lines.append("Sobol error falls ~1/N; random ~1/sqrt(N). The gap widens with N.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        W, H = 760, 440
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="30" y="26" fill="{TEXT}" font-size="15">'
                 f'Sobol quasi-random points fill space evenly; random ones clump</text>')
        _cloud(s, sob, 40, 55, 200, f"Sobol ({N} pts)", GREEN)
        _cloud(s, rand, 270, 55, 200, f"random ({N} pts)", RED)
        _convergence(s, 510, 55, 210, 200)
        s.append(f'<text x="40" y="{H-18}" fill="{GRAY}" font-size="10">'
                 f'Left/middle: the random cloud leaves gaps and clusters; Sobol places exactly one '
                 f'point per dyadic cell. Right: the payoff -- QMC error decays far faster.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "sobol.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
