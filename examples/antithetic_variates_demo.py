"""Antithetic variates demo: mirror-paired sampling reduces variance for monotone integrands (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import antithetic_variates as AV


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def main(outdir=None):
    true = math.e - 1
    f = lambda u: math.exp(u)

    lines = []
    lines.append("Antithetic variates: pair each draw with its mirror")
    lines.append("=" * 54)
    lines.append(f"estimate E[e^U], U ~ Uniform(0,1);  true = e - 1 = {true:.6f}")
    lines.append("each pair uses (U, 1-U): negatively correlated, so errors cancel")
    lines.append("")
    est, se, red = AV.estimate_uniform(f, n_pairs=20000, seed=1)
    cov = AV.pair_covariance(f, 20000, 1)
    lines.append(f"antithetic estimate:        {est:.6f}")
    lines.append(f"pair covariance Cov(f(U), f(1-U)): {cov:.4f}  (negative -> helps)")
    lines.append(f"variance reduction factor:  {red:.5f}")
    lines.append(f"effective sample multiplier:{1/red:>6.0f}x")
    lines.append("")
    lines.append("reduction depends on the integrand's shape (monotone -> big win, symmetric -> none):")
    lines.append(f"{'integrand':>18}{'reduction':>12}{'pair cov':>12}")
    tests = [("e^U (monotone)", lambda u: math.exp(u)),
             ("3U+1 (linear)", lambda u: 3 * u + 1),
             ("sqrt(U) (monotone)", lambda u: math.sqrt(u)),
             ("(U-0.5)^2 (symmetric)", lambda u: (u - 0.5) ** 2),
             ("sin(2*pi*U) (periodic)", lambda u: math.sin(2 * math.pi * u))]
    for name, g in tests:
        _, _, r = AV.estimate_uniform(g, 20000, seed=3)
        cvv = AV.pair_covariance(g, 20000, 3)
        lines.append(f"{name:>18}{r:>12.5f}{cvv:>12.5f}")
    lines.append("")
    lines.append("Monotone integrands have strongly negative pair covariance and shrink most;")
    lines.append("symmetric ones have f(U)=f(1-U), zero benefit -- know your integrand.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: show the paired points on the e^U curve, and the estimator-spread comparison
        W, H = 720, 380
        # left panel: the curve with a few antithetic pairs
        lx, ly, lw, lh = 40, 60, 300, 250

        def sx(u):
            return lx + u * lw

        def sy(v):
            return ly + lh - (v - 1) / (math.e - 1) * lh

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{lx}" y="30" fill="{TEXT}" font-size="14">'
                 f'antithetic pairs (U, 1-U) on f(u)=e^u</text>')
        curve = " ".join(f"{sx(i/200):.1f},{sy(math.exp(i/200)):.1f}" for i in range(201))
        s.append(f'<polyline points="{curve}" fill="none" stroke="{BLUE}" stroke-width="2"/>')
        rng = AV._Rng(7)
        for _ in range(7):
            u = rng.uniform()
            for uu, col in ((u, GREEN), (1 - u, RED)):
                s.append(f'<circle cx="{sx(uu):.1f}" cy="{sy(math.exp(uu)):.1f}" r="4" fill="{col}"/>')
            # dashed link showing the pair
            s.append(f'<line x1="{sx(u):.1f}" y1="{sy(math.exp(u)):.1f}" '
                     f'x2="{sx(1-u):.1f}" y2="{sy(math.exp(1-u)):.1f}" '
                     f'stroke="{GRAY}" stroke-width="0.6" stroke-dasharray="2,2"/>')
        s.append(f'<text x="{lx}" y="{ly+lh+18}" fill="{GRAY}" font-size="10">'
                 f'when one point (green) is high, its mirror (red) is low</text>')

        # right panel: estimator spread over runs, plain vs antithetic
        reps = 400
        pe = []
        ae = []
        for r in range(reps):
            pm, _ = AV.plain_estimate(f, n=200, seed=1000 + r)
            am, _, _ = AV.estimate_uniform(f, n_pairs=100, seed=1000 + r)
            pe.append(pm)
            ae.append(am)
        rx, ry, rw, rh = 390, 60, 290, 250
        lo, hi = true - 0.15, true + 0.15
        nb = 32

        def hist(v):
            c = [0] * nb
            for x in v:
                if lo <= x < hi:
                    c[int((x - lo) / (hi - lo) * nb)] += 1
            return c
        hp, ha = hist(pe), hist(ae)
        ymax = max(max(hp), max(ha))

        def rsx(x):
            return rx + (x - lo) / (hi - lo) * rw

        def rsy(c):
            return ry + rh - c / ymax * rh
        s.append(f'<text x="{rx}" y="30" fill="{TEXT}" font-size="14">'
                 f'estimator spread ({reps} runs, 200 evals)</text>')
        s.append(f'<line x1="{rx}" y1="{ry+rh}" x2="{rx+rw}" y2="{ry+rh}" stroke="{GRAY}"/>')
        bw = rw / nb
        for k in range(nb):
            x0 = lo + k * (hi - lo) / nb
            s.append(f'<rect x="{rsx(x0):.1f}" y="{rsy(hp[k]):.1f}" width="{bw-0.6:.1f}" '
                     f'height="{ry+rh-rsy(hp[k]):.1f}" fill="{RED}" fill-opacity="0.45"/>')
            s.append(f'<rect x="{rsx(x0):.1f}" y="{rsy(ha[k]):.1f}" width="{bw-0.6:.1f}" '
                     f'height="{ry+rh-rsy(ha[k]):.1f}" fill="{GREEN}" fill-opacity="0.55"/>')
        s.append(f'<line x1="{rsx(true):.1f}" y1="{ry}" x2="{rsx(true):.1f}" y2="{ry+rh}" '
                 f'stroke="{YELLOW}" stroke-width="1.5" stroke-dasharray="4,3"/>')
        s.append(f'<text x="{rx+10}" y="{ry+50}" fill="{RED}" font-size="10">plain MC</text>')
        s.append(f'<text x="{rx+10}" y="{ry+64}" fill="{GREEN}" font-size="10">antithetic</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "antithetic_variates.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
