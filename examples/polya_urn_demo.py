"""Polya urn demo: rich-get-richer runs freezing into different random limits, matching the Beta law."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import polya_urn as pu


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}
COLORS = ["#4dabf7", "#ffd43b", "#06d6a0", "#b197fc", "#ff6b6b", "#f783ac", "#63e6be", "#ffa94d"]


def main():
    lines = []
    lines.append("Polya urn -- reinforcement: the rich get richer, the limit is random")
    lines.append("=" * 68)
    lines.append("")

    a, b, c = 2, 3, 1
    lines.append(f"Start: {a} black, {b} white. Each draw returns the ball plus {c} of its colour.")
    lines.append(f"Martingale value E[black fraction] = a/(a+b) = {pu.expected_fraction(a, b):.3f} at every step.")
    lines.append("")
    lines.append("Ten runs freeze into DIFFERENT stable fractions (each fixed by early luck):")
    lines.append("   run    final black fraction")
    lines.append("   " + "-" * 30)
    finals = []
    for r in range(10):
        f = pu.black_fraction(a, b, c, 3000, seed=(r + 1) * 7919)
        finals.append(f)
        lines.append(f"   {r+1:3d}    {f:.3f}")
    lines.append(f"  spread: {min(finals):.3f}..{max(finals):.3f} -- no single limit, unlike Ehrenfest")
    lines.append("")

    # Beta law match
    n_runs = 4000
    fr = [pu.black_fraction(a, b, c, 2000, seed=(r + 1) * 7919) for r in range(n_runs)]
    emp_mean = sum(fr) / n_runs
    emp_var = sum((x - emp_mean) ** 2 for x in fr) / n_runs
    lines.append(f"Limiting distribution is Beta({a}, {b}):")
    lines.append(f"  mean:     empirical {emp_mean:.4f}   Beta {pu.beta_mean(a, b):.4f}")
    lines.append(f"  variance: empirical {emp_var:.4f}   Beta {pu.beta_variance(a, b):.4f}")
    lines.append("")

    # exchangeability
    lines.append("Exchangeability -- any ordering of the same draws is equally likely:")
    for seq in (["B", "W", "B"], ["B", "B", "W"], ["W", "B", "B"]):
        lines.append(f"  P({''.join(seq)}) = {pu.sequence_probability(seq, a, b, c):.5f}")
    lines.append("  (de Finetti: this is exactly what makes the limit a random Beta frequency.)")

    text = "\n".join(lines)
    print(text)

    svg = _svg(a, b, c)
    return text, svg


def _svg(a, b, c):
    W, H = 640, 450
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Polya urn: fraction trajectories (top) freezing into the Beta limit (bottom)</text>')

    # TOP: several trajectories of the black fraction over time
    tx0, tx1, ty0, ty1 = 55, 610, 50, 210
    T = 500

    def tx(t):
        return tx0 + t / T * (tx1 - tx0)

    def ty(f):
        return ty1 - f * (ty1 - ty0)

    # martingale value line
    ev = pu.expected_fraction(a, b)
    parts.append(f'<line x1="{tx0}" y1="{ty(ev):.1f}" x2="{tx1}" y2="{ty(ev):.1f}" '
                 f'stroke="{P["gray"]}" stroke-width="1" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{tx1 - 70}" y="{ty(ev) - 4:.1f}" fill="{P["gray"]}" font-size="10">a/(a+b)</text>')
    for r in range(8):
        _draws, counts = pu.simulate(a, b, c, T, seed=(r + 1) * 7919)
        pts = " ".join(f"{tx(t):.1f},{ty(counts[t] / (a + b + c * t)):.1f}" for t in range(T + 1))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[r % len(COLORS)]}" '
                     f'stroke-width="1" opacity="0.8"/>')
    parts.append(f'<text x="{tx0}" y="{ty0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'black fraction vs draws: each run wanders early then locks onto its own random limit</text>')

    # BOTTOM: histogram of final fractions vs Beta pdf
    bx0, bx1, by0, by1 = 55, 610, 260, 410
    n_runs = 4000
    fr = [pu.black_fraction(a, b, c, 2000, seed=(r + 1) * 7919) for r in range(n_runs)]
    nbins = 40
    hist = [0] * nbins
    for f in fr:
        hist[min(nbins - 1, int(f * nbins))] += 1
    hmax = max(hist)

    def bx(i):
        return bx0 + i / nbins * (bx1 - bx0)

    def by(h):
        return by1 - h / hmax * (by1 - by0)

    for i in range(nbins):
        h = by1 - by(hist[i])
        parts.append(f'<rect x="{bx(i):.1f}" y="{by(hist[i]):.1f}" width="{(bx1-bx0)/nbins - 0.5:.1f}" '
                     f'height="{h:.1f}" fill="{P["blue"]}" opacity="0.6"/>')
    # Beta pdf overlay (scaled)
    pdf_max = max(pu.beta_pdf((i + 0.5) / nbins, a, b) for i in range(nbins))
    pts = " ".join(f"{bx(i) + (bx1-bx0)/nbins/2:.1f},{by1 - pu.beta_pdf((i+0.5)/nbins, a, b)/pdf_max*(by1-by0):.1f}"
                   for i in range(nbins))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["yellow"]}" stroke-width="2"/>')
    parts.append(f'<line x1="{bx0}" y1="{by1}" x2="{bx1}" y2="{by1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{bx0}" y="{by0 - 6:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'histogram of {n_runs} final fractions vs the Beta({a},{b}) density (yellow)</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'reinforcement freezes early randomness into a permanent Beta-distributed limit -- '
                 f'the opposite of Ehrenfest\'s single equilibrium.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
