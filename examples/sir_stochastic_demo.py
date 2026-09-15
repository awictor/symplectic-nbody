"""Stochastic SIR demo: outbreak trajectories, the bimodal final size, and the R0 final-size equation."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import sir_stochastic as sir


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Stochastic SIR epidemic -- take-off, fizzle, and the final-size threshold")
    lines.append("=" * 72)
    lines.append("")

    n = 2000
    beta, gamma = 0.5, 0.2
    R0 = sir.r0(beta, gamma)
    lines.append(f"Population N={n}, beta={beta}, gamma={gamma} -> R0 = {R0}.")
    lines.append("")

    # ensemble of outbreaks from one infective
    sizes = [sir.simulate(n, 1, beta, gamma, seed=r + 1)["final_size"] for r in range(500)]
    minor = [s for s in sizes if s < 0.05 * n]
    major = [s for s in sizes if s >= 0.05 * n]
    lines.append(f"500 outbreaks from ONE infective (R0={R0} > 1) are BIMODAL:")
    lines.append(f"  {len(minor)} minor (fizzle out, mean size {sum(minor)/len(minor):.1f})")
    lines.append(f"  {len(major)} major (take off, mean size {sum(major)/len(major):.0f} = "
                 f"{sum(major)/len(major)/n*100:.0f}% of N)")
    lines.append(f"  minor fraction {len(minor)/len(sizes):.3f} vs theory 1/R0 = {1/R0:.3f}")
    lines.append("")

    # final-size equation
    lines.append("Major-outbreak size solves the final-size equation 1 - z = exp(-R0 z):")
    lines.append("   R0     final fraction z    fraction infected")
    lines.append("   " + "-" * 46)
    for r0v in (0.8, 1.2, 1.5, 2.5, 4.0, 8.0):
        z = sir.final_size_fraction(r0v)
        lines.append(f"   {r0v:4.1f}   {z:.4f}              {z*100:.1f}%")
    lines.append("")
    lines.append("Below R0=1 no epidemic; above, the fraction infected rises sharply toward 100%.")
    lines.append("")

    # take-off probability vs R0
    lines.append("Take-off (major outbreak) probability vs R0, from one infective:")
    lines.append("   R0     P(major) ~ 1 - 1/R0")
    lines.append("   " + "-" * 30)
    for r0v in (1.5, 2.0, 3.0, 5.0):
        lines.append(f"   {r0v:4.1f}   {1 - 1/r0v:.3f}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(n, beta, gamma, R0)
    return text, svg


def _svg(n, beta, gamma, R0):
    W, H = 640, 450
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Infected(t) over many outbreaks (top) and the bimodal final size (bottom)</text>')

    # TOP: several I(t) trajectories from a few infectives (major ones peak, minor fizzle)
    tx0, tx1, ty0, ty1 = 55, 610, 50, 210
    trajs = [sir.simulate(n, 3, beta, gamma, seed=r * 13 + 1) for r in range(30)]
    tmax = max(tr["duration"] for tr in trajs) or 1.0
    imax = max(max(tr["I"]) for tr in trajs) or 1

    def tx(t):
        return tx0 + t / tmax * (tx1 - tx0)

    def ty(i):
        return ty1 - i / imax * (ty1 - ty0)

    for tr in trajs:
        major = tr["final_size"] >= 0.05 * n
        col = P["red"] if major else P["gray"]
        # subsample points for the polyline
        step = max(1, len(tr["times"]) // 200)
        pts = " ".join(f"{tx(tr['times'][k]):.1f},{ty(tr['I'][k]):.1f}"
                       for k in range(0, len(tr["times"]), step))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="0.8" '
                     f'opacity="{0.8 if major else 0.5}"/>')
    parts.append(f'<line x1="{tx0}" y1="{ty1}" x2="{tx1}" y2="{ty1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{tx0}" y="{ty0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'I(t): red = major outbreaks (big peak), gray = minor (fizzle) -- same R0={R0}</text>')

    # BOTTOM: histogram of final sizes
    sizes = [sir.simulate(n, 1, beta, gamma, seed=r + 1)["final_size"] for r in range(500)]
    bx0, bx1, by0, by1 = 55, 610, 260, 410
    nbins = 40
    hist = [0] * nbins
    for s in sizes:
        hist[min(nbins - 1, int(s / n * nbins))] += 1
    hmax = max(hist)

    def bx(i):
        return bx0 + i / nbins * (bx1 - bx0)

    def by(h):
        return by1 - h / hmax * (by1 - by0)

    for i in range(nbins):
        h = by1 - by(hist[i])
        parts.append(f'<rect x="{bx(i):.1f}" y="{by(hist[i]):.1f}" width="{(bx1-bx0)/nbins - 0.5:.1f}" '
                     f'height="{h:.1f}" fill="{P["blue"]}" opacity="0.7"/>')
    # final-size-equation prediction line
    z = sir.final_size_fraction(R0)
    parts.append(f'<line x1="{bx(int(z*nbins)):.1f}" y1="{by0}" x2="{bx(int(z*nbins)):.1f}" y2="{by1}" '
                 f'stroke="{P["yellow"]}" stroke-width="1.5" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{bx(int(z*nbins)) - 60:.1f}" y="{by0 + 12:.1f}" fill="{P["yellow"]}" '
                 f'font-size="10">final-size eq z={z:.2f}</text>')
    parts.append(f'<line x1="{bx0}" y1="{by1}" x2="{bx1}" y2="{by1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{bx0}" y="{by0 - 6:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'final size / N over 500 outbreaks: a spike near 0 (fizzle) + a peak at z (major)</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'the same R0>1 epidemic either dies out early (prob ~1/R0) or infects the predicted '
                 f'fraction z -- chance decides which.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
