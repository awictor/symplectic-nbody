"""Ehrenfest urn demo: irreversible diffusion from reversible moves, relaxation to equilibrium, and 2^N recurrence."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ehrenfest_urn as eu


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Ehrenfest urn -- irreversible diffusion from reversible microscopic moves")
    lines.append("=" * 74)
    lines.append("")

    n = 50
    lines.append(f"{n} balls, all starting in urn A. Each step moves one random ball to the other urn.")
    lines.append("")

    # single trajectory relaxes toward N/2
    counts = eu.simulate(n, n, 400, seed=3)
    lines.append("A single trajectory (count in urn A) relaxes from N toward N/2 then fluctuates:")
    lines.append("   step     count A")
    lines.append("   " + "-" * 22)
    for t in (0, 20, 50, 100, 200, 400):
        lines.append(f"   {t:4d}     {counts[t]}")
    lines.append(f"  equilibrium is N/2 = {n//2}; fluctuations of order sqrt(N) ~ {n**0.5:.1f} persist.")
    lines.append("")

    # stationary is binomial
    lines.append("Stationary distribution is Binomial(N, 1/2) -- each ball independently in either urn:")
    pi = eu.stationary_distribution(n)
    peak = pi.index(max(pi))
    lines.append(f"  peak at k={peak} (=N/2), pi_peak={pi[peak]:.4f}, pi at all-in-one={pi[0]:.2e}")
    lines.append(f"  detailed balance residual: {eu.detailed_balance_residual(n):.1e} (reversible chain)")
    lines.append("")

    # entropy rise
    lines.append("Entropy of the ensemble distribution rises as the gas spreads (Boltzmann's H-theorem):")
    lines.append("   step     entropy (bits)")
    lines.append("   " + "-" * 28)
    T_marks = [0, 5, 15, 40, 100]
    n_runs = 3000
    end_dists = {}
    for T in T_marks:
        dist = [0] * (n + 1)
        for r in range(n_runs):
            c = eu.simulate(n, n, T, seed=r + 1)
            dist[c[-1]] += 1
        lines.append(f"   {T:4d}     {eu.entropy(dist):.3f}")
    lines.append("")

    # recurrence
    lines.append("Poincare recurrence IS satisfied -- but the mean return time to all-in-one is 2^N:")
    for nn in (10, 30, 50, 100):
        lines.append(f"   N={nn:3d}: mean recurrence = 2^{nn} = {2**nn:.3e} steps")
    lines.append("  (for N=100 that dwarfs the age of the universe -- why gases never un-mix).")

    text = "\n".join(lines)
    print(text)

    svg = _svg(n, counts)
    return text, svg


def _svg(n, counts):
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Relaxation to equilibrium (top) and binomial stationary law (bottom)</text>')

    # TOP: trajectory relaxing to N/2
    T = len(counts) - 1
    tx0, tx1, ty0, ty1 = 55, 610, 50, 200

    def tx(t):
        return tx0 + t / T * (tx1 - tx0)

    def ty(k):
        return ty1 - k / n * (ty1 - ty0)

    # N/2 equilibrium line
    parts.append(f'<line x1="{tx0}" y1="{ty(n/2):.1f}" x2="{tx1}" y2="{ty(n/2):.1f}" '
                 f'stroke="{P["green"]}" stroke-width="1" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{tx1 - 40}" y="{ty(n/2) - 4:.1f}" fill="{P["green"]}" font-size="10">N/2</text>')
    pts = " ".join(f"{tx(t):.1f},{ty(counts[t]):.1f}" for t in range(T + 1))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{P["yellow"]}" stroke-width="1"/>')
    parts.append(f'<text x="{tx0}" y="{ty0 - 4:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'count in urn A: starts at N, decays to N/2, then fluctuates (never un-mixes)</text>')

    # BOTTOM: binomial stationary distribution
    pi = eu.stationary_distribution(n)
    bx0, bx1, by0, by1 = 55, 610, 250, 400
    pmax = max(pi)
    bw = (bx1 - bx0) / (n + 1)

    def by(p):
        return by1 - p / pmax * (by1 - by0)

    for k in range(n + 1):
        h = by1 - by(pi[k])
        parts.append(f'<rect x="{bx0 + k * bw:.1f}" y="{by(pi[k]):.1f}" width="{max(bw-0.5,1):.1f}" '
                     f'height="{h:.1f}" fill="{P["blue"]}" opacity="0.8"/>')
    parts.append(f'<line x1="{bx0 + (n/2) * bw + bw/2:.1f}" y1="{by0}" x2="{bx0 + (n/2) * bw + bw/2:.1f}" '
                 f'y2="{by1}" stroke="{P["green"]}" stroke-width="1" stroke-dasharray="3,3"/>')
    parts.append(f'<line x1="{bx0}" y1="{by1}" x2="{bx1}" y2="{by1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{bx0}" y="{by0 - 6:.1f}" fill="{P["gray"]}" font-size="10">'
                 f'stationary pi_k = C(N,k)/2^N -- sharply peaked at N/2, negligible at the extremes</text>')

    parts.append(f'<text x="20" y="{H - 10}" fill="{P["gray"]}" font-size="11">'
                 f'every move is reversible, yet the count drifts irreversibly to N/2 -- irreversibility '
                 f'is overwhelming probability, not a law.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
