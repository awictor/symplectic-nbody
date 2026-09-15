"""Wright-Fisher demo: allele-frequency drift trajectories, fixation vs frequency, and the selection advantage."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import wright_fisher as wf


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Wright-Fisher model -- genetic drift, fixation, and selection")
    lines.append("=" * 62)
    lines.append("")

    two_n = 50
    lines.append(f"Population 2N={two_n} gene copies.")
    lines.append("")

    # neutral fixation probability = initial frequency
    lines.append("Neutral fixation probability equals the starting frequency (fair game):")
    lines.append("   p0     simulated fix. prob   theory (=p0)")
    lines.append("   " + "-" * 42)
    for p0 in (0.1, 0.3, 0.5, 0.7, 0.9):
        est = wf.fixation_probability(two_n, p0, s=0.0, n_runs=1000, seed=1)
        lines.append(f"   {p0:4.1f}   {est:.3f}                {p0:.3f}")
    lines.append("")

    # selection raises fixation, matches Kimura
    lines.append("A new beneficial mutant (p0=1/2N) -- selection vs Kimura diffusion formula:")
    lines.append("   s       simulated fix.   Kimura       neutral (=1/2N)")
    lines.append("   " + "-" * 52)
    p0 = 1.0 / two_n
    for s in (0.0, 0.02, 0.05, 0.1, 0.2):
        est = wf.fixation_probability(two_n, p0, s=s, n_runs=4000, seed=1)
        kim = wf.kimura_fixation_probability(two_n, p0, s)
        lines.append(f"   {s:5.2f}   {est:.4f}          {kim:.4f}       {p0:.4f}")
    lines.append("")

    # heterozygosity decay
    lines.append("Heterozygosity (genetic variation) decays by 1-1/(2N) per generation:")
    lines.append("   generation   H (2N=50)   H (2N=500)")
    lines.append("   " + "-" * 36)
    for g in (0, 25, 50, 100, 200):
        h50 = wf.heterozygosity_decay(50, 0.5, g)
        h500 = wf.heterozygosity_decay(500, 0.5, g)
        lines.append(f"   {g:6d}       {h50:.4f}      {h500:.4f}")
    lines.append("")
    lines.append("Small populations lose variation fast; large ones retain it -- drift scales as 1/N.")

    text = "\n".join(lines)
    print(text)

    svg = _svg(two_n)
    return text, svg


def _svg(two_n):
    W, H = 640, 440
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="24" fill="{P["text"]}" font-size="15">'
                 f'Wright-Fisher drift: many neutral trajectories from p0=0.5</text>')

    x0, x1, y0, y1 = 50, 615, 50, 340
    max_gen = 200

    def px(g):
        return x0 + g / max_gen * (x1 - x0)

    def py(p):
        return y1 - p * (y1 - y0)

    # absorbing boundaries
    parts.append(f'<line x1="{x0}" y1="{py(1.0):.1f}" x2="{x1}" y2="{py(1.0):.1f}" '
                 f'stroke="{P["gray"]}" stroke-width="1" stroke-dasharray="3,3"/>')
    parts.append(f'<line x1="{x0}" y1="{py(0.0):.1f}" x2="{x1}" y2="{py(0.0):.1f}" '
                 f'stroke="{P["gray"]}" stroke-width="1" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{x1 - 60}" y="{py(1.0) - 4:.1f}" fill="{P["gray"]}" font-size="10">fixation</text>')
    parts.append(f'<text x="{x1 - 40}" y="{py(0.0) + 12:.1f}" fill="{P["gray"]}" font-size="10">loss</text>')

    # simulate several trajectories, color by outcome
    n_traj = 25
    fixed_count = 0
    for r in range(n_traj):
        traj, is_fixed = wf.simulate(two_n, 0.5, s=0.0, max_gen=max_gen, seed=r + 1)
        if is_fixed:
            fixed_count += 1
        col = P["green"] if is_fixed else P["red"] if traj[-1] == 0.0 else P["yellow"]
        # clip to max_gen
        pts = " ".join(f"{px(g):.1f},{py(traj[g]):.1f}" for g in range(min(len(traj), max_gen + 1)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1" opacity="0.7"/>')

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{x0 - 8}" y="{py(0.5) + 4:.1f}" fill="{P["gray"]}" font-size="10" '
                 f'text-anchor="end">0.5</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1 + 20:.1f}" fill="{P["gray"]}" '
                 f'font-size="11" text-anchor="middle">generation</text>')
    parts.append(f'<text x="20" y="{H - 24}" fill="{P["gray"]}" font-size="11">'
                 f'{n_traj} neutral runs from p0=0.5: {fixed_count} fixed (green), rest lost (red). '
                 f'Each is a random walk to an absorbing barrier.</text>')
    parts.append(f'<text x="20" y="{H - 8}" fill="{P["gray"]}" font-size="11">'
                 f'~half fix and half are lost, as the martingale predicts -- but each single lineage '
                 f'is pure chance.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
