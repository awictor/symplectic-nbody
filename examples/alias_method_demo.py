"""Demo: the alias method -- O(1) sampling from a weighted die.

Builds an alias table for a skewed distribution, confirms the empirical frequencies match the
weights, and shows the two-outcomes-per-column structure. Draws the target-vs-sampled bars and
each column's main/alias split.

    python examples/alias_method_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alias_method import AliasSampler, empirical_distribution, chi_square  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    labels = ["common", "sometimes", "rare", "uncommon"]
    weights = [12, 5, 1, 3]
    sampler = AliasSampler(weights, seed=7)

    print("Alias method: O(n) setup, then O(1) per weighted draw\n")
    draws = 100000
    emp = empirical_distribution(sampler, draws)
    target = sampler.probabilities()
    print(f"  {'outcome':>10}{'weight':>8}{'target':>9}{'sampled':>9}")
    for i, lab in enumerate(labels):
        print(f"  {lab:>10}{weights[i]:>8}{target[i]:>9.3f}{emp[i]:>9.3f}")
    print(f"\n  chi-square vs target over {draws} draws: {chi_square(sampler, draws):.2f} "
          f"({len(weights)-1} dof, 5% critical ~ 7.8) -> matches\n")

    print("  The alias table (each column holds a main outcome + an alias, both equal-area):")
    print(f"  {'column':>8}{'P(main)':>9}{'main':>12}{'alias':>12}")
    for i in range(sampler.n):
        print(f"  {i:>8}{sampler.prob[i]:>9.3f}{labels[i]:>12}{labels[sampler.alias[i]]:>12}")
    print("\n  A draw is one integer roll (pick a column) plus one float (main or its alias) --")
    print("  two operations, no search, however many outcomes there are. It is the standard for")
    print("  loot tables, particle spawning, and any hot loop sampling the same distribution.")

    _svg(os.path.join(outdir, "alias_method.svg"), labels, weights, sampler, emp)
    print(f"\n  wrote {os.path.join(outdir, 'alias_method.svg')}")


def _svg(path, labels, weights, sampler, emp, w=760, h=390):
    n = sampler.n
    target = sampler.probabilities()

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Alias method: sampled frequencies match the weights</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'target vs sampled probability (left); the equal-area alias columns (right)</text>',
    ]

    # left: target vs sampled bars
    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 60, 70
    pmax = max(max(target), max(emp)) * 1.15

    def LY(p):
        return ly0 - p / pmax * (ly0 - ly1)

    slot = (lx1 - lx0) / n
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    for i in range(n):
        cx = lx0 + (i + 0.5) * slot
        bw = slot * 0.3
        parts.append(f'<rect x="{cx-bw-1:.1f}" y="{LY(target[i]):.1f}" width="{bw:.1f}" '
                     f'height="{ly0-LY(target[i]):.1f}" fill="#4dabf7"/>')
        parts.append(f'<rect x="{cx+1:.1f}" y="{LY(emp[i]):.1f}" width="{bw:.1f}" '
                     f'height="{ly0-LY(emp[i]):.1f}" fill="#06d6a0"/>')
        parts.append(f'<text x="{cx:.1f}" y="{ly0+14:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{labels[i][:6]}</text>')
    parts.append(f'<rect x="{lx0+8}" y="{ly1}" width="9" height="9" fill="#4dabf7"/>'
                 f'<text x="{lx0+21}" y="{ly1+8}" fill="#e6edf3" font-size="9">target</text>')
    parts.append(f'<rect x="{lx0+8}" y="{ly1+14}" width="9" height="9" fill="#06d6a0"/>'
                 f'<text x="{lx0+21}" y="{ly1+22}" fill="#e6edf3" font-size="9">sampled</text>')

    # right: the alias columns, each a unit-height bar split into main (bottom) and alias (top)
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 60, 70
    colw = (rx1 - rx0) / n
    colors = ["#4dabf7", "#ffd43b", "#ff6b6b", "#8338ec", "#06d6a0", "#ff922b", "#b197fc"]
    barh = ry0 - ry1
    for i in range(n):
        x = rx0 + i * colw
        pmain = sampler.prob[i]
        # bottom part: the main outcome i, height pmain
        mh = barh * pmain
        parts.append(f'<rect x="{x+2:.1f}" y="{ry0-mh:.1f}" width="{colw-4:.1f}" height="{mh:.1f}" '
                     f'fill="{colors[i % len(colors)]}" opacity="0.85"/>')
        # top part: the alias outcome
        ah = barh * (1 - pmain)
        if ah > 0.5:
            ai = sampler.alias[i]
            parts.append(f'<rect x="{x+2:.1f}" y="{ry1:.1f}" width="{colw-4:.1f}" height="{ah:.1f}" '
                         f'fill="{colors[ai % len(colors)]}" opacity="0.85"/>')
        parts.append(f'<text x="{x+colw/2:.1f}" y="{ry0+14:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">col{i}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry1-6:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="middle">each column: main (bottom) + alias (top), equal area</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
