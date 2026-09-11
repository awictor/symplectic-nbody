"""Demo: Fisher-Yates -- the only correct shuffle.

Shuffles a list, then enumerates every permutation over many trials to show Fisher-Yates is
uniform while the naive swap-anywhere shuffle is measurably biased. Draws the permutation-
frequency histograms side by side (Fisher-Yates flat, naive lumpy).

    python examples/fisher_yates_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fisher_yates import (shuffle, naive_shuffle, sample_without_replacement,  # noqa: E402
                          sattolo_cycle, is_single_cycle, permutation_counts, chi_square_uniform)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Fisher-Yates: swap each position with a random one in the UNSHUFFLED tail\n")
    deck = list("ABCDEFGH")
    print(f"  {''.join(deck)}  ->  {''.join(shuffle(deck, seed=7))}\n")

    n, trials = 4, 60000
    nfact = math.factorial(n)
    fy = permutation_counts(n, trials, shuffle, seed=1)
    nv = permutation_counts(n, trials, naive_shuffle, seed=1)
    print(f"  Uniformity over all {nfact} permutations of {n} items, {trials} trials each:")
    print(f"    Fisher-Yates: {len(fy)}/{nfact} permutations seen, "
          f"chi-square {chi_square_uniform(fy, nfact, trials):.1f}  -> uniform")
    print(f"    naive swap:   {len(nv)}/{nfact} permutations seen, "
          f"chi-square {chi_square_uniform(nv, nfact, trials):.1f}  -> BIASED")
    print(f"    (23 dof, 5% critical ~ 35.2; the naive method fails by orders of magnitude)\n")

    print(f"  sample 5 without replacement from 0..19: {sample_without_replacement(range(20), 5, seed=3)}")
    sc = sattolo_cycle(list(range(8)), seed=1)
    print(f"  Sattolo cyclic shuffle of 0..7: {sc}  (single {8}-cycle: {is_single_cycle(sc)})")
    print("\n  The naive 'swap with any position' makes n^n equally likely swap sequences but")
    print("  only n! permutations -- and n^n isn't divisible by n!, so some orderings win. Only")
    print("  the shrinking range makes every permutation exactly equally likely.")

    _svg(os.path.join(outdir, "fisher_yates.svg"), n, fy, nv, trials)
    print(f"\n  wrote {os.path.join(outdir, 'fisher_yates.svg')}")


def _svg(path, n, fy, nv, trials, w=760, h=390):
    import itertools
    perms = [tuple(p) for p in itertools.permutations(range(n))]
    nfact = len(perms)
    expected = trials / nfact

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Fisher-Yates is flat; the naive shuffle is lumpy</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'how often each of the {nfact} permutations appears -- '
        f'Fisher-Yates (green) hugs the uniform line, naive (red) does not</text>',
    ]

    lx0, lx1 = 55, w - 30
    ly0, ly1 = h - 55, 70
    vmax = max(max(fy.get(p, 0) for p in perms), max(nv.get(p, 0) for p in perms)) * 1.1

    def X(i):
        return lx0 + (i + 0.5) / nfact * (lx1 - lx0)

    def Y(v):
        return ly0 - v / vmax * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    # uniform expectation line
    parts.append(f'<line x1="{lx0}" y1="{Y(expected):.1f}" x2="{lx1}" y2="{Y(expected):.1f}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{lx1-2:.1f}" y="{Y(expected)-4:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">uniform = trials/n!</text>')
    # naive bars (red) then FY bars (green), thin, side by side
    bw = (lx1 - lx0) / nfact * 0.36
    for i, p in enumerate(perms):
        cx = X(i)
        parts.append(f'<rect x="{cx-bw-0.5:.1f}" y="{Y(nv.get(p,0)):.1f}" width="{bw:.1f}" '
                     f'height="{ly0-Y(nv.get(p,0)):.1f}" fill="#ff6b6b" opacity="0.85"/>')
        parts.append(f'<rect x="{cx+0.5:.1f}" y="{Y(fy.get(p,0)):.1f}" width="{bw:.1f}" '
                     f'height="{ly0-Y(fy.get(p,0)):.1f}" fill="#06d6a0" opacity="0.85"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">the {nfact} permutations (each a bar pair)</text>')
    parts.append(f'<rect x="{lx0+8}" y="{ly1}" width="9" height="9" fill="#06d6a0"/>'
                 f'<text x="{lx0+21}" y="{ly1+8}" fill="#e6edf3" font-size="9">Fisher-Yates (uniform)</text>')
    parts.append(f'<rect x="{lx0+8}" y="{ly1+14}" width="9" height="9" fill="#ff6b6b"/>'
                 f'<text x="{lx0+21}" y="{ly1+22}" fill="#e6edf3" font-size="9">naive (biased)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
