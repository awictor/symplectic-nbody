"""Demo: sum over subsets (SOS DP) -- zeta/Moebius transforms and subset convolutions.

Shows the subset-sum transform on a small function, checks the Moebius inverse round-trips, and uses
subset-sum convolution to count ways to partition a set into two disjoint labelled parts. Draws the
zeta sweep bit by bit over the boolean lattice.

    python examples/sos_dp_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sos_dp import (  # noqa: E402
    zeta_subset,
    moebius_subset,
    or_convolution,
    subset_sum_convolution,
)


def _mask_str(m, n):
    return "{" + ",".join(str(i) for i in range(n) if m & (1 << i)) + "}"


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Sum over subsets (SOS DP): cumulative sums on the boolean hypercube in O(n 2^n)\n")

    n = 3
    size = 1 << n
    # f(S) = number of elements in S, just to have distinct values
    f = [bin(S).count("1") for S in range(size)]
    F = zeta_subset(f)

    print(f"  Universe of {n} elements, f(S) = |S|:\n")
    print(f"  {'S':>8}  {'f(S)':>4}   {'F(S)=sum_{T<=S} f(T)':>22}")
    for S in range(size):
        print(f"  {_mask_str(S, n):>8}  {f[S]:>4}   {F[S]:>22}")

    back = moebius_subset(F)
    print(f"\n  Moebius inverse recovers f exactly: {back == f}")

    # OR-convolution counting: f = g = indicator of singletons; OR-conv counts ordered pairs of
    # singletons whose union is S.
    singles = [1 if bin(S).count("1") == 1 else 0 for S in range(size)]
    orc = or_convolution(singles, singles)
    print("\n  OR-convolution of the singleton indicator with itself (ordered pairs of singletons")
    print("  whose UNION is S):")
    for S in range(size):
        if orc[S]:
            print(f"    {_mask_str(S, n)}: {orc[S]}")

    # subset-sum convolution: same but DISJOINT union (each pair uses distinct elements).
    ssc = subset_sum_convolution(singles, singles)
    print("\n  Subset-sum convolution (DISJOINT pairs of singletons; union is S, no shared element):")
    for S in range(size):
        if ssc[S]:
            print(f"    {_mask_str(S, n)}: {ssc[S]}")
    print("\n  Note the difference: OR-conv counts {0}|{0}={0}; subset-sum conv forbids reusing 0,")
    print("  so only 2-element sets get a count (2, the two orderings). Disjointness for free via")
    print("  the ranked-by-popcount zeta trick.")

    _svg(os.path.join(outdir, "sos_dp.svg"), n, f, F)
    print(f"\n  wrote {os.path.join(outdir, 'sos_dp.svg')}")


def _svg(path, n, f, F, width=760, height=430):
    size = 1 << n
    # arrange masks in a lattice by popcount (Hasse diagram)
    levels = {}
    for m in range(size):
        levels.setdefault(bin(m).count("1"), []).append(m)
    maxlvl = n
    pos = {}
    for lvl in range(maxlvl + 1):
        row = levels[lvl]
        for j, m in enumerate(row):
            x = width * (j + 1) / (len(row) + 1)
            y = 70 + lvl * (height - 120) / maxlvl
            pos[m] = (x, y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="30" fill="#e6edf3" font-size="15">'
        'Boolean lattice: node shows F(S) = sum of f over all subsets of S</text>',
    ]
    # cover edges (differ by one bit, subset below)
    for m in range(size):
        for i in range(n):
            if not (m & (1 << i)):
                up = m | (1 << i)
                x1, y1 = pos[m]
                x2, y2 = pos[up]
                parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                             f'stroke="#30363d" stroke-width="1"/>')
    fmax = max(F) or 1
    for m in range(size):
        x, y = pos[m]
        t = F[m] / fmax
        # colour ramp blue->green by F value
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="19" fill="#161b22" '
                     f'stroke="#4dabf7" stroke-width="{1 + 3 * t:.1f}"/>')
        label = "{" + ",".join(str(i) for i in range(n) if m & (1 << i)) + "}"
        parts.append(f'<text x="{x:.0f}" y="{y-1:.0f}" fill="#8b949e" font-size="7" '
                     f'text-anchor="middle">{label}</text>')
        parts.append(f'<text x="{x:.0f}" y="{y+9:.0f}" fill="#06d6a0" font-size="10" '
                     f'text-anchor="middle">{F[m]}</text>')
    parts.append('<text x="20" y="%d" fill="#8b949e" font-size="10">'
                 'bottom = empty set, top = full universe; ring weight scales with F(S)</text>'
                 % (height - 12))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
