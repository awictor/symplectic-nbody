"""Demo: the matrix permanent -- counting perfect matchings, and Ryser vs the naive sum.

Computes the permanent by the naive n! sum and by Ryser's O(2^n n) formula (showing the speedup),
counts the perfect matchings of a bipartite graph via its permanent, and draws the exponential-vs-
factorial operation counts.

    python examples/permanent_demo.py [output_dir]
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from permanent import (permanent_naive, permanent_ryser, count_perfect_matchings,  # noqa: E402
                       brute_count_matchings)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Matrix permanent: the determinant's #P-complete twin -- it counts matchings\n")

    # a bipartite graph: which workers can do which jobs
    workers = ["Ann", "Bob", "Cy", "Dee"]
    jobs = ["web", "db", "ml", "ops"]
    B = [
        [1, 1, 0, 0],   # Ann: web, db
        [1, 1, 1, 0],   # Bob: web, db, ml
        [0, 1, 1, 1],   # Cy: db, ml, ops
        [0, 0, 1, 1],   # Dee: ml, ops
    ]
    n = len(B)
    print("  bipartite qualification matrix (worker x job):")
    print("           " + "  ".join(f"{j:>3}" for j in jobs))
    for i, w in enumerate(workers):
        print(f"    {w:>4}  " + "    ".join(str(B[i][j]) for j in range(n)))
    pm = count_perfect_matchings(B)
    print(f"\n  perfect matchings (everyone assigned a distinct qualified job): {pm}")
    print(f"    (brute-force enumeration agrees: {brute_count_matchings(B)})\n")

    # naive vs Ryser
    print("  operation count: naive n! sum vs Ryser's O(2^n n):")
    for n in (5, 8, 10, 12, 15, 20):
        naive_ops = math.factorial(n) * n
        ryser_ops = (2 ** n) * n
        print(f"    n={n:2d}: naive ~{naive_ops:>18,}   Ryser ~{ryser_ops:>12,}   "
              f"({naive_ops/ryser_ops:,.0f}x fewer)")
    print()

    # timing check on a moderate matrix
    class LCG:
        def __init__(s, seed):
            s.s = seed & 0xFFFFFFFF

        def r(s, n):
            s.s = (1664525 * s.s + 1013904223) & 0xFFFFFFFF
            return (s.s >> 8) % n
    rng = LCG(7)
    M = [[rng.r(4) for _ in range(9)] for _ in range(9)]
    t0 = time.time()
    pn = permanent_naive(M)
    tn = time.time() - t0
    t0 = time.time()
    pr = permanent_ryser(M)
    tr = time.time() - t0
    print(f"  9x9 random matrix: naive {tn*1000:.1f} ms, Ryser {tr*1000:.2f} ms "
          f"({tn/max(tr,1e-9):.0f}x faster), same answer {pn == pr}\n")

    print("  perm(A) = sum over permutations of prod A[i][sigma(i)] -- the determinant without signs.")
    print("  That missing sign makes it #P-complete (harder than NP), but Ryser's inclusion-exclusion")
    print("  over the 2^n column subsets, iterated in Gray-code order, computes it in O(2^n n) -- the")
    print("  method behind bipartite matching counts and boson-sampling amplitudes.")

    _svg(os.path.join(outdir, "permanent.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'permanent.svg')}")


def _svg(path, width=760, height=400):
    ox, oy = 60, 340
    pw, ph = width - 100, 280
    ns = list(range(2, 21))
    naive = [math.lgamma(n + 1) / math.log(10) + math.log10(n) for n in ns]   # log10(n! * n)
    ryser = [n * math.log10(2) + math.log10(n) for n in ns]                    # log10(2^n * n)
    ymax = max(naive)

    def px(n):
        return ox + (n - ns[0]) / (ns[-1] - ns[0]) * pw

    def py(v):
        return oy - v / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Operations to compute the permanent (log scale): naive n! vs Ryser 2^n</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'both explode (it is #P-complete), but Ryser is astronomically smaller</text>',
    ]
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d" stroke-width="1"/>')
    for exp in (0, 5, 10, 15):
        y = py(exp)
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+pw}" y2="{y:.1f}" stroke="#21262d" '
                     f'stroke-width="0.6"/>')
        parts.append(f'<text x="{ox-6}" y="{y+4:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">1e{exp}</text>')
    pn = " ".join(f"{px(n):.1f},{py(v):.1f}" for n, v in zip(ns, naive))
    parts.append(f'<polyline points="{pn}" fill="none" stroke="#ff6b6b" stroke-width="2"/>')
    pr = " ".join(f"{px(n):.1f},{py(v):.1f}" for n, v in zip(ns, ryser))
    parts.append(f'<polyline points="{pr}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    parts.append(f'<text x="{px(14):.0f}" y="{py(naive[12])-6:.0f}" fill="#ff6b6b" font-size="11">naive n!</text>')
    parts.append(f'<text x="{px(14):.0f}" y="{py(ryser[12])+16:.0f}" fill="#06d6a0" font-size="11">Ryser 2^n</text>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+28}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">matrix size n</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
