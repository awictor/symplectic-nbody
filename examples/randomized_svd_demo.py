"""Demo: randomized SVD -- near-optimal low-rank factorization by random projection.

Compresses a low-rank-plus-noise matrix with randomized SVD, compares its singular values and
reconstruction error to the exact SVD, and shows power iteration sharpening the fit. Draws the
singular-value spectrum (exact vs randomized) and the error-vs-rank curve.

    python examples/randomized_svd_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from randomized_svd import randomized_svd, reconstruct, frobenius  # noqa: E402
from svd import svd as exact_svd  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _matmul(A, B):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Randomized SVD: top-k singular vectors by random projection (Halko-Martinsson-Tropp)\n")

    rng = _lcg(2024)
    m, n, r = 40, 30, 5
    # a rank-5 signal plus small noise
    L = [[rng() * 2 - 1 for _ in range(r)] for _ in range(m)]
    R = [[rng() * 2 - 1 for _ in range(n)] for _ in range(r)]
    signal = _matmul(L, R)
    A = [[signal[i][j] + (rng() - 0.5) * 0.1 for j in range(n)] for i in range(m)]

    print(f"  {m}x{n} matrix: rank-{r} signal + small noise.\n")

    Ue, Se, Vte = exact_svd(A)
    k = 5
    U, S, Vt = randomized_svd(A, k, oversample=5, n_power=2, seed=7)

    print(f"  {'i':>3}  {'exact sigma_i':>14}  {'randomized':>11}")
    for i in range(k):
        print(f"  {i:>3}  {Se[i]:>14.4f}  {S[i]:>11.4f}")

    err_r = frobenius(A, reconstruct(U, S, Vt))
    opt = math.sqrt(sum(Se[i] ** 2 for i in range(k, len(Se))))
    print(f"\n  Rank-{k} reconstruction error: randomized {err_r:.4f}, optimal (exact) {opt:.4f}")
    print(f"  Randomized is within {err_r/opt if opt>0 else 1:.3f}x of the best possible.\n")

    # power iteration effect
    print("  Effect of power iterations on top singular value accuracy:")
    for q in [0, 1, 2, 4]:
        _, Sq, _ = randomized_svd(A, k, oversample=2, n_power=q, seed=9)
        err = abs(Sq[0] - Se[0])
        print(f"    q={q}: |sigma_0 - exact| = {err:.2e}")

    print("\n  Random directions projected through A span its dominant range; a small exact SVD in")
    print("  that subspace recovers the top-k factors in O(m n k) instead of O(m n min(m,n)).")

    _svg(os.path.join(outdir, "randomized_svd.svg"), Se, S, A, Ue, Se, Vte)
    print(f"\n  wrote {os.path.join(outdir, 'randomized_svd.svg')}")


def _svg(path, Se, Sr, A, Ue, Sfull, Vte, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Left: singular-value spectrum (exact bars, randomized dots). Right: error vs rank</text>',
    ]

    # ---- left: spectrum -----------------------------------------------------------------
    ox, oy, ow, oh = 55, 60, 320, 300
    nshow = min(15, len(Sfull))
    smax = Sfull[0]
    bw = ow / nshow
    parts.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox+ow}" y2="{oy+oh}" stroke="#8b949e"/>')
    for i in range(nshow):
        h = oh * Sfull[i] / smax
        x = ox + i * bw
        parts.append(f'<rect x="{x:.1f}" y="{oy+oh-h:.1f}" width="{bw-3:.1f}" height="{h:.1f}" '
                     f'fill="#30363d"/>')
    for i in range(len(Sr)):
        x = ox + i * bw + (bw - 3) / 2
        h = oh * Sr[i] / smax
        parts.append(f'<circle cx="{x:.1f}" cy="{oy+oh-h:.1f}" r="4" fill="#ffd43b"/>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+20}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">singular value index</text>')
    parts.append(f'<text x="{ox}" y="{oy-6}" fill="#8b949e" font-size="9">'
                 f'gray = exact, yellow = randomized top-k</text>')

    # ---- right: reconstruction error vs rank --------------------------------------------
    from randomized_svd import randomized_svd, reconstruct, frobenius
    sx0, sy0, sw, sh = 430, 60, 300, 300
    ranks = list(range(1, 11))
    errs = []
    opts = []
    for k in ranks:
        U, S, Vt = randomized_svd(A, k, oversample=5, n_power=2, seed=7)
        errs.append(frobenius(A, reconstruct(U, S, Vt)))
        opts.append(math.sqrt(sum(Sfull[i] ** 2 for i in range(k, len(Sfull)))))
    emax = max(max(errs), max(opts)) or 1

    def px(i):
        return sx0 + sw * (ranks[i] - 1) / (len(ranks) - 1)

    def py(v):
        return sy0 + sh * (1 - v / emax)

    parts.append(f'<rect x="{sx0}" y="{sy0}" width="{sw}" height="{sh}" fill="none" stroke="#30363d"/>')
    pe = " ".join(f"{px(i):.1f},{py(errs[i]):.1f}" for i in range(len(ranks)))
    po = " ".join(f"{px(i):.1f},{py(opts[i]):.1f}" for i in range(len(ranks)))
    parts.append(f'<polyline points="{po}" fill="none" stroke="#8b949e" stroke-width="2" '
                 f'stroke-dasharray="4 3"/>')
    parts.append(f'<polyline points="{pe}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    parts.append(f'<text x="{sx0+sw/2:.0f}" y="{sy0+sh+20}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">rank k</text>')
    parts.append(f'<text x="{sx0}" y="{sy0-6}" fill="#8b949e" font-size="9">'
                 f'green = randomized error, gray dashed = optimal</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
