"""Demo: orthogonal matching pursuit -- recovering a sparse signal from few measurements.

Plants a sparse signal, measures it with far fewer random projections than its length, and recovers
the exact support and coefficients with OMP -- the compressed-sensing miracle. Shows how few
measurements suffice and draws the true vs recovered signal.

    python examples/omp_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omp import omp, random_matrix, matvec, residual_norm, support  # noqa: E402


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Orthogonal matching pursuit: sparse recovery from compressed measurements\n")

    rng = _lcg(2024)
    n, k = 60, 4
    x_true = [0.0] * n
    spikes = {8: 2.5, 22: -1.8, 41: 3.0, 55: -2.2}
    for j, v in spikes.items():
        x_true[j] = v

    print(f"  Signal length n = {n}, sparsity k = {k} (only {k} nonzero coefficients).\n")
    print(f"  {'measurements m':>15}  {'recovered support?':>19}  {'max coeff error':>16}")
    for m in [8, 15, 20, 25]:
        A = random_matrix(m, n, seed=1)
        y = matvec(A, x_true)
        x_rec = omp(A, y, sparsity=k)
        ok = support(x_rec) == support(x_true)
        err = max(abs(x_rec[i] - x_true[i]) for i in range(n))
        print(f"  {m:>15}  {'YES' if ok else 'no':>19}  {err:>16.2e}")

    print(f"\n  Around m ~ 20 random measurements (a few times k) OMP nails all {n} coefficients")
    print("  exactly, though the system is wildly underdetermined -- the sparsity is what makes it")
    print("  solvable. This is the principle behind MRI acceleration and single-pixel cameras.")

    m = 20
    A = random_matrix(m, n, seed=1)
    y = matvec(A, x_true)
    x_rec = omp(A, y, sparsity=k)
    _svg(os.path.join(outdir, "omp.svg"), x_true, x_rec, m, n)
    print(f"\n  wrote {os.path.join(outdir, 'omp.svg')}")


def _svg(path, x_true, x_rec, m, n, width=760, height=360):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Sparse signal (n={n}) recovered from just m={m} measurements by OMP</text>',
    ]
    ox, oy, ow, oh = 40, 60, width - 80, height - 110
    amax = max(max(abs(v) for v in x_true), 1e-9)

    def px(i):
        return ox + ow * i / (n - 1)

    def py(v):
        return oy + oh / 2 * (1 - v / amax)

    # zero line
    parts.append(f'<line x1="{ox}" y1="{py(0):.1f}" x2="{ox+ow}" y2="{py(0):.1f}" '
                 f'stroke="#30363d" stroke-width="1"/>')
    # true stems (gray) and recovered (green dots)
    for i in range(n):
        if abs(x_true[i]) > 1e-9:
            parts.append(f'<line x1="{px(i):.1f}" y1="{py(0):.1f}" x2="{px(i):.1f}" '
                         f'y2="{py(x_true[i]):.1f}" stroke="#8b949e" stroke-width="2"/>')
        if abs(x_rec[i]) > 1e-9:
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(x_rec[i]):.1f}" r="4" '
                         f'fill="none" stroke="#06d6a0" stroke-width="2"/>')
    parts.append(f'<text x="{ox}" y="{oy+oh+18}" fill="#8b949e" font-size="10">'
                 f'gray stems = true signal, green rings = OMP recovery (they coincide exactly)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
