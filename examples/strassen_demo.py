"""Demo: Strassen's algorithm -- sub-cubic matrix multiplication.

Multiplies matrices via Strassen, verifies against the schoolbook product, and shows how the seven
block products (instead of eight) drop the complexity exponent from 3 to log2(7). Draws the
scalar-multiplication count vs matrix size for schoolbook and Strassen.

    python examples/strassen_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import strassen  # noqa: E402
from strassen import multiply, schoolbook, equal  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Strassen's algorithm: multiplying matrices with 7 sub-products, not 8\n")

    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    print("  a 2x2 example:")
    print(f"    A = {a}")
    print(f"    B = {b}")
    print(f"    A*B = {multiply(a, b)}  (schoolbook: {schoolbook(a, b)})\n")

    # verify on a larger random matrix (force recursion)
    orig = strassen._CUTOFF
    strassen._CUTOFF = 2
    import random
    random.seed(7)
    n = 16
    A = [[random.randint(-5, 5) for _ in range(n)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(n)] for _ in range(n)]
    match = equal(multiply(A, B), schoolbook(A, B))
    strassen._CUTOFF = orig
    print(f"  {n}x{n} random matrices: Strassen matches schoolbook -> {match}\n")

    print("  The seven products (block form, splitting A and B into 2x2 blocks):")
    lines = [
        "M1 = (A11+A22)(B11+B22)", "M2 = (A21+A22)B11", "M3 = A11(B12-B22)",
        "M4 = A22(B21-B11)", "M5 = (A11+A12)B22", "M6 = (A21-A11)(B11+B12)",
        "M7 = (A12-A22)(B21+B22)",
    ]
    for ln in lines:
        print(f"    {ln}")
    print("    C11=M1+M4-M5+M7  C12=M3+M5  C21=M2+M4  C22=M1-M2+M3+M6")

    print("\n  scalar-multiplication count (leading term) at doubling sizes:")
    print(f"    {'n':>5} {'schoolbook n^3':>16} {'Strassen n^2.807':>18}")
    for n in (64, 128, 256, 512, 1024):
        print(f"    {n:>5} {n**3:>16,} {int(n ** math.log2(7)):>18,}")

    print("\n  Trading one recursive multiplication for a handful of matrix additions cuts the")
    print("  exponent from 3 to log2(7) ~ 2.807 -- the first sub-cubic matrix multiply, and the")
    print("  starting gun for the decades-long race to lower the exponent further.")

    _svg(os.path.join(outdir, "strassen.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'strassen.svg')}")


def _svg(path, width=740, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'Scalar multiplications to multiply two n x n matrices</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'schoolbook n^3 (red) vs Strassen n^2.807 (green); log-log, the gap widens with n</text>',
    ]

    ox, oy = 70, 330
    pw, ph = width - 130, 250
    exp_s = math.log2(7)
    ns = [2 ** (i / 2) for i in range(4, 25)]      # n from 4 to ~4096

    def px(n):
        return ox + (math.log10(n) - math.log10(ns[0])) / (math.log10(ns[-1]) - math.log10(ns[0])) * pw

    costs = [3 * math.log10(n) for n in ns] + [exp_s * math.log10(n) for n in ns]
    cmin, cmax = min(costs), max(costs)

    def py(clog):
        return oy - (clog - cmin) / (cmax - cmin) * ph

    for exp, col, label in [(3.0, "#ff6b6b", "schoolbook n^3"),
                            (exp_s, "#06d6a0", "Strassen n^2.807")]:
        pts = " ".join(f"{px(n):.1f},{py(exp * math.log10(n)):.1f}" for n in ns)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<text x="{px(ns[-1])-2:.0f}" y="{py(exp * math.log10(ns[-1]))-6:.0f}" '
                     f'fill="{col}" font-size="11" text-anchor="end">{label}</text>')

    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d"/>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+30}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">matrix size n (log scale) -></text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
