"""Demo: Stirling and Bell numbers -- counting set partitions and permutation cycles.

Prints the Stirling triangles of both kinds, lists set partitions of a small set, shows Bell numbers
via three routes (sum of Stirling, Bell triangle, Dobinski), and draws the second-kind triangle as a
heatmap with Bell numbers along the row sums.

    python examples/stirling_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stirling import (  # noqa: E402
    stirling_second,
    stirling_first,
    bell,
    bell_triangle,
    bell_dobinski,
    stirling_second_row,
    brute_set_partitions,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Stirling and Bell numbers: the arithmetic of set partitions and cycles\n")

    print("  Stirling numbers of the SECOND kind S(n,k) -- set into k blocks:")
    print("      n\\k " + "".join(f"{k:>6}" for k in range(8)))
    for n in range(8):
        row = "".join(f"{stirling_second(n, k):>6}" for k in range(8))
        print(f"      {n:>2}  {row}   sum = B({n}) = {bell(n)}")

    print("\n  Stirling numbers of the FIRST kind c(n,k) -- permutations with k cycles:")
    print("      n\\k " + "".join(f"{k:>6}" for k in range(7)))
    for n in range(7):
        row = "".join(f"{stirling_first(n, k):>6}" for k in range(7))
        import math
        print(f"      {n:>2}  {row}   sum = {n}! = {math.factorial(n)}")

    print("\n  The 5 set partitions of {1,2,3} (B(3) = 5):")
    for p in brute_set_partitions(3):
        blocks = " | ".join("".join(str(x + 1) for x in sorted(b)) for b in p)
        print(f"    {{{blocks}}}")

    print("\n  Bell numbers three ways:")
    print(f"    {'n':>3}  {'sum of S(n,k)':>14}  {'Bell triangle':>14}  {'Dobinski':>10}")
    tri = bell_triangle(9)
    for n in range(9):
        print(f"    {n:>3}  {sum(stirling_second_row(n)):>14}  {tri[n][0]:>14}  "
              f"{bell_dobinski(n):>10.2f}")

    _svg(os.path.join(outdir, "stirling.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'stirling.svg')}")


def _svg(path, width=760, height=430, N=9):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="28" fill="#e6edf3" font-size="15">'
        'Stirling triangle S(n,k) (2nd kind); each row sums to the Bell number B(n)</text>',
    ]
    cell = 44
    x0, y0 = 60, 60
    vmax = max(stirling_second(n, k) for n in range(N) for k in range(N))
    import math
    logmax = math.log10(vmax + 1)
    for n in range(N):
        for k in range(n + 1):
            v = stirling_second(n, k)
            t = math.log10(v + 1) / logmax if logmax > 0 else 0
            r = int(0x16 + t * (0xff - 0x16))
            g = int(0x1b + t * (0xd4 - 0x1b))
            b = int(0x22 + t * (0x3b - 0x22))
            x = x0 + k * cell
            y = y0 + n * cell * 0.8
            parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cell-4}" height="{cell*0.8-4:.0f}" '
                         f'rx="3" fill="rgb({r},{g},{b})"/>')
            parts.append(f'<text x="{x + (cell-4)/2:.0f}" y="{y + cell*0.4:.0f}" '
                         f'fill="{"#0d1117" if t > 0.5 else "#8b949e"}" font-size="9" '
                         f'text-anchor="middle">{v}</text>')
        # Bell number as row sum
        bx = x0 + (n + 1.3) * cell
        by = y0 + n * cell * 0.8 + cell * 0.4
        parts.append(f'<text x="{bx:.0f}" y="{by:.0f}" fill="#06d6a0" font-size="10">'
                     f'B({n})={bell(n)}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
