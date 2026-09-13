"""Demo: integer partitions -- counting the ways to sum to n, and Euler's identities.

Shows the explosive growth of p(n) via the pentagonal recurrence, lists the partitions of a small n
as Ferrers diagrams, and verifies two of Euler's classic bijective identities. Draws p(n) on a log
scale and a Ferrers diagram.

    python examples/partitions_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from partitions import (  # noqa: E402
    partition_count,
    partition_counts_up_to,
    generate_partitions,
    count_distinct_partitions,
    count_odd_partitions,
    conjugate,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Integer partitions: the ways to write n as a sum, and Euler's pentagonal magic\n")

    print("  Explosive growth of p(n) (Euler pentagonal recurrence, O(n^1.5)):")
    for n in [1, 5, 10, 20, 50, 100, 200]:
        print(f"    p({n:>3}) = {partition_count(n):,}")

    print("\n  The 7 partitions of 5, as Ferrers diagrams:")
    for p in generate_partitions(5):
        rows = "  ".join("*" * part for part in p)
        print(f"    {'+'.join(map(str, p)):<12} {rows}")

    print("\n  Euler's theorem -- #(distinct parts) == #(odd parts):")
    print(f"    {'n':>3}  {'distinct':>9}  {'odd':>5}")
    for n in [5, 10, 15, 20]:
        print(f"    {n:>3}  {count_distinct_partitions(n):>9}  {count_odd_partitions(n):>5}")

    # conjugate example
    p = (4, 2, 1)
    print(f"\n  Conjugate (Ferrers transpose) of {p} is {conjugate(p)}:")
    print(f"    largest part {p[0]} <-> number of parts {len(conjugate(p))}")

    _svg(os.path.join(outdir, "partitions.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'partitions.svg')}")


def _svg(path, width=760, height=430):
    counts = partition_counts_up_to(60)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="28" fill="#e6edf3" font-size="15">'
        'Left: p(n) grows sub-exponentially (log scale). Right: Ferrers diagram of 5+3+3+1</text>',
    ]

    # ---- left: log-scale plot of p(n) ---------------------------------------------------
    ox, oy, ow, oh = 55, 60, 340, 320
    ns = list(range(1, len(counts)))
    ly = [math.log10(counts[n]) for n in ns]
    lymax = max(ly)

    def px(n):
        return ox + ow * (n - ns[0]) / (ns[-1] - ns[0])

    def py(v):
        return oy + oh * (1 - v / lymax)

    parts.append(f'<line x1="{ox}" y1="{oy+oh}" x2="{ox+ow}" y2="{oy+oh}" stroke="#30363d"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+oh}" stroke="#30363d"/>')
    pts = " ".join(f"{px(n):.1f},{py(math.log10(counts[n])):.1f}" for n in ns)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    for n in (10, 20, 40, 60):
        if n < len(counts):
            parts.append(f'<circle cx="{px(n):.1f}" cy="{py(math.log10(counts[n])):.1f}" r="3" '
                         f'fill="#ffd43b"/>')
            parts.append(f'<text x="{px(n):.1f}" y="{py(math.log10(counts[n]))-8:.1f}" '
                         f'fill="#8b949e" font-size="8" text-anchor="middle">p({n})</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+20}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">n</text>')
    parts.append(f'<text x="{ox-8}" y="{oy-4}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">log10 p(n)</text>')

    # ---- right: Ferrers diagram of (5,3,3,1) -------------------------------------------
    diagram = (5, 3, 3, 1)
    dx, dy, cell = 440, 80, 26
    for i, row in enumerate(diagram):
        for j in range(row):
            parts.append(f'<rect x="{dx + j*cell}" y="{dy + i*cell}" width="{cell-4}" '
                         f'height="{cell-4}" rx="3" fill="#06d6a0"/>')
    parts.append(f'<text x="{dx}" y="{dy + len(diagram)*cell + 22}" fill="#8b949e" font-size="10">'
                 f'rows = parts (5,3,3,1); columns = conjugate {conjugate(diagram)}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
