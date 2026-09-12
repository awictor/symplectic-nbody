"""Demo: degree sequences -- which wish-lists of connections can a real network satisfy?

Tests several degree sequences for realizability, shows the Havel-Hakimi reduction peeling one down,
builds a witness graph, and draws it with each vertex labelled by its achieved degree.

    python examples/degree_sequence_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from degree_sequence import (is_graphic_erdos_gallai, is_graphic_havel_hakimi,  # noqa: E402
                             realize, reduction_steps, degree_sequence, brute_is_graphic)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Degree sequences: is a list of wanted connection-counts realizable?\n")

    tests = [
        ("triangle", [2, 2, 2]),
        ("star K(1,4)", [4, 1, 1, 1, 1]),
        ("K4", [3, 3, 3, 3]),
        ("impossible (3,3,3,1)", [3, 3, 3, 1]),
        ("odd sum (1,1,1)", [1, 1, 1]),
        ("mixed (3,2,2,2,1)", [3, 2, 2, 2, 1]),
    ]
    print(f"  {'sequence':22s} {'graphic?':10s} check")
    for name, seq in tests:
        eg = is_graphic_erdos_gallai(seq)
        agree = (eg == is_graphic_havel_hakimi(seq) == brute_is_graphic(seq))
        print(f"  {name:22s} {'yes' if eg else 'no':10s} "
              f"(EG=HH=brute agree: {agree})")

    print("\n  Havel-Hakimi reduction of (3,3,2,2) -- connect the top vertex, subtract, repeat:")
    steps, ok = reduction_steps([3, 3, 2, 2])
    for s in steps:
        print(f"    {s}")
    print(f"    -> {'graphic' if ok else 'not graphic'}")

    seq = [3, 2, 2, 2, 1]
    g = realize(seq)
    print(f"\n  witness graph for {seq}:")
    print(f"    edges: {g}")
    print(f"    achieved degree sequence: {degree_sequence(len(seq), g)}")

    print("\n  Erdos-Gallai tests the inequality sum(first k) <= k(k-1) + sum min(d_i,k) for every k;")
    print("  Havel-Hakimi instead constructs a graph greedily. Both always agree, and match an")
    print("  exhaustive search over all simple graphs -- the theory is exact, not heuristic.")

    _svg(os.path.join(outdir, "degree_sequence.svg"), len(seq), g, seq)
    print(f"\n  wrote {os.path.join(outdir, 'degree_sequence.svg')}")


def _svg(path, n, edges, seq, width=680, height=430):
    cx, cy, r = width / 2, height / 2 + 15, 140
    pos = {}
    for v in range(n):
        ang = 2 * math.pi * v / n - math.pi / 2
        pos[v] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'A graph realizing the degree sequence {sorted(seq, reverse=True)}</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'Havel-Hakimi built this simple graph; each node label is the degree it achieves</text>',
    ]

    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#4dabf7" stroke-width="2"/>')
    for v, (x, y) in pos.items():
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="#06d6a0"/>')
        parts.append(f'<text x="{x:.1f}" y="{y-2:.1f}" fill="#0d1117" font-size="12" '
                     f'text-anchor="middle" font-weight="bold">v{v}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y+13:.1f}" fill="#0d1117" font-size="10" '
                     f'text-anchor="middle">deg {deg[v]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
