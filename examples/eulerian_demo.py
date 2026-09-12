"""Demo: Eulerian paths and circuits -- traversing every edge once, from Konigsberg to Hierholzer.

Classifies several graphs (including the famous Seven Bridges of Konigsberg), builds an Eulerian trail
where one exists, and draws a graph with its trail's edges numbered in walk order.

    python examples/eulerian_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eulerian import (undirected_euler_status, undirected_eulerian_trail,  # noqa: E402
                      is_valid_undirected_trail)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Eulerian paths and circuits: every edge exactly once\n")

    cases = [
        ("triangle", 3, [(0, 1), (1, 2), (2, 0)]),
        ("path 0-1-2-3", 4, [(0, 1), (1, 2), (2, 3)]),
        ("Konigsberg bridges", 4, [(0, 1), (0, 1), (0, 2), (0, 2), (0, 3), (1, 3), (2, 3)]),
        ("K4 (complete, 4 nodes)", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        ("bowtie (two triangles)", 5, [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]),
    ]

    for name, n, edges in cases:
        status = undirected_euler_status(n, edges)
        label = {"circuit": "Eulerian CIRCUIT (closed)",
                 "path": "Eulerian PATH (open)",
                 "none": "no Eulerian trail"}[status]
        trail = undirected_eulerian_trail(n, edges)
        extra = ""
        if trail is not None:
            extra = "  trail: " + "->".join(map(str, trail))
        print(f"  {name:26s} {label}{extra}")

    print("\n  The Seven Bridges of Konigsberg has all four land masses at odd degree, so no walk can")
    print("  cross every bridge exactly once -- Euler's 1736 proof, the birth of graph theory.")
    print("\n  Hierholzer builds a trail by walking until stuck (a closed sub-tour), then splicing in")
    print("  detours from any vertex with unused edges, until every edge is consumed -- linear time.")

    # draw the bowtie with its Eulerian circuit numbered
    n, edges = 5, [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
    trail = undirected_eulerian_trail(n, edges)
    print(f"\n  bowtie trail valid + closed: "
          f"{is_valid_undirected_trail(edges, trail) and trail[0] == trail[-1]}")

    _svg(os.path.join(outdir, "eulerian.svg"), n, edges, trail)
    print(f"\n  wrote {os.path.join(outdir, 'eulerian.svg')}")


def _svg(path, n, edges, trail, width=720, height=420):
    pos = {
        0: (360, 210),
        1: (200, 100), 2: (200, 320),
        3: (520, 100), 4: (520, 320),
    }
    # order in which edges are traversed
    step_of = {}
    for i, (a, b) in enumerate(zip(trail, trail[1:])):
        key = (min(a, b), max(a, b))
        step_of.setdefault(key, []).append(i + 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'An Eulerian circuit: each edge numbered in the order the trail walks it</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'a closed walk crossing all {len(edges)} edges exactly once, returning to the start</text>',
    ]

    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#06d6a0" '
                     f'stroke-width="2.5"/>')
        key = (min(u, v), max(u, v))
        steps = step_of.get(key, [])
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="11" fill="#0d1117" '
                     f'stroke="#ffd43b" stroke-width="1"/>')
        parts.append(f'<text x="{mx:.0f}" y="{my+4:.0f}" fill="#ffd43b" font-size="11" '
                     f'text-anchor="middle">{",".join(map(str, steps))}</text>')

    for v, (x, y) in pos.items():
        parts.append(f'<circle cx="{x}" cy="{y}" r="20" fill="#4dabf7"/>')
        parts.append(f'<text x="{x}" y="{y+5}" fill="#0d1117" font-size="15" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')

    parts.append(f'<text x="20" y="{height-20}" fill="#8b949e" font-size="12">'
                 f'trail: {"->".join(map(str, trail))}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
