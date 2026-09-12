"""Demo: maximum bipartite matching -- assigning workers to jobs they're qualified for.

Builds a workers-to-jobs qualification graph, finds the maximum matching (most workers placed) with
Hopcroft-Karp, recovers the minimum set of people/jobs that touches every qualification (Konig), and
draws the bipartite graph with matched edges highlighted.

    python examples/bipartite_matching_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bipartite_matching import (BipartiteGraph, hopcroft_karp,  # noqa: E402
                                minimum_vertex_cover, has_perfect_left_matching)

WORKERS = ["Ada", "Ben", "Cam", "Dee", "Eli"]
JOBS = ["frontend", "backend", "data", "ops", "design"]

# who is qualified for what (left worker -> right job)
QUALIFIED = [
    (0, 0), (0, 1),           # Ada: frontend, backend
    (1, 1), (1, 2),           # Ben: backend, data
    (2, 2), (2, 3),           # Cam: data, ops
    (3, 0), (3, 4),           # Dee: frontend, design
    (4, 4),                   # Eli: design only
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    g = BipartiteGraph(len(WORKERS), len(JOBS))
    for u, v in QUALIFIED:
        g.add_edge(u, v)

    size, match_l, match_r = hopcroft_karp(g)
    left_cover, right_cover = minimum_vertex_cover(g)

    print("Maximum bipartite matching: staffing 5 workers onto 5 jobs\n")
    print("  Qualifications:")
    for u in range(len(WORKERS)):
        jobs = ", ".join(JOBS[v] for v in g.adj[u])
        print(f"    {WORKERS[u]:4s} -> {jobs}")

    print(f"\n  Maximum placement: {size} of {len(WORKERS)} workers assigned")
    for u in range(len(WORKERS)):
        if match_l[u] != -1:
            print(f"    {WORKERS[u]:4s} -> {JOBS[match_l[u]]}")
        else:
            print(f"    {WORKERS[u]:4s} -> (unplaced)")

    print(f"\n  Perfect placement possible? {has_perfect_left_matching(g)}")
    cover_names = [WORKERS[u] for u in left_cover] + [JOBS[v] for v in right_cover]
    print(f"\n  Konig minimum cover ({len(cover_names)} = matching size): {cover_names}")
    print("  Every qualification edge touches at least one of these -- the smallest such set, and by")
    print("  Konig's theorem its size equals the maximum matching. Hopcroft-Karp finds the matching")
    print("  in O(E*sqrt(V)) by augmenting many shortest paths per BFS phase, not one at a time.")

    _svg(os.path.join(outdir, "bipartite_matching.svg"), g, match_l)
    print(f"\n  wrote {os.path.join(outdir, 'bipartite_matching.svg')}")


def _svg(path, g, match_l, width=680, height=440):
    lx, rx = 190, 490
    top = 90
    gap = 70
    matched = {(u, match_l[u]) for u in range(g.nl) if match_l[u] != -1}

    def ly(u):
        return top + u * gap

    def ry(v):
        return top + v * gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">'
        f'Maximum matching: green edges are the optimal worker-to-job assignment</text>',
        f'<text x="20" y="54" fill="#8b949e" font-size="12">'
        f'grey edges are qualifications; the solver picks a largest vertex-disjoint set of them</text>',
        f'<text x="{lx-30}" y="{top-24}" fill="#4dabf7" font-size="13" text-anchor="middle">'
        f'WORKERS</text>',
        f'<text x="{rx+40}" y="{top-24}" fill="#ffd43b" font-size="13" text-anchor="middle">'
        f'JOBS</text>',
    ]

    # edges
    for u in range(g.nl):
        for v in g.adj[u]:
            if (u, v) in matched:
                parts.append(f'<line x1="{lx}" y1="{ly(u)}" x2="{rx}" y2="{ry(v)}" '
                             f'stroke="#06d6a0" stroke-width="3.5"/>')
            else:
                parts.append(f'<line x1="{lx}" y1="{ly(u)}" x2="{rx}" y2="{ry(v)}" '
                             f'stroke="#8b949e" stroke-width="1.2" opacity="0.5"/>')

    # left nodes
    for u in range(g.nl):
        y = ly(u)
        parts.append(f'<circle cx="{lx}" cy="{y}" r="14" fill="#4dabf7"/>')
        parts.append(f'<text x="{lx-24}" y="{y+5}" fill="#e6edf3" font-size="13" '
                     f'text-anchor="end">{WORKERS[u]}</text>')
    # right nodes
    for v in range(g.nr):
        y = ry(v)
        parts.append(f'<circle cx="{rx}" cy="{y}" r="14" fill="#ffd43b"/>')
        parts.append(f'<text x="{rx+24}" y="{y+5}" fill="#e6edf3" font-size="13" '
                     f'text-anchor="start">{JOBS[v]}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
