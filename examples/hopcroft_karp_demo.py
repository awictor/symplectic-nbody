"""Demo: Hopcroft-Karp maximum bipartite matching + Koenig's minimum vertex cover.

Assigns applicants to jobs in a small hiring graph, prints the maximum matching and the dual
minimum vertex cover (equal in size, by Koenig's theorem), and draws the bipartite graph with the
matched edges highlighted.

    python examples/hopcroft_karp_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hopcroft_karp import HopcroftKarp  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    applicants = ["Ada", "Ben", "Cid", "Dot", "Eli"]
    jobs = ["Backend", "Frontend", "Data", "DevOps"]
    # who can do what (left index -> right index)
    edges = [
        (0, 0), (0, 2),           # Ada: Backend, Data
        (1, 0), (1, 1),           # Ben: Backend, Frontend
        (2, 1), (2, 3),           # Cid: Frontend, DevOps
        (3, 2), (3, 3),           # Dot: Data, DevOps
        (4, 1),                   # Eli: Frontend only
    ]
    nl, nr = len(applicants), len(jobs)

    print("Hopcroft-Karp: maximum bipartite matching in O(E sqrt(V))\n")
    print(f"  {nl} applicants, {nr} jobs, {len(edges)} qualifications.\n")
    print("  Who qualifies for what:")
    quals = {}
    for u, v in edges:
        quals.setdefault(u, []).append(jobs[v])
    for u in range(nl):
        print(f"    {applicants[u]:>4}: {', '.join(quals.get(u, ['(none)']))}")

    hk = HopcroftKarp(nl, nr, edges)
    size = hk.max_matching()
    print(f"\n  Maximum matching size: {size}  (jobs filled without double-booking)")
    for u, v in sorted(hk.pairs()):
        print(f"    {applicants[u]:>4} -> {jobs[v]}")
    unfilled = [applicants[u] for u in range(nl) if hk.match_l[u] == -1]
    if unfilled:
        print(f"    unassigned: {', '.join(unfilled)}")

    lc, rc = hk.min_vertex_cover()
    cover = [f"{applicants[u]} (applicant)" for u in sorted(lc)] + \
            [f"{jobs[v]} (job)" for v in sorted(rc)]
    print(f"\n  Koenig minimum vertex cover (size {len(lc) + len(rc)} == matching {size}):")
    print(f"    {', '.join(cover)}")
    print("\n  Koenig's theorem: in any bipartite graph the largest matching equals the smallest")
    print("  set of vertices touching every edge. The cover is the bottleneck -- pick these few")
    print("  people/roles and every qualification runs through one of them.")

    _svg(os.path.join(outdir, "hopcroft_karp.svg"), applicants, jobs, edges, hk)
    print(f"\n  wrote {os.path.join(outdir, 'hopcroft_karp.svg')}")


def _svg(path, applicants, jobs, edges, hk, width=760, height=430):
    nl, nr = len(applicants), len(jobs)
    lx, rx = 190, 570
    top = 70
    lgap = (height - top - 40) / max(1, nl - 1)
    rgap = (height - top - 40) / max(1, nr - 1)
    lpos = {u: (lx, top + u * lgap) for u in range(nl)}
    rpos = {v: (rx, top + v * rgap) for v in range(nr)}
    matched = set(hk.pairs())
    lc, rc = hk.min_vertex_cover()

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="30" fill="#e6edf3" font-size="15">'
        'Maximum matching (green) and minimum vertex cover (gold rings)</text>',
        f'<text x="{lx}" y="55" fill="#8b949e" font-size="11" text-anchor="middle">applicants</text>',
        f'<text x="{rx}" y="55" fill="#8b949e" font-size="11" text-anchor="middle">jobs</text>',
    ]

    for u, v in edges:
        x1, y1 = lpos[u]
        x2, y2 = rpos[v]
        on = (u, v) in matched
        col = "#06d6a0" if on else "#30363d"
        parts.append(f'<line x1="{x1}" y1="{y1:.0f}" x2="{x2}" y2="{y2:.0f}" stroke="{col}" '
                     f'stroke-width="{3 if on else 1}"/>')

    def node(cx, cy, label, in_cover, side_col):
        ring = '<circle cx="%d" cy="%.0f" r="21" fill="none" stroke="#ffd43b" stroke-width="2"/>' \
               % (cx, cy) if in_cover else ""
        return (
            ring +
            f'<circle cx="{cx}" cy="{cy:.0f}" r="15" fill="#161b22" stroke="{side_col}" '
            f'stroke-width="2"/>'
            f'<text x="{cx}" y="{cy+4:.0f}" fill="{side_col}" font-size="9" '
            f'text-anchor="middle">{label[:3]}</text>'
        )

    for u in range(nl):
        cx, cy = lpos[u]
        parts.append(node(cx, cy, applicants[u], u in lc, "#4dabf7"))
    for v in range(nr):
        cx, cy = rpos[v]
        parts.append(node(cx, cy, jobs[v], v in rc, "#b197fc"))

    parts.append(f'<text x="{width//2}" y="{height-15}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">|matching| = |cover| = {len(matched)} '
                 f'(Koenig duality)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
