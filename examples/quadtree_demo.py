"""Demo: a quadtree indexing a point cloud for fast region queries.

Builds a quadtree over random points, shows how it subdivides denser regions more deeply, answers a
rectangle and a circle range query (confirmed against a brute-force scan), and finds a nearest
neighbour. Draws the point cloud with the quadtree's cell boundaries and a highlighted query region.

    python examples/quadtree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quadtree import QuadTree, brute_rect, brute_circle, brute_nearest  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 20260911

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    # a point cloud with a dense cluster in one corner
    pts = [(rng() * 100, rng() * 100) for _ in range(120)]
    pts += [(15 + rng() * 15, 75 + rng() * 15) for _ in range(60)]   # dense NW cluster

    qt = QuadTree(bounds=(0, 0, 100, 100), capacity=4)
    for p in pts:
        qt.insert(p)

    print("Quadtree: recursive 2-D spatial partition for region queries\n")
    print(f"  {qt.count()} points over a 100x100 area, node capacity 4")
    print(f"  tree depth {qt.depth()} (denser regions subdivide deeper)\n")

    # rectangle query
    rq = qt.query_rect_bounds(30, 20, 70, 60)
    print(f"  Rectangle query [30,20]-[70,60]: {len(rq)} points "
          f"(matches brute force: {sorted(rq) == sorted(brute_rect(pts, 30, 20, 70, 60))})")

    # circle query
    cq = qt.query_circle(22, 82, 12)
    print(f"  Circle query centre (22,82) r=12 (in the dense cluster): {len(cq)} points "
          f"(matches brute force: {sorted(cq) == sorted(brute_circle(pts, 22, 82, 12))})")

    # nearest
    nq = qt.nearest(50, 50)
    print(f"  Nearest point to (50,50): ({nq[0]:.1f}, {nq[1]:.1f}) "
          f"(matches brute force: {nq == brute_nearest(pts, 50, 50)})\n")

    # subdivision depth reflects density
    print("  A quadtree splits a square into four quadrants whenever a node exceeds its capacity,")
    print("  so empty regions stay shallow and crowded ones subdivide deeply. A range query visits")
    print("  only the cells whose square overlaps the query, pruning whole branches -- the basis of")
    print("  collision broad-phase and the Barnes-Hut n-body approximation.")

    _svg(os.path.join(outdir, "quadtree.svg"), qt, pts, (30, 20, 70, 60))
    print(f"\n  wrote {os.path.join(outdir, 'quadtree.svg')}")


def _svg(path, qt, pts, rect_bounds, width=760, height=430):
    size = 380
    ox, oy = 45, 40

    def X(x):
        return ox + x / 100 * size

    def Y(y):
        return oy + (100 - y) / 100 * size          # flip so +y is up

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Quadtree: cell boundaries (grey) adapt to point density; query rectangle (yellow)</text>',
    ]

    # draw the quadtree cell boundaries
    def draw_cells(node):
        b = node.boundary
        parts.append(f'<rect x="{X(b.cx - b.hw):.1f}" y="{Y(b.cy + b.hh):.1f}" '
                     f'width="{b.hw * 2 / 100 * size:.1f}" height="{b.hh * 2 / 100 * size:.1f}" '
                     f'fill="none" stroke="#30363d" stroke-width="0.7"/>')
        if node.divided:
            for c in (node.nw, node.ne, node.sw, node.se):
                draw_cells(c)

    draw_cells(qt)

    # the query rectangle
    x0, y0, x1, y1 = rect_bounds
    parts.append(f'<rect x="{X(x0):.1f}" y="{Y(y1):.1f}" width="{(x1 - x0) / 100 * size:.1f}" '
                 f'height="{(y1 - y0) / 100 * size:.1f}" fill="#ffd43b" fill-opacity="0.12" '
                 f'stroke="#ffd43b" stroke-width="1.5" stroke-dasharray="4 3"/>')

    # points, those inside the query rect highlighted
    inrect = set(map(tuple, brute_rect(pts, x0, y0, x1, y1)))
    for p in pts:
        hot = tuple(p) in inrect
        parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="{2.6 if hot else 1.8:.1f}" '
                     f'fill="{"#ffd43b" if hot else "#4dabf7"}" '
                     f'{"" if hot else "opacity=0.7"}/>')

    parts.append(f'<text x="{ox}" y="{oy + size + 22:.1f}" fill="#8b949e" font-size="11">'
                 f'{qt.count()} points, depth {qt.depth()}; yellow = inside the query rectangle</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
