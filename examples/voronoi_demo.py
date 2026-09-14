"""Voronoi demo: colored cells over a point cloud, and Lloyd relaxation toward a centroidal diagram (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import voronoi as V


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
SITE = "#ffd43b"
PALETTE = ["#4dabf7", "#06d6a0", "#b197fc", "#ff922b", "#ff6b6b", "#ffd43b",
           "#63e6be", "#faa2c1", "#74c0fc", "#ffc078", "#8ce99a", "#e599f7"]


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _panel(s, sites, box, ox, oy, w, h, title):
    xmin, ymin, xmax, ymax = box

    def sx(x):
        return ox + (x - xmin) / (xmax - xmin) * w

    def sy(y):
        return oy + h - (y - ymin) / (ymax - ymin) * h

    s.append(f'<text x="{ox+w/2:.0f}" y="{oy-8}" fill="{TEXT}" font-size="13" '
             f'text-anchor="middle">{title}</text>')
    s.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" fill="none" stroke="{GRAY}"/>')
    cells = V.cells(sites, box)
    for i, c in enumerate(cells):
        if len(c) < 3:
            continue
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in c)
        col = PALETTE[i % len(PALETTE)]
        s.append(f'<polygon points="{pts}" fill="{col}" fill-opacity="0.25" '
                 f'stroke="{col}" stroke-width="1.2"/>')
    for (x, y) in sites:
        s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{SITE}"/>')


def _svg(path, sites, box):
    W, H = 720, 400
    pw, ph = 300, 300
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    _panel(s, sites, box, 40, 50, pw, ph, "random sites")
    relaxed = sites
    for _ in range(12):
        relaxed = V.lloyd_step(relaxed, box)
    _panel(s, relaxed, box, 380, 50, pw, ph, "after 12 Lloyd steps (centroidal)")
    s.append(f'<text x="40" y="375" fill="{GRAY}" font-size="10">'
             f'Lloyd relaxation moves each site to its cell centroid -> uniform, honeycomb-like cells</text>')
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(s))


def main(outdir=None):
    rnd = _lcg(20260913)
    sites = [(rnd() * 10, rnd() * 10) for _ in range(12)]
    box = V.bounding_box(sites, 0.15)

    lines = []
    lines.append("Voronoi cells by half-plane intersection")
    lines.append("=" * 58)
    lines.append(f"{len(sites)} random sites in a {box[2]-box[0]:.1f} x {box[3]-box[1]:.1f} box")
    lines.append("")
    cells = V.cells(sites, box)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    lines.append(f"{'site':>4}{'x':>8}{'y':>8}{'verts':>7}{'cell area':>12}")
    for i, (x, y) in enumerate(sites):
        lines.append(f"{i:>4}{x:>8.2f}{y:>8.2f}{len(cells[i]):>7}{V.polygon_area(cells[i]):>12.3f}")
    tot = sum(V.polygon_area(c) for c in cells)
    lines.append("-" * 39)
    lines.append(f"{'sum':>27}{tot:>12.3f}")
    lines.append(f"{'box':>27}{box_area:>12.3f}  (cells tile the box exactly)")
    lines.append("")

    # Lloyd relaxation: area variance drops
    def area_var(ss):
        ar = [V.polygon_area(c) for c in V.cells(ss, box)]
        m = sum(ar) / len(ar)
        return sum((a - m) ** 2 for a in ar) / len(ar)

    lines.append("Lloyd relaxation (move each site to its cell centroid):")
    lines.append(f"{'step':>6}{'cell-area variance':>22}")
    s = sites
    cur = 0
    for cp in (0, 2, 4, 8, 12):
        while cur < cp:
            s = V.lloyd_step(s, box)
            cur += 1
        lines.append(f"{cp:>6}{area_var(s):>22.4f}")
    lines.append("Variance falls: relaxation equalizes cell sizes (centroidal Voronoi tessellation).")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        _svg(os.path.join(outdir, "voronoi_cells.svg"), sites, box)

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
