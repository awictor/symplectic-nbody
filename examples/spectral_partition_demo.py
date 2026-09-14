"""Spectral partition demo: cut a two-community graph by the Fiedler vector, colored by side (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import spectral_partition as SP


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
RED = "#ff6b6b"
CUT = "#ffd43b"
EDGE = "#30363d"


def _build_graph():
    """Two clusters of 6 vertices, dense inside, 2 bridge edges between."""
    edges = []
    # cluster A: vertices 0..5, cluster B: 6..11
    import itertools
    for base in (0, 6):
        verts = list(range(base, base + 6))
        for i, j in itertools.combinations(verts, 2):
            # dense but not complete: connect if index difference small (ring + chords)
            if (j - i) <= 2 or (j - i) == 5:
                edges.append((i, j))
    # bridges
    edges += [(2, 8), (5, 6)]
    return 12, edges


def _layout(n, edges, labels):
    """Place the two sides in two clusters for a readable drawing."""
    pos = {}
    a = [i for i in range(n) if labels[i] == 0]
    b = [i for i in range(n) if labels[i] == 1]
    for grp, cx in ((a, 200), (b, 520)):
        m = len(grp)
        for k, v in enumerate(grp):
            ang = 2 * math.pi * k / max(m, 1)
            pos[v] = (cx + 90 * math.cos(ang), 210 + 90 * math.sin(ang))
    return pos


def _svg(path, n, edges, labels, fvec):
    W, H = 720, 420
    pos = _layout(n, edges, labels)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    s.append(f'<text x="20" y="28" fill="{TEXT}" font-size="15">'
             f'Spectral bisection: two communities cut along their bridges</text>')
    # edges
    for (u, v) in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        crossing = labels[u] != labels[v]
        col = CUT if crossing else EDGE
        wdt = 2.5 if crossing else 1.2
        dash = ' stroke-dasharray="4,3"' if crossing else ""
        s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{col}" stroke-width="{wdt}"{dash}/>')
    # nodes colored by side
    for i in range(n):
        x, y = pos[i]
        col = BLUE if labels[i] == 0 else RED
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="{col}" '
                 f'stroke="{BG}" stroke-width="2"/>')
        s.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="{BG}" font-size="11" '
                 f'text-anchor="middle" font-weight="bold">{i}</text>')
    # legend
    s.append(f'<circle cx="30" cy="{H-60}" r="8" fill="{BLUE}"/>')
    s.append(f'<text x="45" y="{H-56}" fill="{TEXT}" font-size="12">side A (Fiedler &lt; 0)</text>')
    s.append(f'<circle cx="30" cy="{H-36}" r="8" fill="{RED}"/>')
    s.append(f'<text x="45" y="{H-32}" fill="{TEXT}" font-size="12">side B (Fiedler &gt; 0)</text>')
    s.append(f'<line x1="240" y1="{H-40}" x2="270" y2="{H-40}" stroke="{CUT}" '
             f'stroke-width="2.5" stroke-dasharray="4,3"/>')
    s.append(f'<text x="278" y="{H-36}" fill="{TEXT}" font-size="12">cut edge (bridge)</text>')
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(s))


def main(outdir=None):
    n, edges = _build_graph()
    fval, fvec = SP.fiedler(n, edges)
    labels = SP.partition(n, edges)

    lines = []
    lines.append("Spectral graph partitioning via the Fiedler vector")
    lines.append("=" * 56)
    lines.append(f"graph: {n} vertices, {len(edges)} edges, two planted communities + 2 bridges")
    lines.append("")
    ncomp, _ = SP.connected_components(n, edges)
    lines.append(f"connected components (BFS):      {ncomp}")
    lines.append(f"near-zero Laplacian eigenvalues: {SP.count_zero_eigenvalues(n, edges)}  (= components)")
    lines.append(f"Fiedler value (algebraic conn.): {fval:.5f}  (> 0 => connected)")
    lines.append("")
    lines.append(f"{'vertex':>7}{'Fiedler entry':>16}{'side':>7}")
    for i in range(n):
        lines.append(f"{i:>7}{fvec[i]:>16.5f}{('B' if labels[i] else 'A'):>7}")
    lines.append("")
    cut = SP.cut_size(edges, labels)
    lines.append(f"cut size (edges crossing):  {cut:.0f}")
    lines.append(f"ratio cut:                  {SP.ratio_cut(n, edges, labels):.5f}")
    lines.append(f"normalized cut:             {SP.normalized_cut(n, edges, labels):.5f}")
    lines.append("The sign of each Fiedler entry assigns the vertex to a side; the cut lands on the")
    lines.append("bridge edges, recovering the two communities.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        _svg(os.path.join(outdir, "spectral_partition.svg"), n, edges, labels, fvec)

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
