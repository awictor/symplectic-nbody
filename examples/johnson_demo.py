"""Johnson demo: reweight a graph with negative edges to non-negative, then all-pairs shortest paths (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import johnson as J


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"
EDGE = "#484f58"


def _node_positions(n):
    import math as m
    pos = {}
    cx, cy, r = 0, 0, 1.0
    for i in range(n):
        ang = m.pi / 2 - 2 * m.pi * i / n
        pos[i] = (cx + r * m.cos(ang), cy + r * m.sin(ang))
    return pos


def _graph_panel(s, n, edges, ox, oy, side, title, weights_override=None):
    pos = _node_positions(n)

    def sx(x):
        return ox + (x + 1.3) / 2.6 * side

    def sy(y):
        return oy + side - (y + 1.3) / 2.6 * side

    s.append(f'<text x="{ox+side/2:.0f}" y="{oy-8}" fill="{TEXT}" font-size="12" '
             f'text-anchor="middle">{title}</text>')
    for idx, (u, v, w) in enumerate(edges):
        wv = w if weights_override is None else weights_override[idx]
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        col = RED if wv < -1e-9 else EDGE
        # shorten toward the target so arrowhead shows
        X1, Y1, X2, Y2 = sx(x1), sy(y1), sx(x2), sy(y2)
        dx, dy = X2 - X1, Y2 - Y1
        L = math.hypot(dx, dy) or 1
        X2b, Y2b = X2 - dx / L * 16, Y2 - dy / L * 16
        s.append(f'<line x1="{X1:.1f}" y1="{Y1:.1f}" x2="{X2b:.1f}" y2="{Y2b:.1f}" '
                 f'stroke="{col}" stroke-width="1.4"/>')
        mx, my = (X1 + X2b) / 2, (Y1 + Y2b) / 2
        s.append(f'<text x="{mx:.1f}" y="{my-3:.1f}" fill="{col if wv<0 else GRAY}" '
                 f'font-size="10" text-anchor="middle">{wv:g}</text>')
    for i in range(n):
        x, y = pos[i]
        s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="12" fill="{BLUE}" '
                 f'stroke="{BG}" stroke-width="2"/>')
        s.append(f'<text x="{sx(x):.1f}" y="{sy(y)+4:.1f}" fill="{BG}" font-size="11" '
                 f'text-anchor="middle" font-weight="bold">{i}</text>')


def _svg(path, n, edges):
    rw, h = J.reweighted_edges(n, edges)
    rww = [w for _, _, w in rw]
    W, H = 720, 400
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    s.append(f'<text x="30" y="26" fill="{TEXT}" font-size="15">'
             f'Johnson reweighting: negative edges (red) become non-negative</text>')
    _graph_panel(s, n, edges, 60, 60, 280, "original (negative edges in red)")
    _graph_panel(s, n, edges, 400, 60, 280, "reweighted w + h(u) - h(v) >= 0", rww)
    s.append(f'<text x="60" y="{H-16}" fill="{GRAY}" font-size="10">'
             f'The potentials h(v) = dist from a virtual source shift every edge so none is negative, '
             f'while every shortest path is preserved -- Dijkstra is now safe from each source.</text>')
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(s))


def main(outdir=None):
    # CLRS Johnson example
    edges = [(0, 1, 3), (0, 2, 8), (0, 4, -4), (1, 3, 1), (1, 4, 7),
             (2, 1, 4), (3, 0, 2), (3, 2, -5), (4, 3, 6)]
    n = 5
    dist, nh = J.johnson(n, edges)
    rw, h = J.reweighted_edges(n, edges)

    lines = []
    lines.append("Johnson's algorithm: all-pairs shortest paths (sparse, negative edges)")
    lines.append("=" * 68)
    lines.append(f"{n} nodes, {len(edges)} directed edges, {sum(1 for _,_,w in edges if w<0)} negative")
    lines.append("")
    lines.append("Bellman-Ford potentials h(v) from a virtual source:")
    lines.append("  " + "  ".join(f"h[{v}]={h[v]:g}" for v in range(n)))
    lines.append("")
    lines.append("Reweighted edge weights w'(u,v) = w + h(u) - h(v) (all >= 0):")
    for (u, v, w), (_, _, wp) in zip(edges, rw):
        flag = "  <- was negative" if w < 0 else ""
        lines.append(f"  {u}->{v}: {w:>4g}  ->  {wp:>4g}{flag}")
    lines.append("")
    lines.append("All-pairs shortest distance matrix:")
    header = "      " + "".join(f"{t:>6}" for t in range(n))
    lines.append(header)
    for sname in range(n):
        row = f"  {sname}:  " + "".join(
            (f"{dist[sname][t]:>6g}" if dist[sname][t] != math.inf else f"{'inf':>6}")
            for t in range(n))
        lines.append(row)
    lines.append("")
    p = J.reconstruct_path(nh, 0, 2)
    lines.append(f"shortest path 0 -> 2: {' -> '.join(map(str, p))}  (length {dist[0][2]:g})")
    lines.append("Reweighting keeps every shortest path identical; only Dijkstra's safety changes.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        _svg(os.path.join(outdir, "johnson.svg"), n, edges)

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
