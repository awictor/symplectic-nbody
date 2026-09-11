"""Demo: De Bruijn sequences via Eulerian circuits, and the window bijection.

Builds B(2,3), B(2,4), and a decimal PIN-pad sequence B(10,4), shows that every window appears
exactly once, and draws the B(2,3) De Bruijn graph with its Eulerian circuit.

    python examples/de_bruijn_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from de_bruijn import de_bruijn, windows, is_de_bruijn  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("De Bruijn sequences: every length-n window exactly once\n")

    # B(2,3): the classic 8-bit cyclic sequence
    b23 = de_bruijn(2, 3)
    print(f"  B(2,3) = {''.join(map(str, b23))}   (length {len(b23)} = 2^3)")
    print("  its 8 cyclic windows (each binary triple exactly once):")
    for w in windows(b23, 3):
        print(f"    {''.join(map(str, w))}")
    print(f"  is a valid De Bruijn sequence: {is_de_bruijn(b23, 2, 3)}\n")

    # B(2,4)
    b24 = de_bruijn(2, 4)
    print(f"  B(2,4) = {''.join(map(str, b24))}   (length {len(b24)} = 2^4)")
    print(f"  all 16 nibbles appear once: {is_de_bruijn(b24, 2, 4)}\n")

    # a PIN-pad sequence: shortest cyclic string trying every 4-digit PIN
    pin = de_bruijn(10, 4)
    print(f"  B(10,4): the shortest cyclic string containing all 10000 four-digit PINs")
    print(f"    length {len(pin)} (vs 40000 keypresses to type each PIN separately)")
    print(f"    first 40 digits: {''.join(map(str, pin[:40]))}...")
    print(f"    every PIN appears exactly once: {is_de_bruijn(pin, 10, 4)}\n")

    # a rotary-encoder style sequence
    enc = de_bruijn(2, 6)
    print(f"  B(2,6): a 64-position absolute rotary encoder track")
    print(f"    reading any 6 adjacent bits gives a unique angle: {is_de_bruijn(enc, 2, 6)}")

    print("\n  Each sequence is an Eulerian circuit of the De Bruijn graph: vertices are the")
    print("  (n-1)-length strings, edges are the n-length strings, and every vertex has equal in-")
    print("  and out-degree, so Hierholzer's cycle-splicing finds a walk using every edge once.")

    _svg(os.path.join(outdir, "de_bruijn.svg"), b23)
    print(f"\n  wrote {os.path.join(outdir, 'de_bruijn.svg')}")


def _svg(path, seq, width=760, height=430):
    import math
    # De Bruijn graph for B(2,3): vertices are the 4 two-bit strings 00,01,10,11
    verts = ["00", "01", "10", "11"]
    # place on a circle
    cx, cy, R = 380, 235, 150
    pos = {}
    for i, v in enumerate(verts):
        a = -math.pi / 2 + i * math.pi / 2
        pos[v] = (cx + R * math.cos(a), cy + R * math.sin(a))
    # edges: for each vertex w and symbol s, edge w -> w[1:]+s
    edges = []
    for w in verts:
        for s in "01":
            edges.append((w, w[1:] + s, s))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'De Bruijn graph B(2,3): {"".join(map(str, seq))}</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'vertices = 2-bit strings, edges = 3-bit strings; an Eulerian circuit reads off the sequence</text>',
        '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" '
        'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4dabf7"/></marker>'
        '<marker id="s" markerWidth="9" markerHeight="9" refX="8" refY="3" '
        'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#ffd43b"/></marker></defs>',
    ]
    for u, v, s in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        col = "#4dabf7" if s == "0" else "#ffd43b"
        mk = "url(#a)" if s == "0" else "url(#s)"
        if u == v:
            # self-loop (00->00 and 11->11)
            lx, ly = x0, y0
            off = -40 if u == "00" else 40
            parts.append(f'<path d="M{lx},{ly} q {off},{off} 0,{2*off if False else 0} " '
                         f'fill="none" stroke="{col}" stroke-width="1.8"/>')
            parts.append(f'<circle cx="{lx + off*0.9:.1f}" cy="{ly + off*0.9:.1f}" r="16" '
                         f'fill="none" stroke="{col}" stroke-width="1.8"/>')
            parts.append(f'<text x="{lx + off*1.6:.1f}" y="{ly + off*1.6:.1f}" fill="{col}" '
                         f'font-size="11" text-anchor="middle">{u+s}</text>')
            continue
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        # curve the edge so opposite-direction edges don't overlap
        nx, ny = -dy / L, dx / L
        bend = 26
        mx, my = (x0 + x1) / 2 + nx * bend, (y0 + y1) / 2 + ny * bend
        ax, ay = x0 + dx / L * 26, y0 + dy / L * 26
        bx, by = x1 - dx / L * 30, y1 - dy / L * 30
        parts.append(f'<path d="M{ax:.1f},{ay:.1f} Q{mx:.1f},{my:.1f} {bx:.1f},{by:.1f}" '
                     f'fill="none" stroke="{col}" stroke-width="1.8" marker-end="{mk}"/>')
        parts.append(f'<text x="{mx:.1f}" y="{my - 4:.1f}" fill="{col}" font-size="11" '
                     f'text-anchor="middle">{u+s}</text>')
    for v, (x, y) in pos.items():
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="24" fill="#161b22" '
                     f'stroke="#06d6a0" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y + 5:.1f}" fill="#e6edf3" font-size="14" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')
    # legend
    parts.append(f'<text x="20" y="{height-40}" fill="#4dabf7" font-size="12">blue edges append 0</text>')
    parts.append(f'<text x="20" y="{height-22}" fill="#ffd43b" font-size="12">yellow edges append 1</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
