"""Demo: PageRank on a small web graph.

Ranks a hand-built miniature web by the stationary distribution of the random surfer, showing that
importance flows from important linkers (not raw in-degree), that the scores are a fixed point
summing to one, and that personalizing the teleport re-weights the whole graph. Draws the graph
with each node sized by its PageRank and the power-iteration convergence curve.

    python examples/pagerank_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pagerank import pagerank, top_k  # noqa: E402


# a small directed "web": a popular hub, a couple of authorities, and some spokes
GRAPH = {
    "home": ["about", "blog", "shop"],
    "about": ["home"],
    "blog": ["home", "post1", "post2"],
    "post1": ["blog", "shop"],
    "post2": ["blog"],
    "shop": ["home"],
    "ad": ["home"],          # a spammy page linking to home, few link back
    "orphan": ["home"],      # links out but nobody links in
}


def _converge_trace(graph, damping=0.85, steps=30):
    """Reproduce the power iteration, recording L1 change per step to show convergence."""
    nodes = set(graph)
    for outs in graph.values():
        nodes.update(outs)
    nodes = sorted(nodes, key=str)
    n = len(nodes)
    idx = {nd: i for i, nd in enumerate(nodes)}
    out = [[] for _ in range(n)]
    for s, outs in graph.items():
        for d in outs:
            out[idx[s]].append(idx[d])
    deg = [len(o) for o in out]
    rank = [1.0 / n] * n
    errs = []
    for _ in range(steps):
        new = [0.0] * n
        dang = 0.0
        for i in range(n):
            if deg[i] == 0:
                dang += rank[i]
            else:
                sh = rank[i] / deg[i]
                for j in out[i]:
                    new[j] += sh
        leaked = damping * dang
        for j in range(n):
            new[j] = damping * new[j] + (1 - damping) / n + leaked / n
        errs.append(sum(abs(new[i] - rank[i]) for i in range(n)))
        rank = new
    return errs


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    scores = pagerank(GRAPH)

    print("PageRank: ranking a small web by the random surfer's stationary distribution\n")
    print(f"  {len(scores)} pages, {sum(len(v) for v in GRAPH.values())} links, damping 0.85\n")

    print("  Rank (share of surfer's time), highest first:")
    in_deg = {}
    for outs in GRAPH.values():
        for d in outs:
            in_deg[d] = in_deg.get(d, 0) + 1
    for node, sc in top_k(scores, len(scores)):
        bar = "#" * int(round(sc * 120))
        print(f"    {node:>7}: {sc:.4f}  in-links {in_deg.get(node, 0)}  {bar}")

    print(f"\n  Scores sum to {sum(scores.values()):.6f} (a probability distribution).")
    print("  Note 'home' wins not just on in-link COUNT but because important pages link to it --")
    print("  that recursive definition is the whole point; raw in-degree would tie several pages.\n")

    # personalization: a surfer who always teleports back to the shop
    personal = pagerank(GRAPH, personalization={"shop": 1.0})
    print("  Personalized PageRank (teleport home = 'shop') re-weights the graph:")
    for node in ("shop", "home", "blog"):
        print(f"    {node:>7}: uniform {scores[node]:.4f} -> shop-personalized {personal[node]:.4f}")

    errs = _converge_trace(GRAPH)
    print(f"\n  Power iteration converges geometrically (rate ~ damping = 0.85):")
    for s in (0, 1, 2, 4, 8, 16):
        if s < len(errs):
            print(f"    step {s:>2}: L1 change {errs[s]:.2e}")

    print("\n  The teleport term makes the Google matrix strictly positive, so Perron-Frobenius")
    print("  guarantees a unique stationary vector and power iteration converges at rate d.")

    _svg(os.path.join(outdir, "pagerank.svg"), GRAPH, scores, errs)
    print(f"\n  wrote {os.path.join(outdir, 'pagerank.svg')}")


def _svg(path, graph, scores, errs, width=760, height=430):
    nodes = sorted(scores, key=str)
    n = len(nodes)
    # lay nodes on a circle in the left panel
    lcx, lcy, lr = 200, 245, 145
    pos = {}
    for i, nd in enumerate(nodes):
        ang = 2 * math.pi * i / n - math.pi / 2
        pos[nd] = (lcx + lr * math.cos(ang), lcy + lr * math.sin(ang))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'PageRank: node size is stationary probability; edges are links</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'importance flows from important linkers; right: power-iteration convergence</text>',
        '<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#484f58"/></marker></defs>',
    ]

    # edges with arrowheads, shortened so they don't overlap the node circles
    for s, outs in graph.items():
        x0, y0 = pos[s]
        for d in outs:
            x1, y1 = pos[d]
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy) or 1.0
            r_s = 6 + scores[s] * 90
            r_d = 6 + scores[d] * 90
            ax, ay = x0 + dx / L * r_s, y0 + dy / L * r_s
            bx, by = x1 - dx / L * (r_d + 6), y1 - dy / L * (r_d + 6)
            parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                         f'stroke="#30363d" stroke-width="1.2" marker-end="url(#arr)"/>')

    # nodes sized by rank
    top = max(scores.values())
    for nd in nodes:
        x, y = pos[nd]
        rad = 6 + scores[nd] * 90
        hot = scores[nd] / top
        col = f"rgb({int(77 + hot*178)},{int(120 + hot*40)},{int(247 - hot*140)})"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{col}" '
                     f'stroke="#0d1117" stroke-width="1.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{y + rad + 11:.1f}" fill="#e6edf3" font-size="10" '
                     f'text-anchor="middle">{nd} {scores[nd]:.2f}</text>')

    # right panel: convergence (log-scale L1 error)
    rx0, rx1 = width // 2 + 60, width - 30
    ry0, ry1 = height - 55, 75
    logs = [math.log10(max(e, 1e-16)) for e in errs]
    lo, hi = min(logs), max(logs)

    def RX(i):
        return rx0 + i / (len(errs) - 1) * (rx1 - rx0)

    def RY(v):
        return ry0 - (v - lo) / (hi - lo + 1e-9) * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = " ".join(f"{RX(i):.1f},{RY(logs[i]):.1f}" for i in range(len(errs)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2.2"/>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">iteration -> L1 change (log scale)</text>')
    parts.append(f'<text x="{rx0-4:.1f}" y="{ry1:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">1e{int(hi)}</text>')
    parts.append(f'<text x="{rx0-4:.1f}" y="{ry0:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">1e{int(lo)}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
