"""Demo: k-core decomposition -- peeling a network into its onion-like layers.

Computes the coreness of every node in a small network, reports the degeneracy and the k-shells, and
draws the graph with nodes coloured and sized by how deep in the core they sit.

    python examples/k_core_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from k_core import (coreness, degeneracy, shells, k_core, brute_coreness)  # noqa: E402

# a network with a dense core {0,1,2,3} (near-complete), a middle ring, and pendant leaves
EDGES = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),   # core: K4
    (0, 4), (1, 4), (4, 5), (2, 5), (5, 6),            # middle
    (3, 7), (6, 8), (7, 9),                            # periphery / leaves
]
N = 10


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    core = coreness(N, EDGES)
    deg = degeneracy(N, EDGES)
    sh = shells(N, EDGES)

    print("k-core decomposition: peeling a network to its dense heart\n")
    print(f"  {N} nodes, {len(EDGES)} edges\n")

    print(f"  coreness of each node: {core}")
    print(f"  matches brute-force peeling: {core == brute_coreness(N, EDGES)}")
    print(f"  degeneracy (max coreness): {deg}\n")

    print("  onion shells (nodes by coreness, periphery -> core):")
    for k in sorted(sh):
        print(f"    {k}-shell: {sorted(sh[k])}")

    print(f"\n  the {deg}-core (densest layer): {k_core(N, EDGES, deg)}")
    print("\n  Smallest-last peeling removes a minimum-degree vertex at a time; the degree it has when")
    print("  removed is its coreness. The maximum coreness is the degeneracy, and the resulting order")
    print("  bounds clique- and coloring-algorithm runtimes. High-coreness nodes are the robust,")
    print("  deeply-embedded core of the network -- the influential centre in social-network analysis.")

    _svg(os.path.join(outdir, "k_core.svg"), core, deg)
    print(f"\n  wrote {os.path.join(outdir, 'k_core.svg')}")


_SHELL_COLORS = ["#484f58", "#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b"]


def _svg(path, core, deg, width=760, height=440):
    # place nodes in concentric rings by coreness: deeper core -> nearer centre
    cx, cy = width / 2, height / 2 + 15
    by_core = {}
    for v in range(N):
        by_core.setdefault(core[v], []).append(v)
    pos = {}
    max_core = deg if deg > 0 else 1
    for k, verts in by_core.items():
        # radius shrinks with coreness
        radius = 40 + (max_core - k) * 70
        for i, v in enumerate(verts):
            ang = 2 * math.pi * i / len(verts) + k    # offset per ring
            pos[v] = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'k-core decomposition: nodes nearer the centre sit deeper in the core</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'colour and size grow with coreness; the {deg}-core is the dense hub, leaves are the '
        f'periphery</text>',
    ]

    for u, v in EDGES:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#30363d" stroke-width="1.3"/>')

    for v, (x, y) in pos.items():
        col = _SHELL_COLORS[core[v] % len(_SHELL_COLORS)]
        r = 10 + core[v] * 5
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" fill="#0d1117" font-size="11" '
                     f'text-anchor="middle" font-weight="bold">{v}</text>')

    # legend
    for i, k in enumerate(sorted(by_core)):
        yy = 90 + i * 22
        parts.append(f'<circle cx="30" cy="{yy}" r="8" fill="{_SHELL_COLORS[k % len(_SHELL_COLORS)]}"/>')
        parts.append(f'<text x="45" y="{yy+4}" fill="#8b949e" font-size="11">coreness {k}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
