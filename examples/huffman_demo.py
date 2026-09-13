"""Demo: Huffman coding -- the optimal prefix-free code.

Builds a Huffman code for a sample text, shows the codeword table (rarest symbols deepest), compares
the compressed size against fixed-width and against Shannon entropy, and draws the code tree.

    python examples/huffman_demo.py [output_dir]
"""

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from huffman import (  # noqa: E402
    build_code,
    canonical_code,
    expected_length,
    entropy,
    encode,
    decode,
    compression_ratio,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    text = "the quick brown fox jumps over the lazy dog the fox"
    freqs = Counter(text)
    n = len(freqs)

    print("Huffman coding: the optimal prefix-free code\n")
    print(f"  Message ({len(text)} chars, {n} distinct symbols):")
    print(f"    \"{text}\"\n")

    code = build_code(freqs)
    canon = canonical_code(freqs)
    print(f"  {'symbol':>8}  {'freq':>4}  {'Huffman':>10}  {'canonical':>10}")
    for sym in sorted(freqs, key=lambda s: (-freqs[s], s)):
        disp = "' '" if sym == " " else repr(sym)
        print(f"  {disp:>8}  {freqs[sym]:>4}  {code[sym]:>10}  {canon[sym]:>10}")

    L = expected_length(freqs)
    H = entropy(freqs)
    fixed = math.ceil(math.log2(n))
    encoded = encode(text, code)
    print(f"\n  Shannon entropy:        {H:.4f} bits/symbol")
    print(f"  Huffman expected length:{L:.4f} bits/symbol  (H <= L < H+1)")
    print(f"  Fixed-width code:       {fixed} bits/symbol")
    print(f"  Compression ratio vs fixed-width: {compression_ratio(freqs):.3f}x")
    print(f"  Encoded size: {len(encoded)} bits vs {len(text) * fixed} bits fixed "
          f"vs {len(text) * 8} bits ASCII")

    decoded = "".join(decode(encoded, code))
    print(f"  Round-trip decode matches original: {decoded == text}")

    print("\n  The greedy merge of the two rarest nodes is provably optimal (exchange argument);")
    print("  no prefix code has smaller expected length. Rare symbols sit deepest in the tree.")

    _svg(os.path.join(outdir, "huffman.svg"), freqs, code)
    print(f"\n  wrote {os.path.join(outdir, 'huffman.svg')}")


def _svg(path, freqs, code, width=760, height=460):
    # rebuild the tree structure from the codewords to lay it out
    # each codeword is a root->leaf path (0=left, 1=right)
    root = {}
    for sym, cw in code.items():
        node = root
        for b in cw:
            node = node.setdefault(b, {})
        node["sym"] = sym

    # assign x by in-order leaf index, y by depth
    leaves = []

    def collect(node, depth):
        if "sym" in node and len(node) == 1:
            leaves.append((node["sym"], depth))
            return
        for b in ("0", "1"):
            if b in node:
                collect(node[b], depth + 1)

    collect(root, 0)
    maxdepth = max(d for _, d in leaves) if leaves else 1

    # position leaves evenly; internal nodes at midpoint of children
    pos = {}
    leaf_x = {}
    for i, (sym, d) in enumerate(leaves):
        leaf_x[sym] = 40 + (width - 80) * (i + 0.5) / len(leaves)

    def layout(node, depth, pathbits):
        y = 60 + depth * (height - 120) / max(1, maxdepth)
        if "sym" in node and len(node) == 1:
            x = leaf_x[node["sym"]]
            pos[pathbits] = (x, y, node["sym"])
            return x
        xs = []
        for b in ("0", "1"):
            if b in node:
                xs.append(layout(node[b], depth + 1, pathbits + b))
        x = sum(xs) / len(xs)
        pos[pathbits] = (x, y, None)
        return x

    layout(root, 0, "")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="30" fill="#e6edf3" font-size="15">'
        'Huffman code tree (left=0, right=1; leaves are symbols)</text>',
    ]
    # edges
    for bits, (x, y, sym) in pos.items():
        for b in ("0", "1"):
            child = bits + b
            if child in pos:
                cx, cy, _ = pos[child]
                col = "#4dabf7" if b == "0" else "#ffd43b"
                parts.append(f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{cx:.0f}" y2="{cy:.0f}" '
                             f'stroke="{col}" stroke-width="1.5"/>')
                mx, my = (x + cx) / 2, (y + cy) / 2
                parts.append(f'<text x="{mx:.0f}" y="{my:.0f}" fill="{col}" font-size="9">{b}</text>')
    # nodes
    for bits, (x, y, sym) in pos.items():
        if sym is not None:
            disp = "sp" if sym == " " else sym
            parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="13" fill="#161b22" '
                         f'stroke="#06d6a0" stroke-width="2"/>')
            parts.append(f'<text x="{x:.0f}" y="{y+4:.0f}" fill="#06d6a0" font-size="10" '
                         f'text-anchor="middle">{disp}</text>')
            parts.append(f'<text x="{x:.0f}" y="{y+26:.0f}" fill="#8b949e" font-size="7" '
                         f'text-anchor="middle">{freqs[sym]}</text>')
        else:
            parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="4" fill="#30363d"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
