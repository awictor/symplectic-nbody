"""Demo: Merkle tree inclusion proofs -- verify membership in O(log n) hashes.

Builds a Merkle tree over transaction-like blocks, produces an inclusion proof for one block, verifies
it against the root, shows that tampering breaks it, and that proof size grows logarithmically. Draws
the hash tree with a highlighted authentication path.

    python examples/merkle_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from merkle import MerkleTree, verify  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Merkle tree: O(log n) proofs that a block belongs to a dataset\n")

    blocks = [b"tx: Alice->Bob 5", b"tx: Bob->Cy 2", b"tx: Cy->Dot 1", b"tx: Dot->Eve 3",
              b"tx: Eve->Ann 4", b"tx: Ann->Cy 6", b"tx: Bob->Eve 1", b"tx: Dot->Ann 2"]
    t = MerkleTree(blocks)
    print(f"  {len(blocks)} transaction blocks -> Merkle root {t.root.hex()[:24]}...")

    idx = 3
    proof = t.proof(idx)
    print(f"\n  Inclusion proof for block {idx} ({blocks[idx].decode()!r}):")
    print(f"    {len(proof)} sibling hashes (log2({len(blocks)}) = {int(math.log2(len(blocks)))}):")
    for h, is_left in proof:
        side = "left " if is_left else "right"
        print(f"      {side} sibling: {h.hex()[:24]}...")
    print(f"    verifies against the root: {verify(blocks[idx], idx, proof, t.root)}")
    print("    (a verifier who trusts only the root confirms membership without the other blocks)")

    # tampering
    print(f"\n  Tamper detection:")
    print(f"    altered block verifies: {verify(b'tx: Dot->Eve 9999', idx, proof, t.root)} (rejected)")
    bad_proof = [(h[::-1], f) for h, f in proof]
    print(f"    forged proof verifies:  {verify(blocks[idx], idx, bad_proof, t.root)} (rejected)")

    # logarithmic scaling
    print(f"\n  Proof size grows logarithmically (download shrinks from O(n) to O(log n)):")
    for n in [8, 64, 1024, 1_000_000]:
        bs = [b"b%d" % i for i in range(min(n, 4096))]     # cap actual build, extrapolate proof len
        tt = MerkleTree(bs)
        plen = len(tt.proof(0)) if n <= 4096 else int(math.ceil(math.log2(n)))
        print(f"    {n:>9} blocks: proof ~ {int(math.ceil(math.log2(n)))} hashes "
              f"({int(math.ceil(math.log2(n))) * 32} bytes) vs {n} blocks to download")

    print("\n  Each leaf is a block's hash; each parent hashes its two children; the root fingerprints")
    print("  the whole ordered set. An inclusion proof is the sibling hash at each level up to the")
    print("  root -- recompute the path, check it lands on the trusted root. Flip any bit and it won't.")

    _svg(os.path.join(outdir, "merkle.svg"), t, idx)
    print(f"\n  wrote {os.path.join(outdir, 'merkle.svg')}")


def _svg(path, tree, proof_idx, width=760, height=430):
    levels = tree.levels
    n_levels = len(levels)

    # authentication path: which nodes are siblings used in the proof
    sibling_nodes = set()
    path_nodes = set()
    idx = proof_idx
    for lvl, level in enumerate(levels[:-1]):
        path_nodes.add((lvl, idx))
        if idx % 2 == 0 and idx + 1 < len(level):
            sibling_nodes.add((lvl, idx + 1))
        elif idx % 2 == 1:
            sibling_nodes.add((lvl, idx - 1))
        idx //= 2
    path_nodes.add((n_levels - 1, 0))     # root

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Merkle tree: the authentication path for one leaf</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'green = the proven leaf and its path to the root; gold = the sibling hashes in the proof</text>',
    ]

    row_h = (height - 100) / n_levels
    pos = {}
    for lvl in range(n_levels):
        level = levels[lvl]
        m = len(level)
        y = height - 60 - lvl * row_h
        for i in range(m):
            x = width * (i + 0.5) / m
            pos[(lvl, i)] = (x, y)

    # edges
    for lvl in range(n_levels - 1):
        for i in range(len(levels[lvl])):
            parent = (lvl + 1, i // 2)
            if parent in pos:
                x0, y0 = pos[(lvl, i)]
                x1, y1 = pos[parent]
                parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                             f'stroke="#30363d" stroke-width="1"/>')
    # nodes
    for (lvl, i), (x, y) in pos.items():
        if (lvl, i) in path_nodes:
            col = "#06d6a0"
        elif (lvl, i) in sibling_nodes:
            col = "#ffd43b"
        else:
            col = "#30363d"
        r = 9 if lvl == 0 else 11
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}" '
                     f'stroke="#0d1117" stroke-width="1.5"/>')

    parts.append(f'<text x="{width/2:.0f}" y="{height-20}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">leaves (block hashes) at the bottom, root at the top</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
