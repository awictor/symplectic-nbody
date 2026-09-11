"""Demo: a wavelet tree answering rank, select, quantile, and range-count queries.

Builds a wavelet tree over a small integer sequence, runs the four query types against brute force,
and draws the tree's recursive alphabet partition with the per-level bit vectors.

    python examples/wavelet_tree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wavelet_tree import WaveletTree  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Wavelet tree: rank, select, quantile, and range-count in O(log sigma)\n")

    seq = [3, 1, 4, 1, 5, 2, 6, 5, 3, 5, 0, 7, 2, 6, 4]
    wt = WaveletTree(seq)
    print(f"  sequence: {seq}")
    print(f"  alphabet range: [{wt.lo}, {wt.hi}]\n")

    print("  RANK -- how many times a value appears in a prefix:")
    for v, i in [(5, len(seq)), (5, 8), (2, len(seq))]:
        print(f"    rank({v}, {i}) = {wt.rank(v, i)}   (brute {seq[:i].count(v)})")

    print("\n  SELECT -- position of the j-th occurrence (0-indexed):")
    for v, j in [(5, 0), (5, 2), (3, 1)]:
        pos = wt.select(v, j)
        print(f"    select({v}, {j}) = {pos}   (seq[{pos}] = {seq[pos]})")

    print("\n  QUANTILE -- k-th smallest value in a range (range median and friends):")
    for lo, hi, k in [(0, len(seq), 7), (2, 9, 0), (2, 9, 6), (4, 11, 3)]:
        got = wt.quantile(lo, hi, k)
        want = sorted(seq[lo:hi])[k]
        print(f"    quantile([{lo},{hi}), k={k}) = {got}   (sorted slice [{k}] = {want})")

    print("\n  RANGE_COUNT -- values in a positional range that fall in a value window:")
    for lo, hi, vlo, vhi in [(0, len(seq), 3, 5), (0, 8, 0, 2), (5, 15, 4, 7)]:
        got = wt.range_count(lo, hi, vlo, vhi)
        want = sum(1 for p in range(lo, hi) if vlo <= seq[p] <= vhi)
        print(f"    range_count([{lo},{hi}), {vlo}..{vhi}) = {got}   (brute {want})")

    print("\n  The tree splits the alphabet at its midpoint level by level; each node stores one bit")
    print("  per element (upper half = 1, lower half = 0). Every query walks the O(log sigma)-deep")
    print("  tree, using bit-rank to map an index into the correct child -- succinct and fast.")

    _svg(os.path.join(outdir, "wavelet_tree.svg"), wt, seq)
    print(f"\n  wrote {os.path.join(outdir, 'wavelet_tree.svg')}")


def _svg(path, wt, seq, width=760, height=460):
    # collect nodes level by level
    levels = []

    def gather(node, depth, values):
        if node is None:
            return
        while len(levels) <= depth:
            levels.append([])
        levels[depth].append((node, values))
        if node.lo != node.hi and node.bits is not None:
            left_vals = [v for v in values if v <= (node.lo + node.hi) // 2]
            right_vals = [v for v in values if v > (node.lo + node.hi) // 2]
            gather(node.left, depth + 1, left_vals)
            gather(node.right, depth + 1, right_vals)

    gather(wt.root, 0, list(seq))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Wavelet tree: recursive alphabet partition with per-node bit vectors</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each node shows its value range and bits (0 = lower half / goes left, 1 = upper half / right)</text>',
    ]

    n_levels = len(levels)
    row_h = (height - 90) / max(1, n_levels)
    for d, level in enumerate(levels):
        y = 80 + d * row_h
        count = len(level)
        for idx, (node, values) in enumerate(level):
            x = (width) * (idx + 0.5) / count
            rng_lbl = f"[{node.lo},{node.hi}]" if node.lo != node.hi else f"={node.lo}"
            is_leaf = node.lo == node.hi
            col = "#ffd43b" if is_leaf else "#4dabf7"
            parts.append(f'<text x="{x:.0f}" y="{y:.0f}" fill="{col}" font-size="12" '
                         f'text-anchor="middle" font-weight="bold">{rng_lbl}</text>')
            if not is_leaf and node.bits is not None:
                bitstr = "".join(str(b) for b in node.bits)
                if len(bitstr) > 40:
                    bitstr = bitstr[:38] + ".."
                parts.append(f'<text x="{x:.0f}" y="{y+15:.0f}" fill="#8b949e" font-size="10" '
                             f'text-anchor="middle">{bitstr}</text>')
            else:
                parts.append(f'<text x="{x:.0f}" y="{y+15:.0f}" fill="#06d6a0" font-size="10" '
                             f'text-anchor="middle">x{node._count}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
