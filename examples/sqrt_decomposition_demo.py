"""Sqrt-decomposition demo: a range query touching partial + whole blocks, with the block layout (SVG)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sqrt_decomposition as SD


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
YELLOW = "#ffd43b"
EDGE = "#30363d"


def main(outdir=None):
    a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]   # n=16, block size 4
    sd = SD.SqrtDecomposition(a, "sum")

    lines = []
    lines.append("Square-root decomposition: O(sqrt n) range queries")
    lines.append("=" * 54)
    lines.append(f"array of {sd.n} elements, block size {sd.block}, {sd.block_count()} blocks")
    lines.append(f"block sums: {sd.blocks}")
    lines.append("")
    l, r = 2, 13
    lines.append(f"query sum[{l}..{r}]:")
    lines.append(f"  = left partial (indices {l}..{(l//sd.block+1)*sd.block-1})")
    lines.append(f"  + whole blocks {l//sd.block+1}..{r//sd.block-1} (from precomputed sums)")
    lines.append(f"  + right partial (indices {r//sd.block*sd.block}..{r})")
    lines.append(f"  = {sd.query(l, r)}  (brute force: {SD.brute_query(a, l, r, 'sum')})")
    lines.append("")
    lines.append("cost: O(sqrt n) partial elements + O(sqrt n) whole blocks, vs O(n) naive.")
    lines.append("")
    lines.append("an update recomputes just one block:")
    sd.update(7, 100)
    lines.append(f"  after a[7] = 100, sum[6..8] = {sd.query(6, 8)} (was {2+6+5})")
    lines.append("")
    lines.append("No tree, no recursion -- just an array plus a short block table; the pragmatic")
    lines.append("middle ground and the engine behind Mo's algorithm for offline range queries.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # redraw with the original array for clarity
        a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]
        sd = SD.SqrtDecomposition(a, "sum")
        l, r = 2, 13
        W, H = 720, 320
        ml, mt = 40, 90
        cw = (W - 2 * ml) / sd.n
        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Range query sum[{l}..{r}]: partial ends (green) + whole blocks (yellow)</text>')
        bl = l // sd.block
        br = r // sd.block
        # classify each index
        for i in range(sd.n):
            x = ml + i * cw
            bi = i // sd.block
            if l <= i <= r:
                if bl < bi < br:
                    col = YELLOW      # covered by a whole block
                else:
                    col = GREEN       # partial end walked element by element
            else:
                col = "#161b22"
            s.append(f'<rect x="{x:.1f}" y="{mt}" width="{cw-2:.1f}" height="40" fill="{col}" '
                     f'fill-opacity="0.5" stroke="{EDGE}"/>')
            s.append(f'<text x="{x+(cw-2)/2:.1f}" y="{mt+26:.1f}" fill="{TEXT}" font-size="11" '
                     f'text-anchor="middle">{a[i]}</text>')
            s.append(f'<text x="{x+(cw-2)/2:.1f}" y="{mt-6:.1f}" fill="{GRAY}" font-size="8" '
                     f'text-anchor="middle">{i}</text>')
        # block boundaries + block sums
        for bi in range(sd.block_count()):
            bx = ml + bi * sd.block * cw
            bwid = min(sd.block, sd.n - bi * sd.block) * cw
            s.append(f'<line x1="{bx:.1f}" y1="{mt-14}" x2="{bx:.1f}" y2="{mt+58}" '
                     f'stroke="{BLUE}" stroke-width="1"/>')
            s.append(f'<text x="{bx+bwid/2:.1f}" y="{mt+74:.1f}" fill="{BLUE}" font-size="10" '
                     f'text-anchor="middle">blk sum {sd.blocks[bi]}</text>')
        s.append(f'<text x="{ml}" y="{H-40}" fill="{GREEN}" font-size="11">green = partial blocks '
                 f'(walked element by element)</text>')
        s.append(f'<text x="{ml}" y="{H-24}" fill="{YELLOW}" font-size="11">yellow = whole blocks '
                 f'(read from precomputed sums)</text>')
        s.append(f'<text x="{ml}" y="{H-8}" fill="{GRAY}" font-size="10">'
                 f'total sum[{l}..{r}] = {sd.query(l, r)}; only ~2*sqrt(n) elements + sqrt(n) blocks '
                 f'are touched.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "sqrt_decomposition.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
