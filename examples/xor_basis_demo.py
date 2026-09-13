"""Demo: the XOR linear basis -- subset-XOR questions answered by GF(2) linear algebra.

Builds a basis from a bag of numbers, finds the maximum achievable XOR, the number of distinct
reachable values, and enumerates them in order, then shows the exponential speedup over brute force.
Draws the basis vectors as a binary echelon form.

    python examples/xor_basis_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xor_basis import XorBasis, brute_reachable, brute_max_xor  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("XOR linear basis: 'largest XOR of a subset?' as linear algebra over GF(2)\n")

    nums = [26, 15, 10, 6, 21]
    b = XorBasis()
    for x in nums:
        b.insert(x)
    print(f"  bag: {nums}  (binary: {[format(x, '05b') for x in nums]})")
    print(f"    rank (span dimension): {b.rank()}")
    print(f"    distinct reachable XOR values: {b.count()} = 2^{b.rank()}")
    print(f"    maximum subset XOR: {b.max_xor()}  ({b.max_xor():05b})   "
          f"[brute force: {brute_max_xor(nums)}]")
    print(f"    minimum nonempty subset XOR: {b.min_xor()}")
    print(f"    is 31 reachable? {b.contains(31)}    is 1 reachable? {b.contains(1)}\n")

    print(f"  all {b.count()} reachable values in order:")
    vals = [b.kth_smallest(k) for k in range(b.count())]
    print("    " + ", ".join(str(v) for v in vals))
    print(f"    (matches brute-force reachable set: {vals == sorted(brute_reachable(nums))})\n")

    print("  operation count for maximum subset XOR of n numbers, b bits:")
    for n in (20, 40, 60, 100):
        print(f"    n={n:3d}: brute force 2^{n} = {2**n:,}   basis ~ n*b = {n*64:,}")
    print()

    print("  Each number is a bit-vector; XOR is vector addition in GF(2). The reachable values are the")
    print("  linear span, so the basis is Gaussian elimination in binary: keep one pivot per bit,")
    print("  reduce each new number, and every query -- max, membership, count, k-th -- is O(bits).")

    _svg(os.path.join(outdir, "xor_basis.svg"), nums, b)
    print(f"\n  wrote {os.path.join(outdir, 'xor_basis.svg')}")


def _svg(path, nums, basis, width=760, height=380):
    width_bits = max(x.bit_length() for x in nums) if nums else 1
    pivots = sorted(basis.basis, reverse=True)
    vecs = [basis.basis[p] for p in pivots]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'XOR basis: input numbers reduced to a binary echelon basis</text>',
    ]
    cell = min(46, (width - 200) / width_bits)
    ox = 120

    def draw_bits(value, y, label, pivot_bit=None):
        out = [f'<text x="20" y="{y+cell*0.65:.0f}" fill="#8b949e" font-size="11">{label}</text>']
        for bi in range(width_bits):
            bit = (value >> (width_bits - 1 - bi)) & 1
            x = ox + bi * cell
            actual_bit = width_bits - 1 - bi
            is_pivot = (pivot_bit == actual_bit)
            fill = ("#ffd43b" if is_pivot else "#4dabf7") if bit else "#161b22"
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-3:.1f}" height="{cell-3:.1f}" '
                       f'rx="2" fill="{fill}" stroke="#30363d" stroke-width="0.5"/>')
            out.append(f'<text x="{x+(cell-3)/2:.1f}" y="{y+cell*0.5:.0f}" fill="#0d1117" '
                       f'font-size="10" text-anchor="middle">{bit}</text>')
        return out

    y = 55
    parts.append(f'<text x="20" y="{y-4}" fill="#8b949e" font-size="11">inputs:</text>')
    y += 12
    for x in nums:
        parts += draw_bits(x, y, str(x))
        y += cell + 3
    y += 12
    parts.append(f'<text x="20" y="{y-2}" fill="#ffd43b" font-size="11">'
                 f'basis (one pivot per bit, yellow):</text>')
    y += 12
    for p, v in zip(pivots, vecs):
        parts += draw_bits(v, y, f"{v}", pivot_bit=p)
        y += cell + 3
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
