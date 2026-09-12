"""Demo: rANS entropy coding -- compressing to near the Shannon limit at high speed.

Encodes text and skewed data with rANS, confirms exact round-trip, and compares the achieved bits per
symbol to the Shannon entropy (the theoretical floor) and to what Huffman's whole-bit codes could
manage. Draws the compression bar chart.

    python examples/rans_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rans import encode, decode, entropy, bits_per_symbol, build_model  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def randint(self, lo, hi):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return lo + (self.s >> 8) % (hi - lo + 1)


def huffman_bits_per_symbol(data):
    """Lower bound on Huffman's cost: build an optimal prefix code and measure its average length."""
    counts = {}
    for b in data:
        counts[b] = counts.get(b, 0) + 1
    if len(counts) <= 1:
        return 0.0 if data else 0.0
    import heapq
    heap = [[c, i, None] for i, (s, c) in enumerate(counts.items())]
    leaves = {id(node): s for node, (s, _) in zip(heap, [(n, (s, c)) for n, (s, c) in zip(heap, counts.items())])}
    # simpler: track depths
    heap = [(c, i) for i, c in enumerate(counts.values())]
    heapq.heapify(heap)
    # Huffman merge counting weighted depth
    freqs = list(counts.values())
    h = [(f, ) for f in freqs]
    heapq.heapify(h)
    total_bits = 0
    while len(h) > 1:
        a = heapq.heappop(h)
        b = heapq.heappop(h)
        merged = a[0] + b[0]
        total_bits += merged
        heapq.heappush(h, (merged,))
    return total_bits / len(data)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("rANS: entropy coding at the Shannon limit -- the coder inside Zstandard and JPEG XL\n")

    samples = []

    text = (b"the quick brown fox jumps over the lazy dog. " * 30)
    rng = LCG(7)
    skewed = bytes(0 if rng.randint(0, 99) < 80 else rng.randint(1, 6) for _ in range(6000))
    rng2 = LCG(11)
    uniform = bytes(rng2.randint(0, 15) for _ in range(6000))

    for name, data in (("english-ish text", text), ("80%-skewed source", skewed),
                       ("uniform 16 symbols", uniform)):
        enc, model = encode(data)
        ok = decode(enc, model, len(data)) == data
        H = entropy(data)
        bps = bits_per_symbol(data)
        huff = huffman_bits_per_symbol(data)
        ratio = 8 / bps if bps else 0
        samples.append((name, H, bps, huff))
        print(f"  {name}:")
        print(f"    {len(data)} bytes -> {len(enc)} bytes  ({ratio:.2f}x smaller), round-trip OK: {ok}")
        print(f"    Shannon entropy {H:.3f} bits/sym   rANS {bps:.3f}   Huffman {huff:.3f}")
    print()

    print("  rANS folds the whole message into one big integer, adding each symbol s by")
    print("    x -> (x // f_s) * M + (x % f_s) + c_s,")
    print("  a reversible step costing exactly -log2(p_s) bits. It reaches arithmetic coding's")
    print("  compression (unlike whole-bit Huffman) at table-lookup speed -- why modern codecs adopted it.")

    _svg(os.path.join(outdir, "rans.svg"), samples)
    print(f"\n  wrote {os.path.join(outdir, 'rans.svg')}")


def _svg(path, samples, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Bits per symbol: rANS hugs the Shannon entropy, below Huffman</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'green = Shannon entropy (floor); yellow = rANS; orange = Huffman (whole-bit)</text>',
    ]
    ox, oy = 60, 320
    pw, ph = width - 110, 240
    ymax = 8.0   # bits (byte)
    ngroups = len(samples)
    gw = pw / ngroups

    def py(v):
        return oy - v / ymax * ph

    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{ox}" y1="{py(8):.1f}" x2="{ox+pw}" y2="{py(8):.1f}" stroke="#30363d" '
                 f'stroke-width="0.7" stroke-dasharray="2 4"/>')
    parts.append(f'<text x="{ox-8}" y="{py(8)+4:.0f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">8 (raw)</text>')

    barw = gw / 4
    for gi, (name, H, bps, huff) in enumerate(samples):
        base = ox + gi * gw + gw * 0.1
        for j, (v, col) in enumerate(((H, "#06d6a0"), (bps, "#ffd43b"), (huff, "#ff922b"))):
            x = base + j * barw
            h = oy - py(v)
            parts.append(f'<rect x="{x:.1f}" y="{py(v):.1f}" width="{barw-3:.1f}" height="{h:.1f}" '
                         f'fill="{col}"/>')
            parts.append(f'<text x="{x+(barw-3)/2:.1f}" y="{py(v)-4:.1f}" fill="{col}" font-size="9" '
                         f'text-anchor="middle">{v:.2f}</text>')
        parts.append(f'<text x="{base + 1.5*barw:.1f}" y="{oy+16}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{name}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
