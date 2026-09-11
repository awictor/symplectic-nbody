"""Demo: arithmetic coding vs Huffman vs the Shannon entropy.

Encodes messages of varying skew and compares the achieved bits-per-symbol of arithmetic coding
against Huffman (which is stuck with whole-bit codewords) and the entropy lower bound -- showing
arithmetic coding hug the entropy where Huffman leaves bits on the table. Confirms lossless
round-trips throughout.

    python examples/arithmetic_coding_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arithmetic_coding import encode, decode, entropy, encoded_bits_per_symbol  # noqa: E402
from shannon import huffman_code, average_length  # noqa: E402


def _huffman_bits_per_symbol(data):
    freqs = {}
    for s in data:
        freqs[s] = freqs.get(s, 0) + 1
    code = huffman_code(freqs)
    return average_length(code, freqs)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Arithmetic coding vs Huffman vs entropy\n")
    print("  Arithmetic coding encodes the whole message as one number, so it is not limited to")
    print("  whole-bit codewords -- it hugs the entropy where Huffman must round up.\n")

    datasets = [
        ("very skewed", "a" * 90 + "b" * 7 + "c" * 3),
        ("skewed", "a" * 60 + "b" * 25 + "c" * 15),
        ("mild skew", "a" * 40 + "b" * 35 + "c" * 25),
        ("near uniform", ("abc" * 33 + "a")),
        ("uniform-4", "abcd" * 30),
        ("English-ish", "the quick brown fox jumps over the lazy dog " * 3),
    ]

    print(f"    {'distribution':>14} {'entropy':>8} {'arithmetic':>11} {'Huffman':>8} "
          f"{'AC saves':>9}")
    results = []
    for name, data in datasets:
        H = entropy(data)
        ac = encoded_bits_per_symbol(data)
        hu = _huffman_bits_per_symbol(data)
        results.append((name, H, ac, hu))
        # confirm lossless
        bits, model, total, length = encode(data)
        assert "".join(decode(bits, model, total, length)) == data
        save = (hu - ac) / hu * 100 if hu > 0 else 0
        print(f"    {name:>14} {H:>8.3f} {ac:>11.3f} {hu:>8.3f} {save:>8.1f}%")

    print("\n  The bigger the skew, the more Huffman's whole-bit codewords waste and the more")
    print("  arithmetic coding wins: at 90% one symbol, entropy is ~0.56 bits but Huffman is")
    print("  forced to spend at least 1 bit on the rarer symbols. All encodings round-trip exactly.")

    print("\n  How it works: start with the interval [0,1); each symbol narrows it to its")
    print("  probability-weighted sub-interval; after the whole message, any number in the final")
    print("  tiny interval names the message. Integer range coding with renormalization keeps it")
    print("  exact without unbounded-precision reals -- the entropy coder inside JPEG and H.264.")

    _svg(os.path.join(outdir, "arithmetic_coding.svg"), results)
    print(f"\n  wrote {os.path.join(outdir, 'arithmetic_coding.svg')}")


def _svg(path, results, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Bits per symbol: entropy (floor) vs arithmetic vs Huffman</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'arithmetic coding (blue) hugs the entropy (green line); Huffman (orange) rounds up, '
        f'wasting most on skewed data</text>',
    ]
    y0, y1 = height - 60, 75
    plot_h = y0 - y1
    maxbits = max(max(H, ac, hu) for _, H, ac, hu in results) * 1.1
    gw = (width - 90) / len(results)
    bw = gw * 0.22
    for i, (name, H, ac, hu) in enumerate(results):
        cx = 70 + i * gw
        for j, (val, col) in enumerate([(ac, "#4dabf7"), (hu, "#ff922b")]):
            h = val / maxbits * plot_h
            x = cx + j * (bw + 3)
            parts.append(f'<rect x="{x:.1f}" y="{y0 - h:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                         f'fill="{col}"/>')
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{y0 - h - 4:.1f}" fill="{col}" '
                         f'font-size="8" text-anchor="middle">{val:.2f}</text>')
        # entropy tick
        ey = y0 - H / maxbits * plot_h
        parts.append(f'<line x1="{cx - 4:.1f}" y1="{ey:.1f}" x2="{cx + 2 * bw + 7:.1f}" y2="{ey:.1f}" '
                     f'stroke="#06d6a0" stroke-width="2"/>')
        parts.append(f'<text x="{cx + bw:.1f}" y="{y0 + 14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{name}</text>')
    parts.append(f'<line x1="55" y1="{y0}" x2="{width-30}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<rect x="{width-200}" y="62" width="10" height="7" fill="#4dabf7"/>')
    parts.append(f'<text x="{width-186}" y="69" fill="#8b949e" font-size="10">arithmetic</text>')
    parts.append(f'<rect x="{width-120}" y="62" width="10" height="7" fill="#ff922b"/>')
    parts.append(f'<text x="{width-106}" y="69" fill="#8b949e" font-size="10">Huffman</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
