"""Demo: Shannon entropy and Huffman coding.

Prints the Huffman code and its average length against the entropy bound for a skewed source,
then draws the binary-source entropy curve (maximal at a fair coin) and, for a sample source,
the per-symbol codeword lengths with the entropy limit and the H <= L < H+1 band.

    python examples/shannon_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shannon import (entropy, huffman_code, average_length, efficiency,  # noqa: E402
                     kraft_sum, encode, decode)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # English-like letter frequencies (a small skewed source)
    freqs = {"e": 0.27, "t": 0.20, "a": 0.16, "o": 0.13, "n": 0.12, "s": 0.07, "z": 0.05}
    code = huffman_code(freqs)
    H = entropy(list(freqs.values()))
    L = average_length(code, freqs)

    print("Shannon entropy & Huffman coding: compressing a skewed source\n")
    print(f"  {'symbol':>7}{'prob':>8}{'codeword':>12}{'bits':>6}")
    for sym in sorted(freqs, key=lambda k: -freqs[k]):
        print(f"  {sym:>7}{freqs[sym]:>8.2f}{code[sym]:>12}{len(code[sym]):>6}")
    print(f"\n  entropy  H = {H:.4f} bits/symbol")
    print(f"  Huffman  L = {L:.4f} bits/symbol   (fixed-length would need 3)")
    print(f"  bound: H <= L < H+1  ->  {H:.4f} <= {L:.4f} < {H + 1:.4f}   OK")
    print(f"  efficiency H/L = {efficiency(code, freqs):.1%},  Kraft sum = {kraft_sum(code):.3f}")

    msg = list("tenants")
    bits = encode(msg, code)
    print(f"\n  'tenants' -> {bits}  ({len(bits)} bits vs {8*len(msg)} raw), decodes back: "
          f"{'ok' if decode(bits, code) == msg else 'FAIL'}")
    print("  No lossless code can beat the entropy; Huffman lands within one bit of it and is")
    print("  optimal among prefix codes -- the reason ZIP, JPEG, and MP3 all carry a Huffman stage.")

    _svg(os.path.join(outdir, "shannon.svg"), freqs, code, H, L)
    print(f"\n  wrote {os.path.join(outdir, 'shannon.svg')}")


def _svg(path, freqs, code, H, L, w=760, h=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Shannon entropy &amp; Huffman coding</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'binary-source entropy, maximal at a fair coin (left); '
        f'Huffman codeword lengths vs the entropy limit (right)</text>',
    ]

    # left: binary entropy H(p) = -p log p - (1-p) log(1-p)
    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 55, 62

    def LX(p):
        return lx0 + p * (lx1 - lx0)

    def LY(hv):
        return ly0 - hv * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = []
    for i in range(1, 200):
        p = i / 200
        pts.append(f"{LX(p):.1f},{LY(entropy([p, 1 - p])):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    # mark the peak at p=1/2, H=1
    parts.append(f'<circle cx="{LX(0.5):.1f}" cy="{LY(1.0):.1f}" r="3.5" fill="#ffd43b"/>')
    parts.append(f'<text x="{LX(0.5):.1f}" y="{LY(1.0)-8:.1f}" fill="#ffd43b" font-size="9" '
                 f'text-anchor="middle">fair coin: 1 bit</text>')
    for v in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{lx0-6:.1f}" y="{LY(v)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{v:.1f}</text>')
    for p in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{LX(p):.1f}" y="{ly0+15:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{p:.1f}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">P(symbol = 1)</text>')
    parts.append(f'<text x="{lx0+4:.1f}" y="{ly1-2:.1f}" fill="#8b949e" font-size="9">H (bits)</text>')

    # right: codeword lengths per symbol with H and L reference lines
    rx0, rx1 = w // 2 + 45, w - 25
    ry0, ry1 = h - 55, 62
    syms = sorted(freqs, key=lambda k: -freqs[k])
    lens = [len(code[s]) for s in syms]
    lmax = max(max(lens), int(H) + 2)

    def RY(v):
        return ry0 - v / lmax * (ry0 - ry1)

    slot = (rx1 - rx0) / len(syms)
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    for i, sym in enumerate(syms):
        cx = rx0 + (i + 0.5) * slot
        bw = slot * 0.6
        parts.append(f'<rect x="{cx-bw/2:.1f}" y="{RY(lens[i]):.1f}" width="{bw:.1f}" '
                     f'height="{ry0-RY(lens[i]):.1f}" fill="#8338ec"/>')
        parts.append(f'<text x="{cx:.1f}" y="{ry0+14:.1f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{sym}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{RY(lens[i])-4:.1f}" fill="#b197fc" font-size="9" '
                     f'text-anchor="middle">{code[sym]}</text>')
    # entropy and average-length reference lines
    parts.append(f'<line x1="{rx0}" y1="{RY(H):.1f}" x2="{rx1}" y2="{RY(H):.1f}" '
                 f'stroke="#06d6a0" stroke-width="1.5" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(H)-4:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">H = {H:.2f}</text>')
    parts.append(f'<line x1="{rx0}" y1="{RY(L):.1f}" x2="{rx1}" y2="{RY(L):.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.5" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(L)+11:.1f}" fill="#ffd43b" font-size="9" '
                 f'text-anchor="end">L = {L:.2f}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">symbol (by frequency) -> codeword length</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
