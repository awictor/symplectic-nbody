"""Fibonacci coding demo: Zeckendorf-based codewords, length vs Elias, and self-synchronization after a bit error."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import fibonacci_coding as fc
import universal_codes as uc


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def main():
    lines = []
    lines.append("Fibonacci coding -- universal, error-resilient integer codes via Zeckendorf")
    lines.append("=" * 76)
    lines.append("")

    lines.append("Codewords (Zeckendorf bits, LSB-first, + '11' terminator):")
    lines.append("   n    Fibonacci code   ends-in-11")
    lines.append("   " + "-" * 36)
    for n in (1, 2, 3, 4, 5, 8, 12, 20, 50, 100):
        c = fc.encode_int(n)
        lines.append(f"   {n:3d}   {c:16s}   {'yes' if c.endswith('11') else 'NO'}")
    lines.append("")

    # length comparison vs Elias gamma / delta
    lines.append("Code length vs Elias (bits):")
    lines.append("   n        Fibonacci   Elias-gamma   Elias-delta")
    lines.append("   " + "-" * 48)
    for n in (1, 4, 16, 64, 256, 1024, 4096):
        lf = fc.code_length(n)
        lg = len(uc.gamma_encode(n))
        ld = len(uc.delta_encode(n))
        lines.append(f"   {n:5d}      {lf:6d}      {lg:8d}      {ld:8d}")
    lines.append("")
    lines.append("Fibonacci ~ 1.44 log2 n: competitive with gamma for small n, near delta for large.")
    lines.append("")

    # self-synchronization demo
    vals = list(range(1, 25))
    bits = fc.encode(vals)
    flip = 7
    recovered = fc.resync_after_error(bits, flip_index=flip)
    damaged = [i for i in range(min(len(recovered), len(vals))) if recovered[i] != vals[i]]
    lines.append(f"Self-synchronization: encode {len(vals)} values ({len(bits)} bits), flip bit {flip}.")
    lines.append(f"  Corrupted codewords (indices): {damaged}")
    lines.append(f"  Values after the damage decode correctly again -- the '11' terminator resynchronizes.")
    lines.append(f"  Last 8 decoded: {recovered[-8:]}")
    lines.append(f"  Last 8 truth:   {vals[-8:]}")

    text = "\n".join(lines)
    print(text)

    svg = _svg(vals, bits, flip, recovered)
    return text, svg


def _svg(vals, bits, flip, recovered):
    W, H = 640, 420
    P = PALETTE
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'Fibonacci code length vs Elias, and self-synchronization</text>')

    # TOP: length curves
    import universal_codes as uc
    ns = [2 ** k for k in range(1, 14)]
    lx0, lx1, ly0, ly1 = 55, 610, 55, 210

    def px(k):
        return lx0 + k / (len(ns) - 1) * (lx1 - lx0)

    fibL = [fc.code_length(n) for n in ns]
    gamL = [len(uc.gamma_encode(n)) for n in ns]
    delL = [len(uc.delta_encode(n)) for n in ns]
    lmax = max(max(fibL), max(gamL), max(delL))

    def py(v):
        return ly1 - v / lmax * (ly1 - ly0)

    def curve(vals_, color, w):
        pts = " ".join(f"{px(i):.1f},{py(vals_[i]):.1f}" for i in range(len(vals_)))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{w}"/>'

    parts.append(curve(gamL, P["gray"], 1.6))
    parts.append(curve(delL, P["purple"], 1.6))
    parts.append(curve(fibL, P["yellow"], 2.2))
    parts.append(f'<line x1="{lx0}" y1="{ly1}" x2="{lx1}" y2="{ly1}" stroke="{P["gray"]}" stroke-width="1"/>')
    parts.append(f'<text x="{lx0}" y="{ly0 - 4}" fill="{P["gray"]}" font-size="10">code length (bits) vs n = 2..8192</text>')
    parts.append(f'<text x="{lx0}" y="{ly1 + 16}" fill="{P["yellow"]}" font-size="10">Fibonacci</text>')
    parts.append(f'<text x="{lx0 + 90}" y="{ly1 + 16}" fill="{P["gray"]}" font-size="10">Elias-gamma</text>')
    parts.append(f'<text x="{lx0 + 210}" y="{ly1 + 16}" fill="{P["purple"]}" font-size="10">Elias-delta</text>')

    # BOTTOM: the bitstream, colored; flipped bit red, corrupted codewords shaded
    bx0 = 20
    by = 300
    bw = min(8.0, (W - 40) / len(bits))
    parts.append(f'<text x="20" y="{by - 14}" fill="{P["text"]}" font-size="12">'
                 f'bitstream ({len(bits)} bits): red = flipped bit, green = "11" terminators (resync points)</text>')
    # mark terminators: scan the ORIGINAL stream for codeword ends
    i = 0
    term_positions = set()
    while i < len(bits):
        v, ni = fc.decode_int(bits, i)
        term_positions.add(ni - 1)   # last bit of terminator
        term_positions.add(ni - 2)
        i = ni
    for k, b in enumerate(bits):
        x = bx0 + k * bw
        if k == flip:
            col = P["red"]
        elif k in term_positions:
            col = P["green"]
        else:
            col = P["blue"] if b == "1" else "#30363d"
        parts.append(f'<rect x="{x:.1f}" y="{by}" width="{max(bw - 0.5, 1):.1f}" height="16" fill="{col}"/>')

    parts.append(f'<text x="20" y="{H - 14}" fill="{P["gray"]}" font-size="11">'
                 f'a flipped bit garbles only its neighborhood; the next green "11" resynchronizes the '
                 f'decoder -- Huffman/Elias would cascade.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
