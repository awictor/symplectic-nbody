"""Demo: LZ77 compression -- pointing back at what you have already seen.

Compresses a repetitive string and shows its token stream (literals and back-references), then
how the compression ratio climbs with repetition while incompressible data stays near 1 --
Shannon's entropy limit showing through.

    python examples/lz77_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lz77 import (compress, decompress, compression_ratio,  # noqa: E402
                  count_tokens, token_cost)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    text = b"the cat sat on the mat, the cat sat on the hat"
    tokens = compress(text)
    print("LZ77: replace repeats with (distance, length) back-references\n")
    print(f"  input ({len(text)} bytes): {text.decode()}\n")
    print("  token stream:")
    line = "   "
    for t in tokens:
        if t[0] == "lit":
            line += chr(t[1])
        else:
            _, dist, length, nxt = t
            tail = chr(nxt) if nxt is not None else ""
            line += f"[<-{dist},{length}]{tail}"
    print(line)
    lits, copies = count_tokens(tokens)
    print(f"\n  {lits} literals + {copies} back-references, round-trip: "
          f"{'OK' if decompress(tokens) == text else 'FAIL'}")
    print(f"  compression ratio ~ {compression_ratio(text, tokens):.2f}x\n")

    print("  Ratio climbs with repetition, but incompressible data stays ~1 (Shannon's limit):")
    print(f"  {'data':>26}{'bytes':>7}{'ratio':>8}")
    samples = [
        ("one 'ab' pair", b"ab"),
        ("'ab' x 5", b"ab" * 5),
        ("'ab' x 50", b"ab" * 50),
        ("English text x 20", b"the quick brown fox " * 20),
        ("pseudo-random bytes", bytes(((i * 2654435761) >> 8) & 0xFF for i in range(400))),
    ]
    for label, d in samples:
        print(f"  {label:>26}{len(d):>7}{compression_ratio(d):>8.2f}")
    print("\n  LZ77 + Huffman together are DEFLATE -- the algorithm inside gzip, ZIP, and PNG.")

    _svg(os.path.join(outdir, "lz77.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'lz77.svg')}")


def _svg(path, w=760, h=390):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'LZ77: back-references replace repeats</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the token stream, literals vs copies (top); compression ratio vs repetition (bottom)</text>',
    ]

    # top: token stream for a repetitive string, colouring literals vs copies
    text = b"abcabcabcabcXYZabcabc"
    tokens = compress(text)
    tx, ty = 40, 80
    cw = 26
    x = tx
    for t in tokens:
        if t[0] == "lit":
            parts.append(f'<rect x="{x}" y="{ty}" width="{cw-3}" height="26" fill="#161b22" '
                         f'stroke="#4dabf7" stroke-width="1.2"/>')
            parts.append(f'<text x="{x + (cw-3)/2:.1f}" y="{ty+17:.1f}" fill="#4dabf7" '
                         f'font-size="11" text-anchor="middle">{chr(t[1])}</text>')
            x += cw
        else:
            _, dist, length, nxt = t
            wd = cw * 2.4
            parts.append(f'<rect x="{x}" y="{ty}" width="{wd-3:.1f}" height="26" fill="#161b22" '
                         f'stroke="#ff922b" stroke-width="1.2"/>')
            parts.append(f'<text x="{x + wd/2:.1f}" y="{ty+17:.1f}" fill="#ff922b" '
                         f'font-size="9" text-anchor="middle">&lt;-{dist},{length}</text>')
            x += wd
    parts.append(f'<text x="{tx}" y="{ty+48:.1f}" fill="#8b949e" font-size="10">'
                 f'"{text.decode()}"  ->  each orange box copies bytes already sent</text>')
    parts.append(f'<rect x="{tx}" y="{ty+58}" width="10" height="10" fill="#161b22" stroke="#4dabf7"/>'
                 f'<text x="{tx+15}" y="{ty+67}" fill="#e6edf3" font-size="9">literal byte</text>')
    parts.append(f'<rect x="{tx+110}" y="{ty+58}" width="10" height="10" fill="#161b22" stroke="#ff922b"/>'
                 f'<text x="{tx+125}" y="{ty+67}" fill="#e6edf3" font-size="9">back-reference (distance, length)</text>')

    # bottom: compression ratio vs number of repetitions of a fixed block
    rx0, rx1 = 80, w - 40
    ry0, ry1 = h - 55, 210
    reps = list(range(1, 61))
    block = b"the quick brown fox "
    ratios = [compression_ratio(block * r) for r in reps]
    rmax = max(ratios) * 1.1

    def RX(r):
        return rx0 + (r - reps[0]) / (reps[-1] - reps[0]) * (rx1 - rx0)

    def RY(v):
        return ry0 - v / rmax * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    # ratio = 1 reference (no compression)
    parts.append(f'<line x1="{rx0}" y1="{RY(1.0):.1f}" x2="{rx1}" y2="{RY(1.0):.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(1.0)-4:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">ratio 1 (no gain)</text>')
    pts = " ".join(f"{RX(r):.1f},{RY(v):.1f}" for r, v in zip(reps, ratios))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    for r in (1, 20, 40, 60):
        parts.append(f'<text x="{RX(r):.1f}" y="{ry0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{r}</text>')
    for v in (1, int(rmax // 2), int(rmax)):
        if v > 0:
            parts.append(f'<text x="{rx0-6:.1f}" y="{RY(v)+3:.1f}" fill="#8b949e" font-size="8" '
                         f'text-anchor="end">{v}x</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">number of repeated blocks -> compression ratio</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
