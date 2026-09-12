"""Demo: LZW dictionary compression building its codebook on the fly.

Compresses text and shows the dictionary learning repeated substrings (so codes get longer and the
ratio improves), round-trips several inputs, and compares the ratio on repetitive vs random data.
Draws the compression ratio versus how repetitive the input is.

    python examples/lzw_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lzw import compress, decompress, compression_ratio  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("LZW: adaptive dictionary compression (GIF / Unix compress)\n")

    text = b"TOBEORNOTTOBEORTOBEORNOT"
    codes = compress(text)
    print(f"  input : {text.decode()}  ({len(text)} bytes)")
    print(f"  codes : {codes}  ({len(codes)} codes)")
    print(f"  decompresses exactly: {decompress(codes) == text}")
    print("  Codes > 255 are dictionary entries the encoder built from repeated substrings;")
    print("  the decoder rebuilds the identical dictionary with no side channel.\n")

    # ratio improves as repetition grows
    print("  Compression ratio (codes / bytes) vs repetition:")
    base = b"the quick brown fox "
    results = []
    for reps in [1, 2, 5, 10, 25, 50, 100]:
        data = base * reps
        r = compression_ratio(data)
        results.append((reps, r))
        print(f"    {reps:3d} repetitions ({len(data):5d} bytes): ratio {r:.3f}")

    # repetitive vs random
    state = 7

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    rep = b"ABCABCABC" * 300
    rnd = bytes(int(rng() * 256) for _ in range(2700))
    print(f"\n  Repetitive 2700 bytes: ratio {compression_ratio(rep):.3f} (great)")
    print(f"  Random     2700 bytes: ratio {compression_ratio(rnd):.3f} (near 1 -- incompressible)")

    # dictionary growth: how many entries get built
    def dict_entries(data):
        d = {bytes([i]): i for i in range(256)}
        nc = 256
        w = b""
        for byte in data:
            c = bytes([byte])
            if w + c in d:
                w = w + c
            else:
                d[w + c] = nc
                nc += 1
                w = c
        return nc - 256
    print(f"\n  Dictionary entries learned from '{base.decode().strip()}' x50: "
          f"{dict_entries(base * 50)}")

    print("\n  LZW replaces repeated substrings with single codes, and because both sides build the")
    print("  same dictionary as they read, the compressed stream is just integer codes -- no")
    print("  dictionary transmitted, one pass, and the ratio improves the longer the patterns run.")

    _svg(os.path.join(outdir, "lzw.svg"), results)
    print(f"\n  wrote {os.path.join(outdir, 'lzw.svg')}")


def _svg(path, results, width=760, height=420):
    import math
    m_left, m_bot, m_top, m_right = 70, 60, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    xs = [math.log10(r[0]) for r in results]
    xmin, xmax = min(xs), max(xs)
    ymax = 1.0

    def px(lx):
        return m_left + (lx - xmin) / (xmax - xmin) * pw

    def py(v):
        return m_top + ph - v / ymax * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'LZW compression ratio improves with repetition</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'codes/bytes falls as the same phrase repeats -- the dictionary learns longer substrings</text>',
    ]
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    # ratio=1 reference
    parts.append(f'<line x1="{m_left}" y1="{py(1.0):.1f}" x2="{m_left+pw}" y2="{py(1.0):.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{m_left+pw-4}" y="{py(1.0)-5:.1f}" fill="#ff6b6b" font-size="10" '
                 f'text-anchor="end">ratio = 1 (no compression)</text>')

    pts = " ".join(f"{px(lx):.1f},{py(r[1]):.1f}" for lx, r in zip(xs, results))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2.2"/>')
    for lx, r in zip(xs, results):
        parts.append(f'<circle cx="{px(lx):.1f}" cy="{py(r[1]):.1f}" r="4" fill="#06d6a0"/>')
        parts.append(f'<text x="{px(lx):.0f}" y="{m_top+ph+18:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{r[0]}x</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">number of phrase repetitions (log scale)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
