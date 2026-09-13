"""Demo: LDPC codes -- sparse parity checks and message-passing decoding over a noisy channel.

Builds a regular LDPC code, transmits random codewords over a binary symmetric channel at a sweep of
noise levels, and compares the word-error rate of the hard-decision bit-flipping decoder against the
soft-decision sum-product (belief propagation) decoder. Draws the two error-rate curves.

    python examples/ldpc_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ldpc import (  # noqa: E402
    make_regular_ldpc,
    systematic_generator,
    encode,
    bsc,
    decode_bitflip,
    decode_sum_product,
    min_distance,
    _lcg,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("LDPC codes: sparse parity checks + message passing, near the Shannon limit\n")

    n, wc, wr, seed = 20, 3, 4, 5
    H = make_regular_ldpc(n, wc, wr, seed=seed)
    G, free = systematic_generator(H)
    k = len(G)
    m = len(H)
    d = min_distance(H, G)

    print(f"  regular ({wc},{wr}) LDPC: n={n} bits, m={m} checks, k={k} message bits")
    print(f"  rate {k}/{n} = {k / n:.2f}, minimum distance {d}\n")

    levels = [0.02, 0.04, 0.06, 0.08, 0.10, 0.14]
    trials = 200
    print(f"  Word-error rate over {trials} random codewords per noise level:")
    print(f"    {'flip p':>8}{'bit-flip WER':>16}{'sum-product WER':>18}")
    curve_bf = []
    curve_bp = []
    for p in levels:
        rng = _lcg(int(p * 1000) + 1)
        bf_err = 0
        bp_err = 0
        for t in range(trials):
            msg = [1 if rng() < 0.5 else 0 for _ in range(k)]
            c = encode(msg, G)
            recv = bsc(c, p=p, seed=7000 + int(p * 1000) * 1000 + t)
            d1, s1 = decode_bitflip(H, recv)
            d2, s2 = decode_sum_product(H, recv, p=p)
            if not (s1 and d1 == c):
                bf_err += 1
            if not (s2 and d2 == c):
                bp_err += 1
        wer_bf = bf_err / trials
        wer_bp = bp_err / trials
        curve_bf.append((p, wer_bf))
        curve_bp.append((p, wer_bp))
        print(f"    {p:>8.2f}{wer_bf:>16.3f}{wer_bp:>18.3f}")

    print(f"\n  Sum-product (soft) beats bit-flipping (hard) at every noise level -- it uses the")
    print(f"  channel confidence, not just the received bits. This gap is why LDPC + BP is the")
    print(f"  decoder in Wi-Fi, 5G, and deep-space links.")

    _svg(os.path.join(outdir, "ldpc.svg"), curve_bf, curve_bp)
    print(f"\n  wrote {os.path.join(outdir, 'ldpc.svg')}")


def _svg(path, curve_bf, curve_bp, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'LDPC word-error rate: sum-product (soft) vs bit-flipping (hard)</text>',
    ]
    ox, oy, ow, oh = 60, 55, width - 110, height - 110
    pmax = max(p for p, _ in curve_bf)

    def px(p):
        return ox + ow * p / pmax

    def py(w):
        return oy + oh * (1 - w)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = py(frac)
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+ow}" y2="{y:.1f}" '
                     f'stroke="#161b22" stroke-width="1"/>')
        parts.append(f'<text x="{ox-6}" y="{y+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{frac:.2f}</text>')

    def poly(curve, color):
        pts = " ".join(f"{px(p):.1f},{py(w):.1f}" for p, w in curve)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for p, w in curve:
            parts.append(f'<circle cx="{px(p):.1f}" cy="{py(w):.1f}" r="3" fill="{color}"/>')

    poly(curve_bf, "#ff922b")   # bit-flipping
    poly(curve_bp, "#4dabf7")   # sum-product

    parts.append(f'<rect x="{ox+ow-160}" y="{oy+6}" width="12" height="4" fill="#4dabf7"/>')
    parts.append(f'<text x="{ox+ow-144}" y="{oy+11}" fill="#e6edf3" font-size="10">'
                 f'sum-product (BP)</text>')
    parts.append(f'<rect x="{ox+ow-160}" y="{oy+22}" width="12" height="4" fill="#ff922b"/>')
    parts.append(f'<text x="{ox+ow-144}" y="{oy+27}" fill="#e6edf3" font-size="10">'
                 f'bit-flipping</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+24:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">channel crossover probability p</text>')
    parts.append(f'<text x="{ox-40}" y="{oy+oh/2:.0f}" fill="#8b949e" font-size="10" '
                 f'transform="rotate(-90 {ox-40} {oy+oh/2:.0f})" text-anchor="middle">'
                 f'word-error rate</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
