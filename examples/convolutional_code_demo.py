"""Demo: convolutional coding and Viterbi decoding over a noisy channel.

Encodes a message with the classic (7,5) code, corrupts it with random bit flips, and shows the
Viterbi decoder recovering the original. Sweeps the channel error rate to draw the decode-success
curve -- the error-correction coding gain.

    python examples/convolutional_code_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from convolutional_code import ConvolutionalCode, hamming_distance  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Convolutional coding + Viterbi decoding over a noisy channel\n")

    code = ConvolutionalCode([0b111, 0b101])       # classic (7,5) rate-1/2, constraint length 3
    print(f"  rate-1/{code.n} code, constraint length {code.constraint}, "
          f"{code.n_states} trellis states")

    msg = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    enc = code.encode(msg)
    print(f"\n  message ({len(msg)} bits): {''.join(map(str, msg))}")
    print(f"  encoded ({len(enc)} bits): {''.join(map(str, enc))}")

    # corrupt 2 bits
    state = 42

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    noisy = enc[:]
    flips = [3, 11]
    for p in flips:
        noisy[p] ^= 1
    print(f"  received (2 bit errors at {flips}):")
    print(f"           {''.join(map(str, noisy))}")
    decoded = code.decode(noisy)
    print(f"  decoded  ({len(decoded)} bits): {''.join(map(str, decoded))}")
    print(f"  recovered the original exactly: {decoded == msg}")

    # coding gain: decode-success vs channel error rate, coded vs uncoded
    print("\n  Decode success vs channel bit-error rate (200 messages of 16 bits each):")
    print(f"    {'error rate':>10}  {'coded':>7}  {'uncoded':>8}")
    results = []
    for p in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        coded_ok = 0
        uncoded_ok = 0
        trials = 200
        for _ in range(trials):
            m = [1 if rng() < 0.5 else 0 for _ in range(16)]
            e = code.encode(m)
            recv = [b ^ (1 if rng() < p else 0) for b in e]
            if code.decode(recv) == m:
                coded_ok += 1
            # uncoded: transmit the raw 16 bits, any flip is an uncorrected error
            raw = [b ^ (1 if rng() < p else 0) for b in m]
            if raw == m:
                uncoded_ok += 1
        results.append((p, coded_ok / trials, uncoded_ok / trials))
        print(f"    {p:>10.2f}  {coded_ok/trials:>7.2f}  {uncoded_ok/trials:>8.2f}")

    print("\n  The encoder runs the message through a shift register, emitting XOR combinations that")
    print("  spread each bit across several outputs. Viterbi finds the maximum-likelihood path")
    print("  through the trellis -- the transmitted sequence closest to what arrived -- in linear time.")

    _svg(os.path.join(outdir, "convolutional_code.svg"), results)
    print(f"\n  wrote {os.path.join(outdir, 'convolutional_code.svg')}")


def _svg(path, results, width=760, height=420):
    m_left, m_bot, m_top, m_right = 60, 60, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    pmax = max(r[0] for r in results)

    def px(p):
        return m_left + p / pmax * pw

    def py(v):
        return m_top + ph - v * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Convolutional coding gain: decode success vs channel error rate</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'green = coded + Viterbi (errors corrected), red = uncoded (any flip corrupts the message)</text>',
    ]
    parts.append(f'<line x1="{m_left}" y1="{m_top+ph}" x2="{m_left+pw}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    parts.append(f'<line x1="{m_left}" y1="{m_top}" x2="{m_left}" y2="{m_top+ph}" '
                 f'stroke="#484f58" stroke-width="1.5"/>')
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<text x="{m_left-8}" y="{py(frac)+4:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{frac:.2f}</text>')
    parts.append(f'<text x="{m_left+pw/2:.0f}" y="{height-12}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">channel bit-error rate</text>')

    coded = " ".join(f"{px(r[0]):.1f},{py(r[1]):.1f}" for r in results)
    uncoded = " ".join(f"{px(r[0]):.1f},{py(r[2]):.1f}" for r in results)
    parts.append(f'<polyline points="{uncoded}" fill="none" stroke="#ff6b6b" stroke-width="2"/>')
    parts.append(f'<polyline points="{coded}" fill="none" stroke="#06d6a0" stroke-width="2.2"/>')
    for r in results:
        parts.append(f'<circle cx="{px(r[0]):.1f}" cy="{py(r[1]):.1f}" r="4" fill="#06d6a0"/>')
        parts.append(f'<circle cx="{px(r[0]):.1f}" cy="{py(r[2]):.1f}" r="3" fill="#ff6b6b"/>')
        parts.append(f'<text x="{px(r[0]):.0f}" y="{m_top+ph+18:.0f}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{r[0]:.2f}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
