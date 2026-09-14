"""Hadamard code demo: decode a message from a heavily corrupted word via one Walsh-Hadamard transform (SVG)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hadamard_code as HC
from walsh_hadamard import fwht


BG = "#0d1117"
TEXT = "#e6edf3"
GRAY = "#8b949e"
BLUE = "#4dabf7"
GREEN = "#06d6a0"
RED = "#ff6b6b"
YELLOW = "#ffd43b"


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main(outdir=None):
    m = 6
    n = 1 << m
    msg = 42
    cw = HC.encode(msg, m)
    t = HC.correctable_errors(m)

    # corrupt with t errors (near the limit)
    rnd = _lcg(20260913)
    ne = t
    pos = set()
    while len(pos) < ne:
        pos.add(int(rnd() * n) % n)
    received = list(cw)
    for p in pos:
        received[p] ^= 1

    decoded, peak = HC.decode_with_confidence(received, m)

    lines = []
    lines.append("Hadamard code: max-distance coding, decoded by one transform")
    lines.append("=" * 60)
    lines.append(f"parameters: m={m}, codeword length n=2^{m}={n}, minimum distance n/2={HC.min_distance(m)}")
    lines.append(f"correction radius: {t} errors (nearly n/4)")
    lines.append(f"rate: {m+1}/{n} info bits per transmitted bit (very low -- pays for robustness)")
    lines.append("")
    lines.append(f"sent message:      {msg}")
    lines.append(f"errors injected:   {ne} (of {n} bits flipped)")
    lines.append(f"decoded message:   {decoded}  ({'correct' if decoded == msg else 'WRONG'})")
    lines.append(f"correlation peak:  {peak} of {n}  (= n - 2*errors = {n - 2*ne})")
    lines.append("")
    lines.append("Decoding is a single fast Walsh-Hadamard transform: it correlates the received")
    lines.append("word against ALL 2^m codewords at once, and the tallest spike names the message.")
    lines.append("This is the code that returned the Mariner 9 photographs from Mars.")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # SVG: the FWHT spectrum with the decoded spike highlighted
        signs = [1 if b == 0 else -1 for b in received]
        spectrum = fwht(signs)
        W, H = 720, 360
        ml, mt, w, h = 50, 55, 620, 250

        def sx(i):
            return ml + i / (n - 1) * w

        def sy(v):
            return mt + h / 2 - v / n * (h / 2)

        s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
        s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
        s.append(f'<text x="{ml}" y="30" fill="{TEXT}" font-size="15">'
                 f'Walsh-Hadamard spectrum of a corrupted word: the spike is the message</text>')
        s.append(f'<line x1="{ml}" y1="{sy(0):.1f}" x2="{ml+w}" y2="{sy(0):.1f}" stroke="{GRAY}"/>')
        for i in range(n):
            col = RED if i == decoded else BLUE
            wdt = 3 if i == decoded else 1.5
            s.append(f'<line x1="{sx(i):.1f}" y1="{sy(0):.1f}" x2="{sx(i):.1f}" '
                     f'y2="{sy(spectrum[i]):.1f}" stroke="{col}" stroke-width="{wdt}"/>')
        # mark the peak
        s.append(f'<text x="{sx(decoded):.1f}" y="{sy(spectrum[decoded])-6:.1f}" fill="{RED}" '
                 f'font-size="11" text-anchor="middle">message {decoded}</text>')
        s.append(f'<text x="{ml}" y="{H-14}" fill="{GRAY}" font-size="10">'
                 f'Every other Walsh coefficient is small noise; the message coefficient towers over '
                 f'them (height n - 2*errors), so max-likelihood decoding is just argmax.</text>')
        s.append("</svg>")
        with open(os.path.join(outdir, "hadamard_code.svg"), "w", encoding="utf-8") as fh:
            fh.write("".join(s))

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
