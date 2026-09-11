"""Demo: Hamming codes -- correcting a bit flip from the syndrome.

Prints a (7,4) encode, then flips each bit in turn and shows the syndrome reading out the flipped
position so it can be corrected, and lists the code rate as the block grows. Draws the parity-bit
coverage of the (7,4) code and how its rate rises toward 1 with block size.

    python examples/hamming_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hamming import (encode, syndrome, decode, code_parameters,  # noqa: E402
                     code_rate, _parity_positions)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    data = [1, 0, 1, 1]
    cw = encode(data, 3)
    print("Hamming(7,4): 4 data bits -> 7 transmitted bits, corrects any single error\n")
    print(f"  data {data} -> codeword {cw}   (parity bits at positions 1,2,4)\n")
    print(f"  {'flipped pos':>12}{'received':>20}{'syndrome':>10}{'corrected?':>12}")
    for i in range(7):
        bad = cw[:]
        bad[i] ^= 1
        syn = syndrome(bad, 3)
        dec, err = decode(bad, 3)
        print(f"  {i + 1:>12}{str(bad):>20}{syn:>10}{('yes' if dec == data else 'NO'):>12}")

    print("\n  The failed parity checks spell out, in binary, the exact position of the flipped")
    print("  bit -- flip it back and the message is restored. No retransmission needed.\n")
    print(f"  {'code':>10}{'data k':>8}{'total n':>9}{'rate k/n':>10}")
    for m in (2, 3, 4, 5, 6):
        n, k = code_parameters(m)
        print(f"  ({n},{k})".rjust(10) + f"{k:>8}{n:>9}{code_rate(m):>10.3f}")
    print("\n  More parity bits per block -> higher rate: the (255,247) code spends just 8 bits")
    print("  to protect 247. Add one overall parity bit for SECDED -- the scheme in ECC memory.")

    _svg(os.path.join(outdir, "hamming.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'hamming.svg')}")


def _svg(path, w=760, h=390):
    import math

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Hamming(7,4): parity coverage and the code rate</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'which positions each parity bit checks (left); code rate k/n vs block size (right)</text>',
    ]

    # left: 7 positions x 3 parity checks grid; a cell is filled if position is covered by p
    lx0, ly0 = 70, 90
    cell = 42
    parity = _parity_positions(3)  # [1, 2, 4]
    parity_set = set(parity)
    # column headers: positions 1..7, marking parity vs data
    for pos in range(1, 8):
        x = lx0 + (pos - 1) * cell
        is_par = pos in parity_set
        col = "#ffd43b" if is_par else "#4dabf7"
        parts.append(f'<text x="{x + cell/2:.1f}" y="{ly0-8:.1f}" fill="{col}" font-size="11" '
                     f'text-anchor="middle">{pos}</text>')
        parts.append(f'<text x="{x + cell/2:.1f}" y="{ly0-22:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{"P" if is_par else "D"}</text>')
    # rows: each parity bit p, fill cells where pos & p
    for r, p in enumerate(parity):
        y = ly0 + r * cell
        parts.append(f'<text x="{lx0-12:.1f}" y="{y + cell/2 + 4:.1f}" fill="#ffd43b" '
                     f'font-size="11" text-anchor="end">p{p}</text>')
        for pos in range(1, 8):
            x = lx0 + (pos - 1) * cell
            covered = bool(pos & p)
            fill = "#8338ec" if covered else "#161b22"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-3:.1f}" height="{cell-3:.1f}" '
                         f'fill="{fill}" stroke="#21262d" stroke-width="1"/>')
            if covered:
                parts.append(f'<text x="{x+cell/2-1.5:.1f}" y="{y+cell/2+4:.1f}" fill="#e6edf3" '
                             f'font-size="10" text-anchor="middle">check</text>')
    parts.append(f'<text x="{lx0:.1f}" y="{ly0 + 3*cell + 24:.1f}" fill="#8b949e" font-size="10">'
                 f'A flipped bit fails exactly the checks whose bit is set in its position number,</text>')
    parts.append(f'<text x="{lx0:.1f}" y="{ly0 + 3*cell + 38:.1f}" fill="#8b949e" font-size="10">'
                 f'so the syndrome (p4 p2 p1 read as binary) IS the error position.</text>')

    # right: code rate vs block size
    rx0, rx1 = w // 2 + 70, w - 30
    ry0, ry1 = h - 60, 90
    ms = list(range(2, 9))
    ns = [code_parameters(m)[0] for m in ms]
    rates = [code_rate(m) for m in ms]
    lo, hi = math.log(ns[0]), math.log(ns[-1])

    def RX(nn):
        return rx0 + (math.log(nn) - lo) / (hi - lo) * (rx1 - rx0)

    def RY(rt):
        return ry0 - rt * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{RY(1.0):.1f}" x2="{rx1}" y2="{RY(1.0):.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(1.0)-4:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">rate -> 1</text>')
    pts = " ".join(f"{RX(nn):.1f},{RY(rt):.1f}" for nn, rt in zip(ns, rates))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    for m, nn, rt in zip(ms, ns, rates):
        parts.append(f'<circle cx="{RX(nn):.1f}" cy="{RY(rt):.1f}" r="2.5" fill="#06d6a0"/>')
    # label the (7,4) point
    n7 = code_parameters(3)[0]
    parts.append(f'<text x="{RX(n7):.1f}" y="{RY(code_rate(3))+16:.1f}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="middle">(7,4)</text>')
    for rt in (0.5, 0.75, 1.0):
        parts.append(f'<text x="{rx0-6:.1f}" y="{RY(rt)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{rt:.2f}</text>')
    for nn in (ns[0], ns[3], ns[-1]):
        parts.append(f'<text x="{RX(nn):.1f}" y="{ry0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{nn}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">block length n (log)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
