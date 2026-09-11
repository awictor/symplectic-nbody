"""Demo: cyclic redundancy check -- catching errors with polynomial division.

Shows the standard CRC check values, a frame being stamped and then verified (and a corruption
caught), and the error-detection coverage: what fraction of random corruptions each CRC width
catches, approaching 1 - 2^-r. Draws the frame layout and the miss-probability by width.

    python examples/crc_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crc import (crc_named, append_crc, check_frame, detects_error,  # noqa: E402
                 flip_bit, PARAMS)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("CRC: divide the message polynomial by a generator, send the remainder\n")
    check = b"123456789"
    print(f"  check string '123456789':")
    for name in ("CRC-8", "CRC-16-CCITT", "CRC-32"):
        w = PARAMS[name]["width"]
        print(f"    {name:>14} = 0x{crc_named(check, name):0{w // 4}X}")

    print("\n  Frame = data + CRC. Receiver recomputes and compares:")
    data = b"GET /index.html"
    frame = append_crc(data, "CRC-32")
    print(f"    data   : {data!r}")
    print(f"    frame  : ...{frame[-4:].hex()} (4 CRC bytes appended), check = "
          f"{'OK' if check_frame(frame, 'CRC-32') else 'FAIL'}")
    bad = bytearray(frame)
    bad[3] ^= 0x20  # flip a bit in the data
    print(f"    corrupt one bit -> check = {'OK' if check_frame(bytes(bad), 'CRC-32') else 'ERROR CAUGHT'}")

    print("\n  Error-detection coverage (random multi-bit corruptions):")
    print(f"  {'CRC':>14}{'width':>7}{'caught':>10}{'miss ~2^-r':>13}")
    for name in ("CRC-8", "CRC-16-CCITT", "CRC-32"):
        w = PARAMS[name]["width"]
        caught, total = _coverage(name, trials=4000)
        print(f"  {name:>14}{w:>7}{caught / total:>10.4f}{2.0 ** -w:>13.2e}")
    print("\n  Every single-bit error, every burst shorter than the width, and all but a")
    print("  ~2^-r fraction of random corruptions are caught -- for 32 check bits that is a")
    print("  miss rate of 1 in 4 billion, with nothing but shifts and XORs. The checksum on")
    print("  Ethernet frames, ZIP files, and PNG chunks.")

    _svg(os.path.join(outdir, "crc.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'crc.svg')}")


def _coverage(name, trials=4000):
    """Fraction of random multi-bit corruptions detected, using a seeded LCG."""
    data = bytes((7 * i + 13) & 0xFF for i in range(24))
    state = 12345
    caught = 0
    for _ in range(trials):
        b = bytearray(data)
        # flip 2-5 random bits (high bits of the LCG)
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        nflip = 2 + (state >> 16) % 4
        for _ in range(nflip):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            bit = (state >> 8) % (len(data) * 8)
            b[bit // 8] ^= 1 << (7 - bit % 8)
        if detects_error(data, bytes(b), name):
            caught += 1
    return caught, trials


def _svg(path, w=760, h=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'CRC: a checksum from polynomial division over GF(2)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'the wire frame layout (top); miss probability ~2^-r shrinks with CRC width (bottom)</text>',
    ]

    # top: frame layout -- data bytes then CRC bytes
    fx, fy = 40, 80
    bw, bh = 34, 34
    ndata, ncrc = 11, 4
    for i in range(ndata):
        x = fx + i * bw
        parts.append(f'<rect x="{x}" y="{fy}" width="{bw-2}" height="{bh}" fill="#161b22" '
                     f'stroke="#4dabf7" stroke-width="1.3"/>')
        parts.append(f'<text x="{x + bw/2 - 1:.1f}" y="{fy + bh/2 + 4:.1f}" fill="#4dabf7" '
                     f'font-size="9" text-anchor="middle">D</text>')
    for i in range(ncrc):
        x = fx + (ndata + i) * bw
        parts.append(f'<rect x="{x}" y="{fy}" width="{bw-2}" height="{bh}" fill="#161b22" '
                     f'stroke="#ff922b" stroke-width="1.3"/>')
        parts.append(f'<text x="{x + bw/2 - 1:.1f}" y="{fy + bh/2 + 4:.1f}" fill="#ff922b" '
                     f'font-size="9" text-anchor="middle">C</text>')
    parts.append(f'<text x="{fx + ndata*bw/2:.1f}" y="{fy + bh + 16:.1f}" fill="#4dabf7" '
                 f'font-size="10" text-anchor="middle">message data</text>')
    parts.append(f'<text x="{fx + (ndata + ncrc/2)*bw:.1f}" y="{fy + bh + 16:.1f}" fill="#ff922b" '
                 f'font-size="10" text-anchor="middle">CRC-32</text>')
    parts.append(f'<text x="{fx:.1f}" y="{fy + bh + 40:.1f}" fill="#8b949e" font-size="10">'
                 f'Receiver divides the whole frame by the generator: zero remainder = no error detected.</text>')

    # bottom: miss probability 2^-r vs width (log y)
    rx0, rx1 = 90, w - 40
    ry0, ry1 = h - 55, 190
    widths = list(range(4, 41, 2))
    miss = [2.0 ** -r for r in widths]
    ymin, ymax = math.log10(miss[-1]), math.log10(miss[0])

    def RX(r):
        return rx0 + (r - widths[0]) / (widths[-1] - widths[0]) * (rx1 - rx0)

    def RY(m):
        return ry0 - (math.log10(m) - ymin) / (ymax - ymin) * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = " ".join(f"{RX(r):.1f},{RY(m):.1f}" for r, m in zip(widths, miss))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    # mark the three standard widths
    for name, col in (("CRC-8", "#ffd43b"), ("CRC-16-CCITT", "#b197fc"), ("CRC-32", "#ff6b6b")):
        r = PARAMS[name]["width"]
        parts.append(f'<circle cx="{RX(r):.1f}" cy="{RY(2.0**-r):.1f}" r="3.5" fill="{col}"/>')
        parts.append(f'<text x="{RX(r):.1f}" y="{RY(2.0**-r)-7:.1f}" fill="{col}" font-size="9" '
                     f'text-anchor="middle">{name.split("-")[0]}-{r}</text>')
    for e in (0, -3, -6, -9, -12):
        yy = RY(10.0 ** e)
        parts.append(f'<text x="{rx0-6:.1f}" y="{yy+3:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="end">1e{e}</text>')
    for r in (8, 16, 32):
        parts.append(f'<text x="{RX(r):.1f}" y="{ry0+14:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{r}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">CRC width r (check bits)</text>')
    parts.append(f'<text x="{rx0+4:.1f}" y="{ry1-4:.1f}" fill="#8b949e" font-size="9">'
                 f'miss probability 2^-r</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
