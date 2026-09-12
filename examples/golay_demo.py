"""Demo: the extended binary Golay code -- correcting 3 errors in 24 bits, the Voyager code.

Encodes a message, blasts it with up to 3 bit errors, and watches the decoder recover it perfectly;
shows the code's perfect weight distribution; and draws a codeword with its flipped bits and the
correction.

    python examples/golay_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from golay import (encode, decode, corrupt, weight_enumerator, minimum_distance,  # noqa: E402
                   _popcount, _LCG)


def bits(word, n=24):
    return [(word >> i) & 1 for i in range(n)]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Golay [24,12,8]: correct any 3 bit errors -- the code that imaged Jupiter and Saturn\n")

    msg = 0b101011001101
    cw = encode(msg)
    print(f"  message  (12 bits): {msg:012b}")
    print(f"  codeword (24 bits): {cw:024b}\n")

    rng = _LCG(2024)
    print("  transmit through a noisy channel and decode:")
    for ne in range(5):
        recv = corrupt(cw, ne, rng)
        dmsg, nc = decode(recv)
        flipped = _popcount(recv ^ cw)
        if dmsg is None:
            status = f"UNCORRECTABLE ({flipped} errors detected)"
        elif dmsg == msg:
            status = f"recovered exactly, fixed {nc} error(s)"
        else:
            status = "MISCORRECTED"
        print(f"    {ne} bit errors: {status}")
    print()

    print(f"  minimum distance: {minimum_distance()} (corrects floor((8-1)/2) = 3 errors)")
    print(f"  weight enumerator: {weight_enumerator()}")
    print("    the counts 1, 759, 2576, 759, 1 are the signature of the Golay code -- its 4096")
    print("    codewords sit at weights 0, 8, 12, 16, 24 in a perfectly symmetric distribution\n")

    print("  Encode by G = [I | B] over GF(2); decode by the syndrome, which for a distance-8 code")
    print("  maps every <=3-error pattern to a unique coset leader. Voyager 1 and 2 used it to send")
    print("  colour pictures of the outer planets across billions of kilometres of noisy space.")

    _svg(os.path.join(outdir, "golay.svg"), cw, corrupt(cw, 3, _LCG(5)))
    print(f"\n  wrote {os.path.join(outdir, 'golay.svg')}")


def _svg(path, codeword, received, width=760, height=320):
    cw_bits = bits(codeword)
    rx_bits = bits(received)
    dmsg, nc = decode(received)
    corrected = encode(dmsg) if dmsg is not None else codeword

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="16">'
        f'Golay code correcting 3 bit errors in a 24-bit codeword</text>',
    ]
    cell = (width - 80) / 24

    def row(label, bitvals, y, highlight=None):
        out = [f'<text x="20" y="{y+15:.0f}" fill="#8b949e" font-size="11">{label}</text>']
        for i in range(24):
            x = 60 + i * cell
            v = bitvals[i]
            flip = highlight and highlight[i]
            fill = "#ff6b6b" if flip else ("#4dabf7" if v else "#161b22")
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-2:.1f}" height="24" rx="2" '
                       f'fill="{fill}" stroke="#30363d" stroke-width="0.5"/>')
            out.append(f'<text x="{x+(cell-2)/2:.1f}" y="{y+16:.0f}" fill="#e6edf3" font-size="10" '
                       f'text-anchor="middle">{v}</text>')
        return out

    flips = [cw_bits[i] != rx_bits[i] for i in range(24)]
    parts.append(f'<text x="60" y="62" fill="#8b949e" font-size="11">'
                 f'sent -> corrupted (red = flipped bits) -> decoded, all recovered</text>')
    parts += row("sent", cw_bits, 75)
    parts += row("recv", rx_bits, 130, highlight=flips)
    parts += row("fixed", bits(corrected), 185)
    parts.append(f'<text x="60" y="245" fill="#06d6a0" font-size="12">'
                 f'{_popcount(codeword ^ received)} errors introduced, '
                 f'{"all corrected" if corrected == codeword else "uncorrectable"} '
                 f'(fixed {nc if nc>=0 else 0})</text>')
    parts.append(f'<text x="60" y="270" fill="#8b949e" font-size="11">'
                 f'blue = 1 bits, dark = 0 bits; the "fixed" row equals the original "sent" row</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
