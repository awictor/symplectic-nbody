"""Demo: Reed-Solomon error correction on a byte string.

Encodes a text message with Reed-Solomon parity, corrupts several bytes (as a scratch or a burst of
noise would), and recovers the original exactly -- showing how many errors it can fix and drawing
the codeword as a byte grid with the corrupted and repaired positions marked.

    python examples/reed_solomon_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import reed_solomon as rs  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    text = "REED-SOLOMON"
    msg = [ord(c) for c in text]
    nsym = 8                         # 8 parity bytes -> correct up to 4 errors
    t = nsym // 2
    cw = rs.rs_encode(msg, nsym)

    print("Reed-Solomon error correction over GF(256)\n")
    print(f"  message: {text!r}  ({len(msg)} bytes)")
    print(f"  parity : {nsym} bytes  ->  corrects up to t = {t} byte-errors per block")
    print(f"  codeword ({len(cw)} bytes): {' '.join(f'{b:02x}' for b in cw)}\n")

    # corrupt t bytes -- a burst of noise
    corrupt = {2: 0x5a, 3: 0xff, 9: 0x01, 13: 0x7e}
    received = cw[:]
    for pos, xor in corrupt.items():
        received[pos] ^= xor
    print(f"  corrupting {len(corrupt)} bytes at positions {sorted(corrupt)} "
          f"(a scratch / noise burst):")
    print(f"  received ({len(received)} bytes): {' '.join(f'{b:02x}' for b in received)}")

    synd = rs._syndromes(received, nsym)
    print(f"  syndromes (nonzero => error detected): "
          f"{'all zero' if max(synd) == 0 else 'nonzero -> ' + str(synd[:4]) + '...'}\n")

    corrected, n_fixed = rs.rs_decode(received, nsym)
    recovered = "".join(chr(b) for b in rs.message_from_codeword(corrected, nsym))
    print(f"  decoded: corrected {n_fixed} byte-errors")
    print(f"  recovered codeword matches original: {corrected == cw}")
    print(f"  recovered message: {recovered!r}  (correct: {recovered == text})\n")

    # show the limit: t+1 errors is uncorrectable
    over = cw[:]
    for p in (1, 4, 7, 10, 14):      # 5 > t = 4 errors
        over[p] ^= 0x33
    try:
        rs.rs_decode(over, nsym)
        status = "silently mis-decoded (should not happen)"
    except ValueError:
        status = "correctly flagged as uncorrectable"
    print(f"  {len(msg)}+{nsym} block with {t+1} errors (one past the limit): {status}.\n")

    print("  A message is the coefficients of a polynomial over GF(256); parity makes the codeword")
    print("  divisible by the generator. Corruption breaks that divisibility, and the syndromes ->")
    print("  Berlekamp-Massey -> Chien -> Forney pipeline recovers both the error positions and")
    print("  their values. This is the code in QR codes, CDs, and deep-space telemetry.")

    _svg(os.path.join(outdir, "reed_solomon.svg"), cw, received, corrected, set(corrupt), len(msg))
    print(f"\n  wrote {os.path.join(outdir, 'reed_solomon.svg')}")


def _svg(path, original, received, corrected, corrupt_positions, msg_len, width=760, height=360):
    n = len(original)
    cell = min(52, (width - 90) // n)
    x0 = 45
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Reed-Solomon: corrupted bytes (red) located and repaired (green)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'top: received codeword with errors; bottom: after decoding -- '
        f'message bytes | parity bytes</text>',
    ]

    def row(y, data, label, mark_fixed=False):
        parts.append(f'<text x="20" y="{y+cell/2+4:.1f}" fill="#8b949e" font-size="10">{label}</text>')
        for i in range(n):
            cx = x0 + i * cell
            if i in corrupt_positions and not mark_fixed:
                fill, txt = "#5a1d1d", "#ff6b6b"          # corrupted byte
            elif i in corrupt_positions and mark_fixed:
                fill, txt = "#173d2a", "#06d6a0"          # repaired byte
            elif i >= msg_len:
                fill, txt = "#1c2333", "#8b949e"          # parity byte
            else:
                fill, txt = "#161b22", "#e6edf3"          # clean message byte
            parts.append(f'<rect x="{cx:.1f}" y="{y:.1f}" width="{cell-2:.1f}" '
                         f'height="{cell-2:.1f}" fill="{fill}" stroke="#30363d" stroke-width="0.6"/>')
            parts.append(f'<text x="{cx+cell/2-1:.1f}" y="{y+cell/2+4:.1f}" fill="{txt}" '
                         f'font-size="10" text-anchor="middle">{data[i]:02x}</text>')
        # message / parity divider
        dx = x0 + msg_len * cell
        parts.append(f'<line x1="{dx-1:.1f}" y1="{y-4:.1f}" x2="{dx-1:.1f}" y2="{y+cell+2:.1f}" '
                     f'stroke="#ffd43b" stroke-width="1.4" stroke-dasharray="3 2"/>')

    row(90, received, "received")
    row(190, corrected, "decoded", mark_fixed=True)

    parts.append(f'<text x="{x0}" y="285" fill="#8b949e" font-size="11">'
                 f'yellow line = message | parity boundary; red = injected error; green = corrected</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
