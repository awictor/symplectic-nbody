"""BCH demo: encode a message, corrupt it with a burst of errors, decode it back (SVG bit strip)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bch import BCH


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


def _bit_strip(s, bits, ox, oy, cell, label, highlight=None, hl_color=RED):
    highlight = highlight or set()
    s.append(f'<text x="{ox}" y="{oy-6}" fill="{TEXT}" font-size="12">{label}</text>')
    for i, b in enumerate(bits):
        x = ox + i * cell
        fill = "#161b22"
        stroke = GRAY
        if i in highlight:
            stroke = hl_color
        txt_col = GREEN if b else "#484f58"
        s.append(f'<rect x="{x}" y="{oy}" width="{cell-2}" height="{cell-2}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if i in highlight else 0.6}"/>')
        s.append(f'<text x="{x+(cell-2)/2:.1f}" y="{oy+cell*0.68:.1f}" fill="{txt_col}" '
                 f'font-size="{cell*0.55:.0f}" text-anchor="middle" font-family="monospace">{b}</text>')


def _svg(path, cw, received, corrected, err_pos, deg, k):
    n = len(cw)
    cell = 30
    W = 60 + n * cell
    H = 260
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="monospace">']
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    s.append(f'<text x="30" y="26" fill="{TEXT}" font-size="15">'
             f'BCH({n},{k}) correcting a burst of {len(err_pos)} bit errors</text>')
    _bit_strip(s, cw, 30, 70, cell, "transmitted codeword")
    _bit_strip(s, received, 30, 130, cell, "received (corrupted)", set(err_pos), RED)
    _bit_strip(s, corrected, 30, 190, cell, "decoded (corrected)", set(err_pos), GREEN)
    # parity/message divider marker under first strip
    xdiv = 30 + deg * cell
    s.append(f'<text x="30" y="{H-8}" fill="{GRAY}" font-size="10">'
             f'positions 0..{deg-1} = parity, {deg}..{n-1} = message (systematic). '
             f'Red = flipped by the channel; green = the decoder found and repaired exactly those bits.</text>')
    s.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(s))


def main(outdir=None):
    # BCH(15, 7) correcting 2 errors
    code = BCH(4, 2)
    rnd = _lcg(20260913)
    msg = [1 if rnd() > 0.5 else 0 for _ in range(code.k)]
    cw = code.encode(msg)

    # inject exactly t errors
    err_pos = []
    while len(err_pos) < code.t:
        p = int(rnd() * code.n) % code.n
        if p not in err_pos:
            err_pos.append(p)
    received = list(cw)
    for p in err_pos:
        received[p] ^= 1

    corrected, ne = code.decode(received)
    decoded_msg, _ = code.decode_message(received)

    lines = []
    lines.append("Binary BCH error correction")
    lines.append("=" * 52)
    lines.append(f"code:            BCH(n={code.n}, k={code.k}), corrects t={code.t} errors")
    lines.append(f"generator g(x):  {''.join(str(b) for b in reversed(code.g))} (degree {code.deg})")
    lines.append(f"field:           GF(2^{code.m}), n = 2^{code.m} - 1 = {code.n}")
    lines.append("")
    lines.append(f"message  ({code.k} bits): {''.join(str(b) for b in msg)}")
    lines.append(f"codeword ({code.n} bits): {''.join(str(b) for b in cw)}")
    lines.append(f"errors at positions:  {sorted(err_pos)}")
    lines.append(f"received:             {''.join(str(b) for b in received)}")
    lines.append("")
    S = code.syndromes(received)
    lines.append(f"syndromes S1..S{2*code.t} (in GF(2^{code.m})): {S}")
    lines.append("  (nonzero => errors detected; Berlekamp-Massey + Chien search locate them)")
    lines.append("")
    lines.append(f"corrected:            {''.join(str(b) for b in corrected)}")
    lines.append(f"errors repaired:      {ne}")
    lines.append(f"decoded message:      {''.join(str(b) for b in decoded_msg)}")
    lines.append(f"match original?       {decoded_msg == msg}")

    text = "\n".join(lines)
    print(text)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        _svg(os.path.join(outdir, "bch.svg"), cw, received, corrected, err_pos, code.deg, code.k)

    return text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
